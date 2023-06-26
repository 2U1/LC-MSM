import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

from ..builder import HEADS
from .aspp_head import ASPPModule
from .decode_head import BaseDecodeHead
import clip
import numpy as np

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.spacial_dim = spacial_dim

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

        cls_pos = self.positional_embedding[0:1, :]
        spatial_pos = F.interpolate(self.positional_embedding[1:,].reshape(1, self.spacial_dim, self.spacial_dim, self.embed_dim).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(self.embed_dim, H*W).permute(1, 0)
        positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)

        x = x + positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        x = x.permute(1, 2, 0)
        global_feat = x[:, :, 0]
        feature_map = x[:, :, 1:].reshape(B, -1, H, W)
        return global_feat, feature_map

class TextEncoder(nn.Module):
    def __init__(self, clip_model, embed_dim):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.text_projection = nn.Parameter(torch.empty(clip_model.transformer.width, embed_dim))
        self.dtype = clip_model.dtype

        self.intialize_parameters(clip_model)

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    def intialize_parameters(self, clip_model):
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std = clip_model.transformer.width ** -0.5)

class PromptLearner(nn.Module):
    def __init__(self, n_ctx, classnames, ctx_init, clip_model, vis_dim):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = n_ctx
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        # cfg_imsize = cfg.INPUT.SIZE[0]


        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split())
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1+n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))
        
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx                     # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts

class depthwise_conv(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # support for 4D tensor with NCHW
        C, H, W = x.shape[1:]
        x = x.reshape(-1, 1, H, W)
        x = self.depthwise(x)
        x = x.view(-1, C, H, W)
        return x


class depthwise_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(depthwise_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x, act=True):
        x = self.depthwise(x)
        if act:
            x = self.activation(x)
        return x


class bottleneck_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(bottleneck_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x, act=True):
        sum_layer = x.max(dim=1, keepdim=True)[0]
        x = self.depthwise(x)
        x = x + sum_layer
        if act:
            x = self.activation(x)
        return x


@HEADS.register_module()
class DLV2HeadCLIP(BaseDecodeHead):

    def __init__(self, class_names, tau, arch_option,
                 block_depth, activation, n_ctx, ctx_init,
                 dilations=(6, 12, 18, 24), **kwargs):
        assert 'channels' not in kwargs
        assert 'dropout_ratio' not in kwargs
        assert 'norm_cfg' not in kwargs
        kwargs['channels'] = 1
        kwargs['dropout_ratio'] = 0
        kwargs['norm_cfg'] = None
        self.tau = tau
        self.class_names = class_names
        super(DLV2HeadCLIP, self).__init__(**kwargs)
        del self.conv_seg
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.aspp_modules = ASPPModule(
            dilations,
            self.in_channels,
            self.in_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)

        clip_model, _ = clip.load('RN50x16', device='cpu', jit=False)

        for idx in range(len(self.class_names)):
            if self.class_names[idx] == 'person':
                self.class_names[idx] = 'pedestrian'
            elif self.class_names[idx] == 'rider':
                self.class_names[idx] = 'driver'
            elif self.class_names[idx] == 'bicycle':
                self.class_names[idx] = 'bike'
            elif self.class_names[idx] == 'motorcycle':
                self.class_names[idx] = 'motorbike'

        for p in clip_model.parameters():
            p.data = p.data.float()
            p.requires_grad = False

        self.prompt_learner = PromptLearner(n_ctx, self.class_names, ctx_init, clip_model, self.in_channels)
        self.text_encoder = TextEncoder(clip_model, self.in_channels)

        self.block_depth = block_depth

        self.arch_option = arch_option
        
        if self.arch_option ==1:
            self.spatial_block = bottleneck_block(activation=activation)
        elif self.arch_option ==2:
            self.spatial_block = depthwise_block(activation=activation)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.tau)).exp()
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.attention_pool = AttentionPool2d(16, 2048, 32, 512)
        
        # self.conv_seg = nn.Conv2d(self.in_channels + self.num_classes, self.num_classes, 1)

    def forward(self, inputs):
        """Forward function."""
        # for f in inputs:
        #     mmcv.print_log(f'{f.shape}', 'mmseg')
        x = self._transform_inputs(inputs)
        global_feature, visual_embedding = self.attention_pool(x)
        out = self.cls_seg(visual_embedding, global_feature)
        
        return out

    def cls_seg(self, feat, img_vector):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)

        tokenized_prompts = self.tokenized_prompts.to(feat.device)

        self.logit_scale = self.logit_scale.to(feat.device)

        prompts = self.prompt_learner(img_vector)
        
        logits = []
        
        for pts_i, imf_i in zip(prompts, feat):
            imf_shape = imf_i.shape
            imf_orig = imf_i
            imf_i = imf_i.permute(1, 2, 0).reshape(-1, imf_shape[0])
            imf_i = imf_i / imf_i.norm(dim=-1, keepdim=True)
            
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            logits_per_image = self.logit_scale * imf_i @ text_features.t()
            
            
            logit = logits_per_image.view(imf_shape[1], imf_shape[2], -1).permute(2, 0, 1)
            
            # logit = logit.unsqueeze(0)
            # if self.arch_option in [1,2]:
            #     for _ in range(self.block_depth - 1):
            #         logit = self.spatial_block(logit)
            #     logit = self.spatial_block(logit, False)
            # logit = logit.squeeze(0)
            
            # concated = torch.cat([imf_orig, logit], dim=0)
            # logits.append(concated)
            logits.append(logit)

        out = torch.stack(logits)

        # out = self.convproj(out)
        
        if self.arch_option in [1,2]:
            for _ in range(self.block_depth - 1):
                out = self.spatial_block(out)
            out = self.spatial_block(out, False)

        # if self.dropout is not None:
        #     out = self.dropout(out)

        # out = self.conv_seg(out)
        
        return out
