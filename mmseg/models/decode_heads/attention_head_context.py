# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.models.decode_heads.isa_head import ISALayer
from mmseg.ops import resize
from mmseg.models.utils import label_to_one_hot_label
from ..builder import HEADS
from .aspp_head import ASPPModule
from .decode_head import BaseDecodeHead
from .segformer_head import MLP
from .sep_aspp_head import DepthwiseSeparableASPPModule
import numpy as np
from collections import OrderedDict
import math
import clip
import torch.nn.functional as F

from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy

def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:

        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}_{name}'] = value

    return outputs


class TextEncoder(nn.Module):
    def __init__(self, clip_model, embed_dim):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        # self.text_projection = nn.Parameter(torch.empty(clip_model.transformer.width, embed_dim))
        self.dtype = clip_model.dtype

        # self.intialize_parameters(clip_model)

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        x = x @ self.text_projection

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

        del clip_model
    
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

class Attention(nn.Module):
    def __init__(self, img_dim, text_dim):
        super(Attention, self).__init__()

        self.d_img = img_dim
        self.d_txt = text_dim

        self.q_embed = nn.Conv2d(self.d_img, self.d_img, kernel_size=1)
        self.k_embed = nn.Conv1d(self.d_txt, self.d_img, kernel_size=1)
        self.v_embed = nn.Conv1d(self.d_txt, self.d_img, kernel_size=1)

        self.visual_embed = nn.Conv2d(self.d_img, self.d_img, kernel_size=1)
        self.fused_embed = nn.Conv2d(self.d_img, self.d_img, kernel_size=1)

        self.out = nn.Conv2d(self.d_img, self.d_img, kernel_size=1)

        self.instance_q = nn.InstanceNorm2d(self.d_img, affine=True)
        self.instance_w = nn.InstanceNorm2d(self.d_img, affine=True)

        self.d_k = img_dim
    
    def forward(self, vis, txt_k, txt_v):
        
        img_shape = vis.shape

        k = txt_k.permute(0, 2, 1)
        v = txt_v.permute(0, 2, 1)

        visual_feat = F.relu(self.visual_embed(vis.clone()))
        
        q = self.instance_q(self.q_embed(vis.clone()))
        k = self.k_embed(k)
        v = self.v_embed(v)

        q = q.view(img_shape[0], img_shape[1], -1).permute(0, 2, 1)
        
        attention_score = torch.matmul(q, k) / np.sqrt(self.d_k)
        attention_prob = torch.softmax(attention_score, dim=-1)
        v = v.permute(0, 2, 1)
        fused = torch.matmul(attention_prob, v)

        fused = fused.permute(0,2,1).view(img_shape[0], img_shape[1], img_shape[2], img_shape[3])
        fused = self.instance_w(self.fused_embed(fused))

        out = F.relu(self.out(visual_feat * fused))

        return out

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

class ConvProjection(nn.Module):
    def __init__(self, c_in):
        super(ConvProjection, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_in, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(c_in, c_in, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(c_in)
        self.norm2 = nn.BatchNorm2d(c_in)

    def forward(self, x):
        return F.relu(self.norm2(self.conv2(F.relu(self.norm1(self.conv1(x))))))


class ASPPWrapper(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 sep,
                 dilations,
                 pool,
                 norm_cfg,
                 act_cfg,
                 align_corners,
                 context_cfg=None):
        super(ASPPWrapper, self).__init__()
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.align_corners = align_corners
        if pool:
            self.image_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        else:
            self.image_pool = None
        if context_cfg is not None:
            self.context_layer = build_layer(in_channels, channels,
                                             **context_cfg)
        else:
            self.context_layer = None
        ASPP = {True: DepthwiseSeparableASPPModule, False: ASPPModule}[sep]
        self.aspp_modules = ASPP(
            dilations=dilations,
            in_channels=in_channels,
            channels=channels,
            norm_cfg=norm_cfg,
            conv_cfg=None,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + int(pool) + int(bool(context_cfg))) * channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        if self.image_pool is not None:
            aspp_outs.append(
                resize(
                    self.image_pool(x),
                    size=x.size()[2:],
                    mode='bilinear',
                    align_corners=self.align_corners))
        if self.context_layer is not None:
            aspp_outs.append(self.context_layer(x))
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)

        output = self.bottleneck(aspp_outs)
        return output


def build_layer(in_channels, out_channels, type, **kwargs):
    if type == 'id':
        return nn.Identity()
    elif type == 'mlp':
        return MLP(input_dim=in_channels, embed_dim=out_channels)
    elif type == 'sep_conv':
        return DepthwiseSeparableConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'conv':
        return ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'aspp':
        return ASPPWrapper(
            in_channels=in_channels, channels=out_channels, **kwargs)
    elif type == 'rawconv_and_aspp':
        kernel_size = kwargs.pop('kernel_size')
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2),
            ASPPWrapper(
                in_channels=out_channels, channels=out_channels, **kwargs))
    elif type == 'isa':
        return ISALayer(
            in_channels=in_channels, channels=out_channels, **kwargs)
    else:
        raise NotImplementedError(type)

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        # p.grad.data = p.grad.data.float() 

@HEADS.register_module()
class AttentionHeadContext(BaseDecodeHead):
    def __init__(self, class_names, tau, input_resolution, n_ctx, ctx_init, loss_decode2, **kwargs):
        self.class_names = class_names
        self.tau = tau
        super(AttentionHeadContext, self).__init__(
            input_transform='multiple_select', **kwargs)

        # clip_model, _ = clip.load('ViT-B/16', device='cpu', jit=False)
        clip_model, _ = clip.load('RN50x16', device='cpu', jit=False)
        convert_models_to_fp32(clip_model)

        self.text_dim = clip_model.transformer.width

        for idx in range(len(self.class_names)):
            if self.class_names[idx] == 'person':
                self.class_names[idx] = 'pedestrian'
            elif self.class_names[idx] == 'bicycle':
                self.class_names[idx] = 'bike'
            elif self.class_names[idx] == 'motorcycle':
                self.class_names[idx] = 'motorbike'
        last_channel = self.in_channels[-1]
        decoder_params = kwargs['decoder_params']
        embed_dims = decoder_params['embed_dims']
        final_embed_dim = embed_dims

        self.loss_decode2 = build_loss(loss_decode2)

        for p in clip_model.parameters():
            p.requires_grad = False

        self.prompt_learner = PromptLearner(n_ctx, self.class_names, ctx_init, clip_model, final_embed_dim)
        self.text_encoder = TextEncoder(clip_model,final_embed_dim)

        self.attn = Attention(final_embed_dim, self.text_dim)
        self.convproj = ConvProjection(self.channels)

        assert not self.align_corners
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(self.in_index)
        embed_cfg = decoder_params['embed_cfg']
        embed_neck_cfg = decoder_params['embed_neck_cfg']
        if embed_neck_cfg == 'same_as_embed_cfg':
            embed_neck_cfg = embed_cfg
        fusion_cfg = decoder_params['fusion_cfg']
        for cfg in [embed_cfg, embed_neck_cfg, fusion_cfg]:
            if cfg is not None and 'aspp' in cfg['type']:
                cfg['align_corners'] = self.align_corners

        self.embed_layers = {}
        for i, in_channels, embed_dim in zip(self.in_index, self.in_channels,
                                             embed_dims):
            if i == self.in_index[-1]:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_neck_cfg)
            else:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_cfg)
        self.embed_layers = nn.ModuleDict(self.embed_layers) 

        self.fuse_layer = build_layer(
            sum(embed_dims), self.channels//2, **fusion_cfg)
        
        self.linear_image = nn.Conv2d(last_channel, last_channel, input_resolution // 32)

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.conv_seg = nn.Conv2d(self.channels, 2, kernel_size=1)

        del clip_model


    def forward(self, inputs):
        x = inputs
        n, _, h, w = x[-1].shape
        # for f in x:
        #     mmcv.print_log(f'{f.shape}', 'mmseg')

        os_size = x[0].size()[2:]
        _c = {}
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}', 'mmseg')
            _c[i] = self.embed_layers[str(i)](x[i])
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous()\
                    .reshape(n, -1, x[i].shape[2], x[i].shape[3])
            # mmcv.print_log(f'_c{i}: {_c[i].shape}', 'mmseg')
            if _c[i].size()[2:] != os_size:
                # mmcv.print_log(f'resize {i}', 'mmseg')
                _c[i] = resize(
                    _c[i],
                    size=os_size,
                    mode='bilinear',
                    align_corners=self.align_corners)
        img_vector = self.linear_image(x[-1]).view(n, -1)
        x = self.fuse_layer(torch.cat(list(_c.values()), dim=1))

        out = self.cls_seg(x, img_vector)

        return out


    def cls_seg(self, feat, img_vector):
        """Classify each pixel."""
        tokenized_prompts = self.tokenized_prompts.to(feat.device)

        prompts = self.prompt_learner(img_vector)

        out = []
        
        for pts_i, imf_i in zip(prompts, feat):

            imf_i = imf_i.unsqueeze(0)
            imf_shape = imf_i.shape
            
            text_features = self.text_encoder(pts_i, tokenized_prompts)

            logit_per_image = []
            
            for i in range(self.num_classes):
                class_text = text_features[i, :, :]
                class_text = class_text.unsqueeze(0)

                class_attention = self.attn(imf_i, class_text, class_text)
                # residual_feat = imf_i + class_attention
                
                # fused_feat = torch.cat([residual_feat, class_attention], dim=1)
                fused_feat = torch.cat([imf_i, class_attention], dim=1)

                del class_attention

                fused_feat = self.convproj(fused_feat)
                
                class_per_logit = self.conv_seg(fused_feat)
                
                logit_per_image.append(class_per_logit.squeeze(0))
            
            logit_per_image = torch.stack(logit_per_image)
            
            out.append(logit_per_image)
        
        out = torch.stack(out)

        return out

    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      seg_weight=None):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        total_losses = dict()
        seg_logits = self.forward(inputs)
        gt_semantic_seg = gt_semantic_seg.squeeze(1)
        one_hot_gt = label_to_one_hot_label(gt_semantic_seg, self.num_classes, ignore_index=self.ignore_index)


        temp_loss = 0
        
        acc_map, _ = torch.max(torch.softmax(seg_logits, dim=2), dim=2)
        
        acc_map = resize(
            input=acc_map,
            size=gt_semantic_seg.shape[1:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        acc_seg = accuracy(acc_map, gt_semantic_seg)

        del acc_map
        
        for i in range(self.num_classes):
            seg_map = seg_logits[:,i,:,:,:]
            class_gt = one_hot_gt[:,i,:,:]
            class_gt = class_gt.unsqueeze(1)
            
            losses = self.losses(seg_map, class_gt, seg_weight)

            temp_loss += losses['loss_seg']

        # total_losses['loss_seg'] = temp_loss / self.num_classes
        total_losses['loss_seg'] = temp_loss
        total_losses['acc_seg'] = acc_seg

        return total_losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        logits = self.forward(inputs)

        B, C, _, H, W = logits.shape

        outputs = torch.zeros((B, C, H, W), device='cuda')

        for idx in range(self.num_classes):
            class_logit = logits[:, idx, :, :, :] 
            mask_prob, _ = torch.max(torch.softmax(class_logit,dim=1), dim=1)
            outputs[:, idx, :, :] = mask_prob

        return outputs


    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label, seg_weight=None):
        """Compute segmentation loss."""
        loss1 = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        seg_label = seg_label.squeeze(1)
        loss1['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        # loss1['acc_seg'] = accuracy(seg_logit, seg_label)

        loss2 = dict()
        loss2['loss_seg'] = self.loss_decode2(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)

        loss = dict()

        loss['loss_seg'] = loss1['loss_seg'] + loss2['loss_seg']
        # loss['loss_seg'] = loss1['loss_seg']
        # loss['acc_seg'] = loss1['acc_seg']

        return loss
        
