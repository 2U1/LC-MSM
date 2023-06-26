# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.models.decode_heads.isa_head import ISALayer
from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPModule
from .decode_head import BaseDecodeHead
from .segformer_head import MLP
from .sep_aspp_head import DepthwiseSeparableASPPModule
import numpy as np
from collections import OrderedDict
import open_clip
from mmcv.runner import auto_fp16


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        # self.text_projection = clip_model.text_projection
        self.text_projection = nn.Parameter(
            torch.empty(clip_model.transformer.width, 512))
        self.dtype = torch.float32

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(
            dim=-1)] @ self.text_projection

        return x

    def intialize_parameters(self, clip_model):
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection,
                            std=clip_model.transformer.width ** -0.5)


class PromptLearner(nn.Module):
    def __init__(self, n_ctx, classnames, ctx_init, clip_model, tokenizer, vis_dim):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = n_ctx
        dtype = torch.float32
        ctx_dim = clip_model.ln_final.weight.shape[0]
        # cfg_imsize = cfg.INPUT.SIZE[0]

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split())
            prompt = tokenizer(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1+n_ctx, :]
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

        tokenized_prompts = torch.cat(
            [tokenizer(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(
                tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer(
            "token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

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
            pts_i = self.construct_prompts(
                ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts


class depthwise_conv(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(
            1, 1, kernel_size=kernel_size, stride=stride, padding=padding)

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


@HEADS.register_module()
class LsegHeadContextOC(BaseDecodeHead):
    def __init__(self, class_names, tau, arch_option, block_depth, activation, input_resolution, n_ctx, ctx_init, **kwargs):
        self.class_names = class_names
        self.tau = tau
        super(LsegHeadContextOC, self).__init__(
            input_transform='multiple_select', **kwargs)

        clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-H-14', pretrained='laion2b_s32b_b79k')
        tokenizer = open_clip.get_tokenizer('ViT-H-14')

        for idx in range(len(self.class_names)):
            if self.class_names[idx] == 'person':
                self.class_names[idx] = 'pedestrian'
            elif self.class_names[idx] == 'rider':
                self.class_names[idx] = 'driver'
            elif self.class_names[idx] == 'bicycle':
                self.class_names[idx] = 'bike'
            elif self.class_names[idx] == 'motorcycle':
                self.class_names[idx] = 'motorbike'
        last_channel = self.in_channels[-1]
        decoder_params = kwargs['decoder_params']
        embed_dims = decoder_params['embed_dims']
        final_embed_dim = embed_dims

        for p in clip_model.parameters():
            p.requires_grad = False

        self.prompt_learner = PromptLearner(
            n_ctx, self.class_names, ctx_init, clip_model, tokenizer, final_embed_dim)
        self.text_encoder = TextEncoder(clip_model)

        self.block_depth = block_depth

        self.arch_option = arch_option
        if self.arch_option == 1:
            self.spatial_block = bottleneck_block(activation=activation)
        elif self.arch_option == 2:
            self.spatial_block = depthwise_block(activation=activation)

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
            sum(embed_dims), self.channels, **fusion_cfg)

        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1 / self.tau)).exp()
        self.linear_image = nn.Conv2d(
            last_channel, last_channel, input_resolution // 32)

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.convproj = nn.Conv2d(
            self.channels + self.num_classes, self.channels, kernel_size=3, padding=1)
        # self.conv_seg = nn.Conv2d(
        #     self.channels + self.num_classes, self.num_classes, 1)
        # self.conv_seg = nn.Conv2d(self.channels, self.num_classes, 1)

        # self.text_embed_net = nn.Sequential(OrderedDict([
        #     ("linear1", nn.Linear(1024, 768)),
        #     ("relu", nn.ReLU(inplace=True)),
        #     ("linear2", nn.Linear(768, 512))
        # ]))

        # self.text_embed_net = nn.Linear(1024, 512)

        del clip_model

    def forward(self, inputs, cross_modal_fusion=False):
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

        if cross_modal_fusion:
            x = self.cross_modal_fusion(x, img_vector)
            x = self.cls_seg(x)

        else:
            x = self.cls_seg(x)

        return x

    def return_embed(self, inputs):
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
            _c[i] = resize(
                _c[i],
                size=os_size,
                mode='bilinear',
                align_corners=self.align_corners)
            # mmcv.print_log(f'_c{i}: {_c[i].shape}', 'mmseg')

        x = self.fuse_layer(torch.cat(list(_c.values()), dim=1))

        return x

    def cross_modal_fusion(self, feat, img_vector):
        """Classify each pixel."""

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
            text_features = self.text_embed_net(text_features)
            text_features = text_features / \
                text_features.norm(dim=-1, keepdim=True)

            logits_per_image = self.logit_scale * imf_i @ text_features.t()

            logit = logits_per_image.view(
                imf_shape[1], imf_shape[2], -1).permute(2, 0, 1)

            logit = logit.unsqueeze(0)
            if self.arch_option in [1, 2]:
                for _ in range(self.block_depth - 1):
                    logit = self.spatial_block(logit)
                logit = self.spatial_block(logit, False)
            logit = logit.squeeze(0)

            # logits.append(logit)

            concated = torch.cat([imf_orig, logit], dim=0)
            logits.append(concated)

        out = torch.stack(logits)

        out = self.convproj(out)

        # if self.arch_option in [1, 2]:
        #     for _ in range(self.block_depth - 1):
        #         out = self.spatial_block(out)
        #     out = self.spatial_block(out, False)

        # out = self.conv_seg(out)

        return out

    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      seg_weight=None,
                      cross_modal_fusion=False):
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
        seg_logits = self.forward(inputs, cross_modal_fusion)
        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, cross_modal_fusion=False):
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
        return self.forward(inputs, cross_modal_fusion)
