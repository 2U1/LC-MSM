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
import open_clip
from mmcv.runner import auto_fp16
from mmseg.core import add_prefix


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
class VLCHeadLarge(BaseDecodeHead):
    def __init__(self, class_names, tau, arch_option, block_depth, activation, **kwargs):
        self.class_names = class_names
        self.tau = tau
        super(VLCHeadLarge, self).__init__(
            input_transform='multiple_select', **kwargs)

        self.clip_model, _, _ = open_clip.create_model_and_transforms(
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

        prompts = [f"a photo of a {name}." for name in self.class_names]

        self.text = tokenizer(prompts)

        decoder_params = kwargs['decoder_params']
        embed_dims = decoder_params['embed_dims']

        for p in self.clip_model.parameters():
            p.requires_grad = False

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

        self.text_embed_net = nn.Linear(1024, 512)

    @auto_fp16()
    def forward(self, inputs, return_cmf=False):
        x = inputs
        n, _, h, w = x[-1].shape

        cross_modal_feat = self.vision_language_connect(x[-1])
        x[-1] = torch.cat([x[-1], cross_modal_feat], dim=1)

        os_size = x[0].size()[2:]
        _c = {}
        for i in self.in_index:
            _c[i] = self.embed_layers[str(i)](x[i])
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous()\
                    .reshape(n, -1, x[i].shape[2], x[i].shape[3])
            if _c[i].size()[2:] != os_size:
                _c[i] = resize(
                    _c[i],
                    size=os_size,
                    mode='bilinear',
                    align_corners=self.align_corners)

        x = self.fuse_layer(torch.cat(list(_c.values()), dim=1))
        x = self.cls_seg(x)

        if return_cmf:
            return x, cross_modal_feat

        else:
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

        losses = dict()

        seg_logits, cross_modal_feat = self.forward(inputs, return_cmf=True)
        decode_loss = self.losses(seg_logits, gt_semantic_seg, seg_weight)
        aux_loss = self.losses(cross_modal_feat/self.tau,
                               gt_semantic_seg, seg_weight)

        aux_loss['loss_seg'] *= 0.4

        losses.update(decode_loss)
        losses.update(add_prefix(aux_loss, 'vl_feat'))

        return losses

    def vision_language_connect(self, feat):

        self.logit_scale = self.logit_scale.to(feat.device)
        text = self.text.to(feat.device)

        feat_shape = feat.shape

        imf = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
        imf = imf / imf.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            text_features = self.clip_model.encode_text(text)

        text_features = self.text_embed_net(text_features)
        text_features = text_features / \
            text_features.norm(dim=-1, keepdim=True)

        # cross_modal_feat = self.logit_scale * imf @ text_features.t()
        cross_modal_feat = imf @ text_features.t()

        cross_modal_feat = cross_modal_feat.view(feat_shape[0], feat_shape[2],
                                                 feat_shape[3], -1).permute(0, 3, 1, 2)

        if self.arch_option in [1, 2]:
            for _ in range(self.block_depth - 1):
                cross_modal_feat = self.spatial_block(cross_modal_feat)
            cross_modal_feat = self.spatial_block(cross_modal_feat, False)

        return cross_modal_feat
