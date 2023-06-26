# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
import torch.nn.functional as F

from mmseg.models.decode_heads.isa_head import ISALayer
from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPModule
from .decode_head import BaseDecodeHead
from .segformer_head import MLP
from .sep_aspp_head import DepthwiseSeparableASPPModule
import numpy as np
import clip
from collections import OrderedDict
import math


class TextEncoder(nn.Module):
    def __init__(self, clip_model, embed_dim):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        # self.text_projection = nn.Parameter(torch.empty(clip_model.transformer.width, embed_dim))
        self.token_embedding = clip_model.token_embedding
        self.dtype = clip_model.dtype

        # self.intialize_parameters(clip_model)

    def forward(self, prompts):
        x = self.token_embedding(prompts).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), prompts.argmax(dim=-1)] @ self.text_projection
        return x
    
    def intialize_parameters(self, clip_model):
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std = clip_model.transformer.width ** -0.5)


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

# class Attention(nn.Module):
#     def __init__(self, d_model, input_dims=None):
#         super(Attention, self).__init__()
#         self.d_model = d_model
#         if input_dims is None:
#             self.q_linear = nn.Linear(d_model, d_model)
#             self.k_linear = nn.Linear(d_model, d_model)
#             self.v_linear = nn.Linear(d_model, d_model)

#         else:
#             self.q_linear = nn.Linear(input_dims[0], d_model)
#             self.k_linear = nn.Linear(input_dims[1], d_model)
#             self.v_linear = nn.Linear(input_dims[2], d_model)

#         self.out = nn.Linear(d_model, d_model)

#         self.d_k = d_model
    
#     def forward(self, q, k, v):
#         k = self.k_linear(k)
#         q = self.q_linear(q)
#         v = self.v_linear(v)
        
#         attention_score = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
#         attention_prob = torch.softmax(attention_score, dim=-1)
#         out = torch.matmul(attention_prob, v)

#         out = self.out(out)

#         return out

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

class TwoLayerNet(nn.Module):
    def __init__(self,C_in):
        super().__init__()
        self.conv1=nn.Conv2d(C_in,C_in,1)
        self.conv2=nn.Conv2d(C_in,C_in,1)
    
    def forward(self,feat):
        return torch.tanh(self.conv2(F.relu(self.conv1(feat))))


class LanguagePath(nn.Module):
    def __init__(self,C_in):
        super().__init__()
        self.twoLayerNet=TwoLayerNet(C_in)
    def forward(self, vis_feat, fusion_feat):
        S=self.twoLayerNet(fusion_feat)
        return fusion_feat*S+vis_feat



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
class CLIPHead(BaseDecodeHead):
    def __init__(self, class_names, tau, **kwargs):
        self.class_names = class_names
        self.tau = tau
        super(CLIPHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        # self.clip_text_encoder, _ = clip.load('ViT-B/32' ,jit=False)
        clip_model , _ = clip.load('RN50x16', device='cpu', jit=False)
        text_dim = clip_model.transformer.width
        convert_models_to_fp32(clip_model)
        
        for p in clip_model.parameters():
            p.requires_grad = False

        decoder_params = kwargs['decoder_params']
        embed_dims = decoder_params['embed_dims']
        final_embed_dim = embed_dims
        
        
        self.clip_text_encoder = TextEncoder(clip_model, final_embed_dim)
        self.attn = Attention(final_embed_dim, text_dim)
        self.LG = LanguagePath(final_embed_dim)

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

        # if the class name is 'person', change the name to 'pedestrain'
        # self.class_names = [name if name != 'person' else 'pedestrian' for name in self.class_names]
        for idx in range(len(self.class_names)):
            if self.class_names[idx] == 'person':
                self.class_names[idx] = 'pedestrian'
            elif self.class_names[idx] == 'bicycle':
                self.class_names[idx] = 'bike'
            elif self.class_names[idx] == 'motorcycle':
                self.class_names[idx] = 'motorbike'
        
        prompts = [f"a photo of a {name}." for name in self.class_names]

        self.text = clip.tokenize(prompts)
        
        self.transformer_width = self.clip_text_encoder.transformer.width
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.tau)).exp()

        del clip_model

    def forward(self, inputs):
        x = inputs
        n, _, h, w = x[-1].shape
        # for f in x:
        #     mmcv.print_log(f'{f.shape}', 'mmseg')

        # self.logit_scale = self.logit_scale.to(x.device)

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
        
        x = self.fuse_layer(torch.cat(list(_c.values()), dim=1))
        cross_attentioned_x = self.image_text_attention(x)
        # x = x + cross_attentioned_x
        # x = self.LG(x, cross_attentioned_x)
        out = torch.cat([x, cross_attentioned_x], dim=1)
        out = self.cls_seg(out)

        return x


    def image_text_attention(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)

        text = self.text.to(feat.device)
        # self.logit_scale = self.logit_scale.to(feat.device)

        with torch.no_grad():
            self.clip_text_encoder.eval()
            text_features = self.clip_text_encoder(text)

        imshape = feat.shape
        # image_features = feat.permute(0, 2, 3, 1).reshape(-1, imshape[1])
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # logits_per_image = self.logit_scale * image_features @ text_features.t()
        text_features = torch.stack([text_features]* imshape[0])
        logits_per_image = self.attn(feat, text_features, text_features)
        # logits_per_image *= self.logit_scale  

        # out = logits_per_image.view(imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)
        
        return logits_per_image


    # def cls_seg(self, feat):
    #     """Classify each pixel."""
    #     if self.dropout is not None:
    #         feat = self.dropout(feat)

    #     text = self.text.to(feat.device)
    #     self.logit_scale = self.logit_scale.to(feat.device)

    #     with torch.no_grad():
    #         self.clip_text_encoder.eval()
    #         text_features = self.clip_text_encoder(text)

    #     imshape = feat.shape
    #     image_features = feat.permute(0, 2, 3, 1).reshape(-1, imshape[1])
    #     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    #     logits_per_image = self.logit_scale * image_features @ text_features.t()

    #     out = logits_per_image.view(imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)
        
    #     if self.arch_option in [1,2]:
    #         for _ in range(self.block_depth - 1):
    #             out = self.spatial_block(out)
    #         out = self.spatial_block(out, False)
        
    #     return out