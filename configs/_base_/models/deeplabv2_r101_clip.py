# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='CNNEncoderDecoder',
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    clip_head=dict(
        type='DLV2HeadCLIP',
        in_channels=512,
        dilations=(6, 12),
        num_classes=19,
        align_corners=False,
        arch_option=1,
        block_depth=2,
        activation='relu',
        n_ctx=16,
        ctx_init='',
        init_cfg=dict(
            type='Normal', std=0.01, override=dict(name='aspp_modules')),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    decode_head=dict(
        type='DLV2Head',
        in_channels=2048,
        in_index=3,
        dilations=(6, 12),
        num_classes=19,
        align_corners=False,
        init_cfg=dict(
            type='Normal', std=0.01, override=dict(name='aspp_modules')),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
