# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/tigda_mic_sepaspp.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/uda_gta_to_cityscapes_512x512.py',
    # Basic UDA Self-Training
    '../_base_/uda/dacs_tigda.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 0
# Modifications to Basic UDA
uda = dict(
    mask_lambda=1,
    mask_generator=dict(
        type='block', mask_ratio=0.5, mask_block_size=32, _delete_=True),
    # Increased Alpha
    alpha=0.999,
    # Thing-Class Feature Distance
    imnet_feature_dist_lambda=0.009,
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    imnet_feature_dist_scale_min_ratio=0.75)
data = dict(
    train=dict(
        # Rare Class Sampling
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)))
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
lr_config = dict(power=1.1)
n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=60000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=20000, max_keep_ckpts=1)
evaluation = dict(interval=2000, metric='mIoU')
# Meta Information for Result Analysis
name = 'TigDA_VLC_MIC'
exp = 'journal'
name_dataset = 'gta2cityscapes'
name_architecture = 'TigDA'
name_encoder = 'mitb5'
name_decoder = 'TigDA'
name_uda = 'dacs_a999_fd_things_rcs0.01_cpl'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
descripion = 'masksize=32, maskratio=0.7, imnet_feature=0.0009, imnet_ratio=0.75, lr_config_power=1.1'
