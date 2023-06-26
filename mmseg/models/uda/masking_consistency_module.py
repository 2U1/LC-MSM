# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import random

import torch
from torch.nn import Module
from mmseg.models.utils.dacs_transforms import get_mean_std, strong_transform
from mmseg.models.utils.masking_transforms import build_mask_generator


class MaskingConsistencyModule(Module):

    def __init__(self, cfg):
        super(MaskingConsistencyModule, self).__init__()

        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.mask_lambda = cfg['mask_lambda']
        self.mask_gen = build_mask_generator(cfg['mask_generator'])

    def __call__(self,
                 model,
                 img,
                 img_metas,
                 gt_semantic_seg,
                 text_features=None,
                 pseudo_weight=None):
        self.debug_output = {}
        dev = img.device
        means, stds = get_mean_std(img_metas, dev)

        masked_img = img
        if gt_semantic_seg.dim() == 3:
            masked_lbl = gt_semantic_seg.unsqueeze(1)
        else:
            masked_lbl = gt_semantic_seg
        masked_seg_weight = pseudo_weight

        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1),
            'mean': means[0].unsqueeze(0),
            'std': stds[0].unsqueeze(0)
        }
        masked_img, _ = strong_transform(
            strong_parameters, data=masked_img.clone())

        # Apply masking to image
        masked_img = self.mask_gen.mask_image(masked_img)

        if text_features is not None:
            masked_loss = model.forward_with_clip(
                masked_img,
                img_metas,
                masked_lbl,
                text_features=text_features,
                seg_weight=masked_seg_weight)

        else:
            masked_loss = model.forward_with_clip(
                masked_img,
                img_metas,
                masked_lbl,
                seg_weight=masked_seg_weight)

        if self.mask_lambda != 1:
            masked_loss['decode.loss_seg'] *= self.mask_lambda

        return masked_loss
