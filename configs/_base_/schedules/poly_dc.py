lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6)