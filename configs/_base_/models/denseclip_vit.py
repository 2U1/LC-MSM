
_base_ = ['denseclip_r50.py']

model = dict(
    type='DenseCLIP',
    pretrained='pretrained/ViT-B-16.pt',
    context_length=5,
    text_head=False,
    text_dim=512,
    score_concat_index=2,
    backbone=dict(
        type='CLIPVisionTransformer',
        patch_size=16,
        width=768,
        output_dim=512,
        get_embeddings=True,
        drop_path_rate=0.1,
        layers=12,
        input_resolution=512,
        style='pytorch'),
    text_encoder=dict(
        type='CLIPTextContextEncoder',
        context_length=13,
        embed_dim=512,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        style='pytorch'),
    context_decoder=dict(
        type='ContextDecoder',
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=3,
        visual_dim=512,
        dropout=0.1,
        outdim=512,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[768, 768, 768+19, 768],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        num_classes=19,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(426, 426)),
    test_cfg=dict(mode='whole')
)