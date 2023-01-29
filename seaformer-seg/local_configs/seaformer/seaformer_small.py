# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

model_cfgs = dict(
        cfg1=[
            # k,  t,  c, s
            [3, 1, 16, 1],  
            [3, 4, 24, 2], 
            [3, 3, 24, 1]], 
        cfg2=[
            [5, 3, 48, 2],  
            [5, 3, 48, 1]],  
        cfg3=[
            [3, 3, 96, 2],  
            [3, 3, 96, 1]],
        cfg4=[
            [5, 4, 160, 2]],  
        cfg5=[
            [3, 6, 192, 2]],  
        channels=[16, 24, 48, 96, 160, 192],
        depths=[3, 3],
        key_dims=[16, 24],
        emb_dims=[160, 192],
        num_heads=6,
        drop_path_rate=0.1,
)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='SeaFormer',
        cfgs=[model_cfgs['cfg1'], model_cfgs['cfg2'], model_cfgs['cfg3'], model_cfgs['cfg4'], model_cfgs['cfg5']],
        channels=model_cfgs['channels'],
        emb_dims=model_cfgs['emb_dims'],
        key_dims=model_cfgs['key_dims'],
        depths=model_cfgs['depths'],
        num_heads=model_cfgs['num_heads'],
        drop_path_rate=model_cfgs['drop_path_rate'],
        norm_cfg=norm_cfg,
        init_cfg=dict(type='Pretrained', checkpoint='modelzoos/classification/SeaFormer_S.pth')
    ),
    decode_head=dict(
        type='LightHead',
        in_channels=[48, 160, 192],
        in_index=[0, 1, 2],
        channels=128,
        dropout_ratio=0.1,
        embed_dims=[96, 128],
        num_classes=150,
        is_dw=True,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

