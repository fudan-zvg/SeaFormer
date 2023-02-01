_base_ = [
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py',
    './seaformer_tiny.py'
]

optimizer = dict(_delete_=True, type='AdamW', lr=0.0005, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_emb': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=4)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)
find_unused_parameters=True
