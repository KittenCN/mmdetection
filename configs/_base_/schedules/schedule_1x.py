# optimizer
opti_rate = 0.01
optimizer = dict(type='SGD', lr=0.02 * opti_rate, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001 * opti_rate,
    step=[9996, 9999]
    )
runner = dict(type='EpochBasedRunner', max_epochs=10000)
