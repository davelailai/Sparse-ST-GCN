_base_ = ['../_init_/lr_schedual_gc.py']
model = dict(
    type='RecognizerGCN_GC',
    backbone=dict(
        type='GCGCN_component',
        in_channels=3,
        causal_channel = 100,
        feature_update = [64,128,1],
        # feature_update = None,
        feature_hidden= [10,1],
        time_len = 9,
        time_serious=25),
    cls_head=dict(type='GCHead', num_classes=60, in_channels=625))

optimizer_config = dict(type='GCOptimizer', grad_clip=None)