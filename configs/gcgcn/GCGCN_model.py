_base_ = ['../_init_/lr_schedual_gc.py']
model = dict(
    type='RecognizerGCN_GC',
    backbone=dict(
         type='GCGCN',
		 in_channels=3,
         mid_channels=50, 
         feature_hidden= [10, 100, 10, 1],
         causal_hidden = [100],
         ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4)],
         time_serious=25),
    cls_head=dict(type='GCHead', num_classes=60, in_channels=625))

optimizer_config = dict(type='GCOptimizer', grad_clip=None)