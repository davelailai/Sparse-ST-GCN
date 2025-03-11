_base_ = ['../_init_/lr_schedual_sparse.py']
sparse = 0.8
model = dict(
    type='RecognizerGCN_sparse',
    backbone=dict(
        type='CTRGCN_sparse',
		sparse_decay = False,
        linear_sparsity =sparse,
        gcn_sparse_ratio=sparse,
        gcn_type = 'unit_ctrgcn_sparse',
        tcn_type = 'mstcn_sparse',
        tcn_sparse_ratio=sparse,
        warm_up = 0,
        # graph_cfg=dict(layout='nturgb+d', mode='stgcn_spatial'),
        # semantic_stage=[1, 2, 3, 4],
        graph_cfg=dict(layout='nturgb+d', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=256),
    train_cfg = dict(panelty=None,lam='gradual'),
    test_cfg = dict(current_epoch=101, max_epoch=100))

optimizer_config = dict(type='SparseOptimizer', grad_clip=None, warmup=101)