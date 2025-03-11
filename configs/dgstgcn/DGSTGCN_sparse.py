_base_ = ['../_init_/lr_schedual_sparse.py']
graph = 'nturgb+d'
# graph = 'coco'
sparse = 0.34
model = dict(
    type='RecognizerGCN_sparse',
    backbone=dict(
        type='DGSTGCN_sparse',
        gcn_type = 'dggcn_sparse',
        gcn_ratio=0.125,
        sparse_decay = False,
        linear_sparsity = sparse,
        gcn_sparse_ratio=sparse,
       
        warm_up = 60,
        graph_cfg=dict(layout=graph, mode='random', num_filter=8, init_off=.04, init_std=.02),
        # graph_cfg=dict(layout='nturgb+d', mode='stgcn_spatial'),
        gcn_ctr='T',
        gcn_ada='T',
        # tcn_type='dgmstcn', # 'dgmsmlp', 'dgmstcn'
        tcn_type = 'mstcn_sparse',
        tcn_sparse_ratio=sparse,
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1']
        ),
   cls_head=dict(type='GCNHead', num_classes=60, in_channels=256),
   train_cfg = dict(panelty='GSGL',lam='gradual'),
   test_cfg = dict(current_epoch=101, max_epoch=100))
        #loss_cls=[dict(type='CrossEntropyLoss',loss_weight=1.0)])
        
optimizer_config = dict(type='SparseOptimizer', grad_clip=None, warmup=101)