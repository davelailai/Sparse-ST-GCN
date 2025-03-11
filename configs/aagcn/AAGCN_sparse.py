_base_ = ['../_init_/lr_schedual_sparse.py']
sparse = 0.99
model = dict(
    type='RecognizerGCN_sparse',
    backbone=dict(
        type='AAGCN_sparse',
        gcn_adaptive='init',
        sparse_decay = False,
        linear_sparsity = sparse,
        gcn_sparse_ratio=sparse,
        gcn_type = 'unit_aagcn_sparse',
        tcn_type = 'unit_tcn_sparse',
        tcn_sparse_ratio=sparse,
        warm_up = 0,
        # tcn_type='mstc', # unit_tcn, mstcn
        # tcn_channel_annention=True,
        # tcn_add_tcn=True,
        # tcn_merge_after=True,
        # gcn_num_types=5,
        # gcn_reduce=8,
        # gcn_edge_num=15,
        # # graph_cfg=dict(layout='nturgb+d', mode='stgcn_spatial'),
        # num_stages=10,
        # inflate_stages=[5, 8],
        # down_stages=[5, 8],
        graph_cfg=dict(layout='nturgb+d', mode='random', num_filter=3, init_off=.04, init_std=.02),
        # graph_cfg=dict(layout='nturgb+d', mode='stgcn_spatial'),
        # tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1']
        ),
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=256),
    train_cfg = dict(panelty=None,lam='gradual'),
    test_cfg = dict(current_epoch=101, max_epoch=100)
    )
    # neck = dict(type='PretrainNeck', in_channels=256, read_op='attention', num_position=25),
    # neck = dict(type='SimpleNeck', in_channels=256, mode='GCN'),
    # cls_head=dict(
    #     type='ClsHead', num_classes=60, in_channels=256,
    #     #loss_cls=[dict(type='CrossEntropyLoss',loss_weight=1.0)]
    #     ))
#load_from='./work_dirs_pretrain/stgcn_read/stgcn_pyskl_ntu60_xsub_3dkp/j/epoch_20.pth'
# optimizer_config = dict(type='SparseHook', grad_clip=None, sparse =True)
optimizer_config = dict(type='SparseOptimizer', grad_clip=None, warmup=101)
# optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True, sparse='score_only')
# optimizer = []
# optimizer = dict(
#             main=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True, sparse='score_only'),
#             mask=dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005, nesterov=True, sparse='score_only'))