_base_ = ['../_init_/lr_schedual.py']
graph = 'nturgb+d'
model = dict(
    type='RecognizerGCN_GT',
    backbone=dict(
        type='GTGCN',
        gcn_edge_attention = True,
        gcn_target_specific = True,
        gcn_global_attention= True,
        gcn_adaptive='init',  # init, offset, importance
        tcn_type='unit_tcn', # unit_tcn, mstcn
        # tcn_channel_annention=True,
        gcn_num_types=5,
        gcn_reduce=8,
        gcn_edge_num=15,
        # graph_cfg=dict(layout='nturgb+d', mode='stgcn_spatial'),
        num_stages=10,
        inflate_stages=[5, 8],
        down_stages=[5, 8],
        graph_cfg=dict(layout=graph, mode='random', num_filter=3, init_off=.04, init_std=.02),
        # tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1']
        ),
    # neck = dict(type='PretrainNeck', in_channels=256, read_op='attention', num_position=25),
    neck = dict(type='SemanticNeck', in_channels=256, mode='GCN'),
    cls_head=dict(
        type='ClsHead', num_classes=60, in_channels=256,
        # loss_cls=[dict(type='CrossEntropyLoss',loss_weight=1.0)]
        ))




