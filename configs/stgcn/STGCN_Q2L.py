_base_ = ['../_init_/lr_schedual.py']
model = dict(
    type='RecognizerGCN_QL',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        tcn_type='unit_tcn', # 'unit_tcn', 'mstcn', 'unit_tcnedge', 'unitmlp', 'msmlp'
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
    neck = dict(type='SimpleNeck', in_channels=256, num_classes=60, mode='GCN'),
    cls_head=dict(
        type='MLdecoderHead',
        token_group='point_wise', #'frame', 'temporal', 'body_part_ntu','body_part_coco','point_wise']
        num_classes=60, 
        # num_of_groups=10,
        in_channels=256,
        dropout_ratio = 0.,
        # decoder_dropout=0.5,
        initial_num_features=256,
        decoder_embedding=512)
        )
 