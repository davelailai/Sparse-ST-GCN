_base_ = ['../_init_/lr_schedual.py']
graph = 'nturgb+d'
# graph = 'coco'
model = dict(
    type='RecognizerGCN_QL',
    backbone=dict(
        type='DGSTGCN',
        gcn_type = 'dgphgcn1',
        gcn_ratio=0.125,
        gcn_node_attention = True,
        gcn_edge_attention = True,
        gcn_decompose = True,
        gcn_subset_wise = True,
        # gcn_part_ratio = 1,
        # gcn_add_type = False,
        # gcn_target_specific = False,
        # gcn_ada_attention = False,
        
        # gcn_sub_att = False,
        # gcn_stage = [1,3,5,7,9], #[0,1,2,3], [4,5,6], [7,8,9], [0,2,4,6,8]
        # gcn_num_types=5,
        # gcn_edge_num=15,
        gcn_ctr='T',
        gcn_ada='T',

        # tcn_type='dgmstcn', # 'dgmsmlp', 'dgmstcn'
        # time_encoding=True,
        tcn_type='dgmsmlp',
        tcn_add_tcn=True,
        tcn_merge_after=True,
        # tcn_adaptive = False,
        # tcn_dropout = 0.5,
        graph_cfg=dict(layout=graph, mode='random', num_filter=3, init_off=.04, init_std=.02),
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1']
        ),
    neck = dict(type='SimpleNeck', in_channels=256, num_classes=60, mode='GCN'),
    cls_head=dict(
        type='MLdecoderHead',
        token_group='temporal', #'frame', 'temporal', 'body_part_ntu','body_part_coco','point_wise']
        num_classes=60, 
        # num_of_groups=10,
        in_channels=256,
        dropout_ratio = 0.5,
        # decoder_dropout=0.5,
        initial_num_features=256,
        decoder_embedding=512)
        )
