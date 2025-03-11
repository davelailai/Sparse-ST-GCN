_base_ = ['../_init_/lr_schedual_sparse.py']
graph = 'nturgb+d'
# graph = 'coco'
# sparse = 0.99
head_num=4
channel=3
model = dict(
    type='RecognizerGCN_assemble',
    backbone=dict(
        type='Assemble_sparse_New',
        # model_list=['ST-GCN', 'AA-GCN', 'CTR-GCN','DG-GCN'],
        # model_list=['ST-GCN']*head_num,
        model_list=['DG-GCN']*head_num,
        # model_list=['CTR-GCN']*head_num,
        # model_list = ['DG-GCN']*head_num,
        # sparse_ratio=[0.0]*head_num,
        # sparse_ratio=[0.8, 0.8, 0.8, 0.8],
        sparse_ratio=[0.6, 0.8,0.95,0.99]*4,
        # sparse_ratio=[0.0]*head_num,
        ST_kwargs = dict(
            # type='STGCN_sparse',
            gcn_adaptive='init',
            # sparse_decay = False,
            # linear_sparsity = sparse,
            # gcn_sparse_ratio=sparse,
            gcn_type = 'unit_gcn_sparse',
            tcn_type = 'unit_tcn_sparse',
            # tcn_sparse_ratio=sparse
            ),
        AA_kwargs = dict(
            # type='AAGCN_sparse',
            gcn_adaptive='init',
            # sparse_decay = False,
            # linear_sparsity = sparse,
            # gcn_sparse_ratio=sparse,
            gcn_type = 'unit_aagcn_sparse',
            tcn_type = 'unit_tcn_sparse',
            # tcn_sparse_ratio=sparse
            ),
        CTR_kwargs = dict(
            #  type='CTRGCN_sparse',
            # sparse_decay = False,
            # linear_sparsity =sparse,
            # gcn_sparse_ratio=sparse,
            gcn_type = 'unit_ctrgcn_sparse',
            tcn_type = 'mstcn_sparse',
            # tcn_sparse_ratio=sparse
            ),
        DG_kwargs = dict(
            # type='DGSTGCN_sparse',
            gcn_type = 'dggcn_sparse',
            gcn_ratio=0.125,
            # sparse_decay = False,
            # linear_sparsity = sparse,
            # gcn_sparse_ratio=sparse,  
            gcn_ctr='T',
            gcn_ada='T',
            tcn_type = 'mstcn_sparse',
            # tcn_sparse_ratio=sparse,
            tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1']),

   
        warm_up = 80,
        graph_cfg=dict(layout=graph, mode='random', num_filter=channel, init_off=.04, init_std=.02),
        ),
   cls_head=dict(type='Assemble_GCNHead', 
                 head=head_num, 
                 num_classes=60, 
                 in_channels=256,
                 head_ada=False,
                #  head_conv=True,
                #  KL_div=True,
                #  each_loss=False
                 ),
   train_cfg = dict(panelty='GSGL',lam='gradual'),
   test_cfg = dict(current_epoch=151, max_epoch=150))
        #loss_cls=[dict(type='CrossEntropyLoss',loss_weight=1.0)])
        
optimizer_config = dict(type='SparseOptimizer', grad_clip=dict(max_norm=45, norm_type=2), warmup=151)