_base_ = ['../STGCN_Q2L.py']
work_dir = './work_dirs_Q2L/class_group_branch2/stgcn/stgcn_pyskl_ntu120_xset_3dkp/j_frame'
# load_from='./work_dirs_sparse/stgcn_base/stgcn_pyskl_ntu120_xset_3dkp/j/latest.pth'
model = dict(
    cls_head=dict(num_classes=120, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = 'data/nturgbd/ntu120_3danno.pkl'
modality = 'j'
train_pipeline = [
    dict(type='PreNormalize3D', align_spine=False),
    dict(type='RandomRot', theta=0.2),
    dict(type='GenSkeFeat', feats=[modality]),
    dict(type='UniformSample', clip_len=60),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize3D', align_spine=False),
    dict(type='GenSkeFeat', feats=[modality]),
    dict(type='UniformSample', clip_len=60, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize3D', align_spine=False),
    dict(type='GenSkeFeat', feats=[modality]),
    dict(type='UniformSample', clip_len=60, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    train=dict(
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='xset_train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='xset_val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='xset_val'))


# load_from='./work_dirs_sparse/stgcn/stgcn_pyskl_ntu120_xset_3dkp/j_full/latest.pth'