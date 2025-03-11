_base_ = ['../GCGCN_component.py']
dataset_type = 'PoseDataset'
ann_file = 'data/nturgbd/ntu60_3danno.pkl'
train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample_order', clip_len=60),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label','total_frames','clip_len','frame_inds'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint','total_frames','clip_len','frame_inds'])
]
val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample_order', clip_len=60, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label','total_frames','clip_len','frame_inds'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint','total_frames','clip_len','frame_inds'])
]
test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample_order', clip_len=60, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label','total_frames','clip_len','frame_inds'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint','total_frames','clip_len','frame_inds'])
]
data = dict(
    train=dict(
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='xsub_train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='xsub_val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='xsub_val'))

work_dir = './work_dirs/gcgcn/gcgcn_pyskl_ntu60_xsub_3dkp/j_comp_up'
