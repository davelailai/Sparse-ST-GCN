# Are Spatial-Temporal Graph Convolution Networks for Human Action Recognition Over-Parameterized?

We build this project based on the OpenSource Project [PYSKL](https://github.com/kennymckormick/pyskl).

This repo is the official implementation of [PoseConv3D](https://arxiv.org/abs/2104.13586) and [STGCN++](https://github.com/kennymckormick/pyskl/tree/main/configs/stgcn%2B%2B).

## install
 
1. download our code 
```shell
git clone https://github.com/davelailai/Sparse-ST-GCN.git
cd Sparse-ST-GCN
conda env create -f pyskl.yaml
conda activate pyskl
pip install -e .
```
2. Train the model with follows
```shell
bash tools/dist_train.sh configs/stgcn/stgcn_pyskl_ntu60_xview_3dkp/j.py
# For other models, replace the config with the corresponding models and replace the _base_ to _base_ = ['../STGCN_sparse.py']
```


