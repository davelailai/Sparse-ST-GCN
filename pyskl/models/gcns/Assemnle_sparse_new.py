import torch
import torch.nn as nn

from pyskl.models.gcns.utils import gcn_sparse, tcn_sparse
from torch.distributions.normal import Normal

from ...utils import Graph
from ..builder import BACKBONES
from .utils import MSTCN, mstcn_sparse, unit_ctrgcn, unit_tcn_sparse,unit_tcn, unit_ctrhgcn, unit_ctrgcn_sparse,get_sparsity
from .ctrgcn_sparse import CTRGCNBlock,CTRGCN_sparse
from .aagcn_sparse import AAGCNBlock,AAGCN_sparse
from .stgcn_sparse import STGCNBlock,STGCN_sparse
from .dggcn_sparse import DGBlock,DGSTGCN_sparse
from .ctrgcn import CTRGCNBlock,CTRGCN
from .dgstgcn import DGBlock,DGSTGCN
from .stgcn import STGCN
from .aagcn import AAGCN

# class JointToBone:

#     def __init__(self, dataset='nturgb+d', target='keypoint'):
#         self.dataset = dataset
#         self.target = target
#         if self.dataset not in ['nturgb+d', 'openpose', 'coco']:
#             raise ValueError(
#                 f'The dataset type {self.dataset} is not supported')
#         if self.dataset == 'nturgb+d':
#             self.pairs = [(0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (8, 20), (9, 8),
#                           (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
#                           (19, 18), (21, 22), (20, 20), (22, 7), (23, 24), (24, 11)]
#         elif self.dataset == 'openpose':
#             self.pairs = ((0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
#                           (11, 5), (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15))
#         elif self.dataset == 'coco':
#             self.pairs = ((0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 0), (6, 0), (7, 5), (8, 6), (9, 7), (10, 8),
#                           (11, 0), (12, 0), (13, 11), (14, 12), (15, 13), (16, 14))

#     def __call__(self, results):

#         keypoint = results
#         N, M, T, V, C = results.shape
#         bone=torch.zeros_like(keypoint)
#         # bone = n.zeros((M, T, V, C), dtype=np.float32)

#         assert C in [2, 3]
#         for v1, v2 in self.pairs:
#             bone[..., v1, :] = keypoint[..., v1, :] - keypoint[..., v2, :]
#             if C == 3 and self.dataset in ['openpose', 'coco']:
#                 score = (keypoint[..., v1, 2] + keypoint[..., v2, 2]) / 2
#                 bone[..., v1, 2] = score

#         results[self.target] = bone
#         return results
    
# class ToMotion:

#     def __init__(self, dataset='nturgb+d', source='keypoint', target='motion'):
#         self.dataset = dataset
#         self.source = source
#         self.target = target

#     def __call__(self, results):
#         data = results[self.source]
#         M, T, V, C = data.shape
#         motion=torch.zeros_like(data)
#         motion = np.zeros_like(data)

#         assert C in [2, 3]
#         motion[:, :T - 1] = np.diff(data, axis=1)
#         if C == 3 and self.dataset in ['openpose', 'coco']:
#             score = (data[:, :T - 1, :, 2] + data[:, 1:, :, 2]) / 2
#             motion[:, :T - 1, :, 2] = score

#         results[self.target] = motion

#         return results


@BACKBONES.register_module()
class Assemble_sparse_New(nn.Module):
    def __init__(self,
                 graph_cfg,
                 model_list,
                 sparse_ratio,
                 in_channels=3,
                 base_channels=64,
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 pretrained=None,
                 sparse_decay=False,
                #  linear_sparsity=0,
                 warm_up=0, 
                 num_person=2,
                 **kwargs):
        super(Assemble_sparse_New, self).__init__()

        # self.graph = Graph(**graph_cfg)
        # A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.num_experts=len(model_list)
        self.num_person = num_person
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.sparse_decay = sparse_decay
        # self.linear_sparsity = linear_sparsity
        self.warm_up = warm_up
        self.model_list = model_list
        self.sparse_ratio = sparse_ratio

        models=[]
        for  i,(model_unit, sparse_unit) in enumerate(zip(model_list, sparse_ratio)): 
            assert model_unit in ['ST-GCN', 'AA-GCN', 'CTR-GCN', 'DG-GCN']
            if model_unit =='ST-GCN':
                ST_kwargs = kwargs['ST_kwargs']
                models.append(STGCN_sparse(graph_cfg,in_channels=in_channels,linear_sparsity=sparse_unit,warm_up=warm_up,
                                           tcn_sparse_ratio=sparse_unit,gcn_sparse_ratio=sparse_unit,**ST_kwargs))
            if model_unit =='AA-GCN':
                AA_kwargs = kwargs['AA_kwargs']
                models.append(AAGCN_sparse(graph_cfg,in_channels=in_channels,linear_sparsity=sparse_unit,warm_up=warm_up,
                                           tcn_sparse_ratio=sparse_unit,gcn_sparse_ratio=sparse_unit,**AA_kwargs))
            if model_unit =='CTR-GCN':
                CTR_kwargs = kwargs['CTR_kwargs']
                models.append(CTRGCN_sparse(graph_cfg,in_channels=in_channels,linear_sparsity=sparse_unit,warm_up=warm_up,
                                            tcn_sparse_ratio=sparse_unit,gcn_sparse_ratio=sparse_unit,**CTR_kwargs))
            if model_unit =='DG-GCN':
                DG_kwargs = kwargs['DG_kwargs']
                models.append(DGSTGCN_sparse(graph_cfg,in_channels=in_channels,linear_sparsity=sparse_unit,warm_up=warm_up,
                                             tcn_sparse_ratio=sparse_unit,gcn_sparse_ratio=sparse_unit,**DG_kwargs))

        self.experts = nn.ModuleList(models)

    def init_weights(self):
        for module in self.experts:
            module.init_weights()


    def forward(self, x, current_epoch, max_epoch):
        N, M, T, V, C = x.size()

        x_assemble = []
        for i in range(len(self.model_list)):
            x_assemble.append(self.experts[i](x, current_epoch, max_epoch))
 

        x_assemble = torch.stack(x_assemble)
        # x_assemble = x_assemble.reshape((x_assemble.shape[0],N, M) + x_assemble.shape[2:])
        
        return x_assemble
    def GCN_feature(self, x):
        pool = nn.AdaptiveAvgPool2d(1)
        N, M, C, T, V = x.shape
        x = x.reshape(N * M, C, T, V)

        x = pool(x)
        # x = pool(x)
        x = x.reshape(N, M, C)
        x = x.mean(dim=1)
        # x = x.reshape(N*M, C)
        # x = x.mean(dim=1)
        return x
    
    def get_threshold(self, model, sparsity,linear_sparsity):
        local=[]
        for name, p in model.named_parameters():
            if hasattr(p, 'is_score') and p.is_score and p.sparsity==linear_sparsity:
                local.append(p.detach().flatten())
        local=torch.cat(local)
        threshold= self.percentile(local,sparsity*100)
        return threshold
    
    def percentile(self, t, q):
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        return t.view(-1).kthvalue(k).values.item()
    
    def get_mask(self, model, current_epoch, max_epoch,linear_sparsity):
        # if current_epoch < self.warm_up:
        #     sparsity = 0
        # else:
        #     if self.sparse_decay:
        #         if current_epoch<(max_epoch/2.0):
        #             sparsity=get_sparsity(linear_sparsity,current_epoch,0,max_epoch/2)
        #         else:
        #             sparsity=linear_sparsity
        #     else:
        #         sparsity=linear_sparsity
        sparsity=linear_sparsity
        threshold = self.get_threshold(model,sparsity,linear_sparsity)
        mask=[]
        weight = []
        for name, p in model.named_parameters():
            if hasattr(p, 'is_score') and p.is_score and p.sparsity==linear_sparsity:
                mask.append(p.detach().flatten())
            if hasattr(p, 'is_mask') and p.is_mask:
                weight.append(p.detach().flatten())
        mask = torch.cat(mask)
        weight = torch.cat(weight)
        mask =mask<=threshold
        weight = weight*mask

        return weight

    def regularize(self, lam, penalty, current_epoch, max_epoch):
        '''
        Calculate regularization term for first layer weight matrix.

        Args:
        network: MLP network.
        penalty: one of GL (group lasso), GSGL (group sparse group lasso),
            H (hierarchical).
        '''
        W =[]
        for j in range(len(self.model_list)):
            for i in range(self.num_stages):
                try:
                    W.append(self.get_mask(self.experts[j].gcn[i], current_epoch, max_epoch,self.sparse_ratio[j]))
                except:
                    W.append(self.get_mask(self.experts[j].net[i], current_epoch, max_epoch,self.sparse_ratio[j]))   
                else:
                    W.append(self.get_mask(self.experts[j].gcn[i], current_epoch, max_epoch,self.sparse_ratio[j]))
                   
                # W
        # W = gc
        # W = network.layers[0].weight
        # W = torch.stack(W)
        # W = torch.cat(W)
        # b,hidden, p, p1, lag = weight.shape
        panelty = []
        if penalty == 'GL':
            W = torch.cat(W)
            return lam * torch.sum(torch.norm(W, dim=0))
        elif penalty == 'GSGL':
            panelty = []
            for weight_layer in W:
                index = torch.sum(torch.norm(weight_layer, dim=0))
                panelty.append(index)
            panelty = torch.stack(panelty)
            return lam*torch.sum(panelty)
            # return panelty
            # return lam * (torch.sum(torch.norm(W, dim=(1, -1)))
            #             + torch.sum(torch.norm(W, dim=1)))
        elif penalty == 'H':
            # Lowest indices along third axis touch most lagged values.
            # return lam * sum([torch.sum(torch.norm(W[:, :, :(i+1)], dim=(0, 1)))
            #                 for i in range(lag)])
            pass
        else:
            raise ValueError('unsupported penalty: %s' % penalty)