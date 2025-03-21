# Copyright (c) OpenMMLab. All rights reserved.
import imp
from .recognizer2d import Recognizer2D
from .recognizer3d import Recognizer3D
from .recognizergcn import RecognizerGCN
from .recognizergcnR import RecognizerGCNR
from .recognizergcnPre import RecognizerGCNPre
from .RecongnizerGCNcau import RecognizerGCNCau
from .Recognizergcn_gc import RecognizerGCN_GC
from .recognizergcn_gt import RecognizerGCN_GT
from .RecongnizerGCN_sparse import RecognizerGCN_sparse
from .RecongnizerGCN_Assemble import RecognizerGCN_assemble
from .RecognizerGCN_sMoE import RecognizerGCN_sMoEassemble
from .recognizergcn_QL import RecognizerGCN_QL

__all__ = ['Recognizer2D', 'Recognizer3D', 'RecognizerGCN','RecognizerGCNR','RecognizerGCNPre','RecognizerGCNCau','RecognizerGCN_GC','RecognizerGCN_GT','RecognizerGCN_sparse',
           'RecognizerGCN_assemble','RecognizerGCN_sMoEassemble','RecognizerGCN_QL']
