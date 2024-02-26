import sys
import numpy as np
import torch
from typing import Tuple, List, Union, Dict, Optional, Any, Callable, Sequence
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as nnf


get_trace = getattr(sys, 'gettrace', None)
DEBUG = get_trace is not None and get_trace() is not None
N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VN = Optional[ARRAY]
VNS = Optional[ARRAYS]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

V_Mesh = Tuple[ARRAY, ARRAY]
T_Mesh = Tuple[T, Optional[T]]
T_Mesh_T = Union[T_Mesh, T]
COLORS = Union[T, ARRAY, Tuple[int, int, int]]

D = torch.device


