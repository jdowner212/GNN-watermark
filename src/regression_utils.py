
import numpy as np
from   tqdm.notebook import tqdm
import torch


from torch_geometric.data import Data  
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx, k_hop_subgraph, subgraph
from torch_geometric.transforms import BaseTransform, NormalizeFeatures, Compose

from config import *
from models import *
from general_utils import *


def compute_kernel(x, reduce):
    ''' Gaussian (RBF) Kernel '''
    assert x.ndim == 2, x.shape
    n, d = x.shape

    if x.shape[0]==0:
       print('Computing gram matrix for single-node subgraph -- betas will be zero!')

    ## Graph Fourier Transform will result in all zeros unless you normalize for numerical stability:
    x = (x-torch.mean(x))/torch.std(x)
    ## 

    dist = x.reshape(1, n, d) - x.reshape(n, 1, d)  # (n, n, d)
    dist = dist ** 2
    if reduce:
        dist = torch.sum(dist, dim=-1, keepdim=True)
    std = np.sqrt(d)            
    K = torch.exp(-dist / (2 * std ** 2 * 0.1 + 1e-10))
    del dist
    return K

def compute_gram_matrix(x):
    # Centers and normalizes
    if x.shape[0]==0:
        print('Computing gram matrix for single-node subgraph -- betas will be zero!')
    G = x - torch.mean(x, axis=0, keepdims=True)
    G = G - torch.mean(G, axis=1, keepdims=True)
    G_norm = torch.norm(G, p='fro', dim=(0, 1), keepdim=True) + 1e-10
    G = G / G_norm
    del G_norm
    return G


def solve_regression(x,y, lambda_=1e-1):
    n, d = x.shape

    K       = compute_kernel(x, reduce=False) #N,N,F
    K_bar   = compute_gram_matrix(K).reshape(n ** 2, d) #NxN, F
    L       = compute_kernel(y, reduce=True)
    L_bar   = compute_gram_matrix(L).reshape(n ** 2,)
    KtK     = torch.matmul(K_bar.T,K_bar)
    I       = torch.eye(KtK.shape[0])
    KtL     = torch.matmul(K_bar.T,L_bar)
    beta    = torch.matmul((KtK+lambda_*I).inverse(),KtL)      
    beta    = beta.reshape(-1)
    del K, K_bar, L, L_bar, KtK, I, KtL
    return beta

# def regress_on_subgraph(data, nodeIndices, probas, regression_kwargs):    x_this_sub = data.x[nodeIndices]
    # y_this_sub = probas[nodeIndices]
    # lambda_ = regression_kwargs['lambda']
    # beta_this_sub = solve_regression(x_this_sub, y_this_sub, lambda_).clone().detach()
    # return beta_this_sub

def regress_on_subgraph(x_this_sub, probas_this_sub, regression_kwargs):
    beta_this_sub = solve_regression(x_this_sub, probas_this_sub, regression_kwargs['lambda']).clone().detach()
    return beta_this_sub