import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from   IPython.display import HTML
import numpy as np
import networkx as nx
import numpy as np
from tqdm.notebook import tqdm
 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GATConv, MessagePassing
from torch_geometric.utils import to_networkx, k_hop_subgraph
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torchviz import make_dot
from torch_geometric.utils import k_hop_subgraph
import os

class HSICLasso(nn.Module):
    def __init__(self, sigma_x, sigma_y, rho, feature_dim):
        super(HSICLasso, self).__init__()
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.rho = rho
        self.beta = nn.Parameter(torch.randn(feature_dim, requires_grad=True))

    def forward(self, x, y):
        n = x.shape[0]

        # Compute the RBF kernels for X
        x = x.view(n, 1, -1) - x.view(1, n, -1)
        Kx = torch.exp(-torch.sum(x**2, dim=2) / (2 * self.sigma_x**2))

        # Compute the RBF kernel for Y
        y = y.view(n, 1, -1) - y.view(1, n, -1)
        Ky = torch.exp(-torch.sum(y**2, dim=2) / (2 * self.sigma_y**2))

        # Centering the kernels
        H = torch.eye(n) - torch.ones((n, n)) / n
        Kx_centered = H @ Kx @ H
        Ky_centered = H @ Ky @ H

        # Calculate HSIC Lasso objective
        HSIC_value = torch.trace(Kx_centered @ Ky_centered)
        loss = -HSIC_value + self.rho * torch.sum(torch.abs(self.beta))

        return loss

class LinearRegression(nn.Module):
    def __init__(self, n_features, n_classes, rho):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(n_features, n_classes, bias=False)
        # nn.init.constant_(self.linear.weight, 0.0)
        self.rho = rho

    def forward(self, x):
        return self.linear(x)

    def get_lasso_loss(self, predictions, targets):
        mse_loss = nn.MSELoss()(predictions, targets.reshape(targets.shape[0],1))
        l1_loss = self.rho * torch.sum(torch.abs(self.linear.weight))
        return mse_loss + l1_loss
    
class LassoRegression(nn.Module):
    def __init__(self, n_features, rho):
        super(LassoRegression, self).__init__()
        self.linear = nn.Linear(n_features, 1, bias=False)
        # nn.init.constant_(self.linear.weight, 0.0)
        self.rho = rho

    def forward(self, x):
        return self.linear(x)

    def get_lasso_loss(self, predictions, targets):
        mse_loss = nn.MSELoss()(predictions, targets.reshape(targets.shape[0],1))
        l1_loss = self.rho * torch.sum(torch.abs(self.linear.weight))
        return mse_loss + l1_loss

class LassoRegression_scratch(nn.Module):
    def __init__(self, rho, n_features, initial_beta):
        super(LassoRegression_scratch, self).__init__()
        self.beta = nn.Parameter(initial_beta)
        self.rho = rho

    def forward(self, x):
        return torch.mm(x, self.beta)
    
    def get_lasso_loss(self, predictions, targets):
        mse_loss = nn.MSELoss()(predictions, targets.reshape(targets.shape[0], 1))
        # l1_loss = self.rho * torch.sum(torch.abs(self.beta))
        return mse_loss#, l1_loss
    

class GraphLIME:
    
    def __init__(self, model, hop=2, 
                 explainer_kwargs={},
                 cached=True, 
                 train=False):
        self.hop = hop
        self.rho = explainer_kwargs['rho']
        self.lr = explainer_kwargs['lr']
        self.epochs = explainer_kwargs['epochs']
        self.model = model
        self.cached = cached
        self.cached_result = None

        if train==False:
            self.model.eval()

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __init_predict__(self, x, edge_index, train=False):
        if self.cached and self.cached_result is not None:
            if x.size(0) != self.cached_result.size(0):
                raise RuntimeError(
                    'Cached {} number of nodes, but found {}.'.format(
                        x.size(0), self.cached_result.size(0)))

        if not self.cached or self.cached_result is None: # this is the case I observe during training
            if train==False:
                with torch.no_grad():
                    log_logits = self.model(x=x, edge_index=edge_index)
                    probas = log_logits.exp()
            else:
                log_logits = self.model(x=x, edge_index=edge_index)
                probas = log_logits.exp()
            self.cached_result = probas
        return self.cached_result

    
    def __compute_kernel__(self, x, reduce):
        # Gaussian (RBF) Kernel
        assert x.ndim == 2, x.shape
        n, d = x.shape
        dist = x.reshape(1, n, d) - x.reshape(n, 1, d)  # (n, n, d)
        dist = dist ** 2
        if reduce:
            dist = torch.sum(dist, dim=-1, keepdim=True)
        std = np.sqrt(d)            
        K = torch.exp(-dist / (2 * std ** 2 * 0.1 + 1e-10))
        return K
    
    def __compute_gram_matrix__(self, x):
        # Centers and normalizes
        G = x - torch.mean(x, axis=0, keepdims=True)
        G = G - torch.mean(G, axis=1, keepdims=True)
        G_norm = torch.norm(G, p='fro', dim=(0, 1), keepdim=True) + 1e-10
        G = G / G_norm
        return G


    def explain_node_pytorch(self, data, node_idx, train=False, d_reduce=None, num_hops=2,**kwargs):
#        probas = self.__init_predict__(data.x, data.edge_index, train=train, **kwargs)
        if train==False:
            with torch.no_grad():
                log_logits = self.model(x=data.x, edge_index=data.edge_index)
                probas = log_logits.exp()
        else:
            log_logits = self.model(x=data.x, edge_index=data.edge_index)
            probas = log_logits.exp()
        self.cached_result = probas
        
        data_sub, _, subgraph_node_idx = generate_subgraph(data, num_hops, node_index_to_watermark=node_idx, d_reduce=d_reduce, show=False)

        x_sub = data_sub.x # (n, d)
        y_sub = probas[subgraph_node_idx] # (n, classes)
        n, d = x_sub.shape

        K = self.__compute_kernel__(x_sub, reduce=False)  # (n, n, d)
        L = self.__compute_kernel__(y_sub, reduce=True)  # (n, n, 1)
        K_bar = self.__compute_gram_matrix__(K).reshape(n ** 2, d)
        L_bar = self.__compute_gram_matrix__(L).reshape(n ** 2,)


        KtK = torch.matmul(K_bar.T,K_bar)
        I = torch.eye(KtK.shape[0])
        l = 1e-6
        KtL = torch.matmul(K_bar.T,L_bar)
        beta =torch.matmul((KtK+l*I).inverse(),KtL)       
        return beta


def soft_sign(t,k=1.6):
    min_abs = torch.min(torch.abs(t))
    factor = min(1/min_abs,1e5)
    zero_to_one_range = (t-torch.min(t))/(torch.max(t)-torch.min(t))
    neg_one_to_one_range = 2*torch.tanh(k*zero_to_one_range)-1
    neg_one_to_one_range = torch.clamp(factor*neg_one_to_one_range,min=-1,max=1)
    return neg_one_to_one_range

def soft_sign_alt(t):
    zero_to_one_range = (t-torch.min(t))/(torch.max(t)-torch.min(t))
    neg_one_to_one_range = 2*zero_to_one_range-1
    return neg_one_to_one_range

    
def compute_normalized_centered_gram_matrix(x, reduce):
    def compute_kernel(x, reduce):
        assert x.ndim == 2, x.shape
        n, d = x.shape
        dist = x.reshape(1, n, d) - x.reshape(n, 1, d)
        dist = dist ** 2
        if reduce:
            dist = torch.sum(dist, dim=-1, keepdim=True)
        std = np.sqrt(d)        
        K = torch.exp(-dist / (2 * std ** 2 * 0.1 + 1e-10))
        return K
    K = compute_kernel(x, reduce)
    G = K - torch.mean(K, axis=0, keepdims=True)
    G = G - torch.mean(G, axis=1, keepdims=True)
    G_norm = torch.norm(G, p='fro', dim=(0, 1), keepdim=True) + 1e-10
    G = G / G_norm
    return G