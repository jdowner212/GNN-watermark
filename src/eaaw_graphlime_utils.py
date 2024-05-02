import copy
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import numpy as np
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.fft import ihfftn as inverse_fourier_transform


from torch_geometric.data import Data
from torch_geometric.nn import GATConv, MessagePassing
from torch_geometric.utils import to_networkx, k_hop_subgraph
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torchviz import make_dot
from torch_geometric.utils import k_hop_subgraph
import os

torch.manual_seed(2)


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 heads_1=8, heads_2=1, att_dropout=0, input_dropout=0):
        super(GAT, self).__init__()
        self.att_dropout = att_dropout
        self.input_dropout = input_dropout
        self.conv1 = GATConv(in_channels=input_dim,
                             out_channels=hidden_dim // heads_1,
                             heads=heads_1,
                             concat=True,
                             dropout=att_dropout)
        self.conv2 = GATConv(in_channels=hidden_dim,
                             out_channels=output_dim,
                             heads=heads_2,
                             concat=False,
                             dropout=att_dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

def prep_data():
    dataset = Planetoid('.', 'CORA', transform=NormalizeFeatures())
    data = dataset[0]
    train_mask = copy.deepcopy(data.train_mask)
    test_mask = copy.deepcopy(data.test_mask)
    data.train_mask = test_mask
    data.test_mask = train_mask
    return data

def accuracy(output, labels):
    _, pred = output.max(dim=1)
    correct = pred.eq(labels).double()
    correct = correct.sum()    
    return correct / len(labels)

def get_1_hop_edge_index(data, central_node, mask=None):
    edge_index_sub = torch.tensor([(n0,n1) for (n0,n1) in data.edge_index.T if n0==central_node or n1==central_node]).T
    if mask is not None:
        edge_index_sub =  torch.tensor([(n0,n1) for (n0,n1) in edge_index_sub.T if mask[n0].item()==True and mask[n1].item()==True]).T
    return edge_index_sub

def map_edges_to_new_ids(edge_index, node_id_mapping):
    mapped_edges = []
    for (n0, n1) in edge_index.T.tolist():
        mapped_edges.append([node_id_mapping[n0], node_id_mapping[n1]])
    return torch.tensor(mapped_edges).T


def get_masked_subgraph_nodes(data, central_node, hops=2, mask=None):
    seen_nodes = set()
    nodes_to_explore = set([central_node])
    subgraph_edge_list = []

    for hop in range(hops):
        nodes_to_explore_temp = set()  
        for this_central_node in nodes_to_explore:
            if this_central_node not in seen_nodes:
                seen_nodes.add(this_central_node)
                this_edge_index = get_1_hop_edge_index(data, this_central_node, mask=mask)
                for [n0, n1] in this_edge_index.T.tolist():
                    if [n0, n1] not in subgraph_edge_list and [n1, n0] not in subgraph_edge_list:
                        subgraph_edge_list.append([n0, n1])
                    if n0 != this_central_node and n0 not in seen_nodes:
                        nodes_to_explore_temp.add(n0)
                    if n1 != this_central_node and n1 not in seen_nodes:
                        nodes_to_explore_temp.add(n1)

        nodes_to_explore = nodes_to_explore.union(nodes_to_explore_temp)


    subgraph_edge_index = torch.tensor(subgraph_edge_list).T
    if hops==0:
        original_node_ids = torch.concat([torch.unique(subgraph_edge_index),torch.tensor([central_node])]).int()
    else:
        original_node_ids = torch.unique(subgraph_edge_index)
    original_node_ids, _ = torch.sort(original_node_ids)
    return original_node_ids



def aggregate_features(data, k, method='max', ignore_trailing_features = True):
    j = data.x.shape[1]
    segment_length = j // k
    new_features = torch.empty((data.x.shape[0], k))

    for j_ in range(k):
        start_idx = j_ * segment_length
        if ignore_trailing_features==True:
            end_idx = start_idx + segment_length # ignore any trailing features that don't fit in feature block
        else:
            end_idx = j if j_==k-1 else start_idx + segment_length # Ensure the last segment includes all remaining features
        
        # Aggregate the features (mean, sum, max, etc.)
        if method=='mean':
            new_features[:, j_] = data.x[:, start_idx:end_idx].mean(dim=1)
        elif method=='max':
            new_features[:, j_] = torch.max(data.x[:, start_idx:end_idx],dim=1)[0]

    data.x = new_features
    return data


def generate_subgraph(data, num_hops, node_index_to_watermark=None, d_reduce=None, show=True):
    train_indices = [i for i in range(len(data.train_mask.tolist())) if data.train_mask.tolist()[i]==True]

    G = to_networkx(data, to_undirected=True)
    degrees = dict(nx.degree(G))
    if node_index_to_watermark is None:
        train_degrees = {k:v for (k,v) in degrees.items() if k in train_indices}
        sorted_degrees = {k: v for k, v in sorted(train_degrees.items(), key=lambda item: item[1], reverse=True)}
        node_index_to_watermark = list(sorted_degrees.keys())[0]

    subgraph_node_idx = get_masked_subgraph_nodes(data, node_index_to_watermark, hops=num_hops, mask=data.train_mask)
    subgraph_node_idx, subgraph_edge_idx, inv, edge_mask = k_hop_subgraph(subgraph_node_idx, 0, edge_index=data.edge_index, num_nodes=data.num_nodes, relabel_nodes=True)
    data_sub = Data(x=data.x[subgraph_node_idx], edge_index=subgraph_edge_idx, y=data.y[subgraph_node_idx])
    if d_reduce is not None:
        data_sub = aggregate_features(copy.deepcopy(data_sub),d_reduce,method='max',ignore_trailing_features=False)

    if show==True:
        G_sub = to_networkx(data_sub, to_undirected=True)
        plt.figure(figsize=(5, 3))
        nx.draw_networkx(G_sub, with_labels=False,  node_color = 'blue', node_size=30)
        plt.title(f'{num_hops}-hop subgraph centered at node {node_index_to_watermark} (degree={degrees[node_index_to_watermark]})')
        plt.show()

    return data_sub, node_index_to_watermark, subgraph_node_idx



def generate_watermark(data, p_neg_ones=0.5, d_reduce=None):#, make_sparse=False):
    k = d_reduce if d_reduce is not None else data.num_node_features
    try:
        j = int(p_neg_ones*d_reduce)
    except:
        print('Note -- subgraph features currently not aggregated. Defaulting to original number of node features.')
        j = int(p_neg_ones*k)
    watermark = torch.ones(k)
    watermark_neg_1_indices = torch.randperm(k)[:j]
    watermark[watermark_neg_1_indices] = -1

#    if make_sparse==True:
    ''' only include values at indices corresponding to features present in data '''
    zero_features_mask = torch.sum(data.x,dim=0)==0
    watermark[zero_features_mask]=0

    return watermark

def __compute_kernel__(x, reduce):
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

def __compute_gram_matrix__(x):
    # Centers and normalizes
    G = x - torch.mean(x, axis=0, keepdims=True)
    G = G - torch.mean(G, axis=1, keepdims=True)
    G_norm = torch.norm(G, p='fro', dim=(0, 1), keepdim=True) + 1e-10
    G = G / G_norm
    return G


def densify(t, method='fourier'):
    if method=='fourier':
        t = torch.real(inverse_fourier_transform(t))
    elif method=='random_noise':
        t += (torch.rand_like(t.float())-1)*1e-5
    elif method=='subtract_small_constant':
        t -=1e-5
    elif method=='add_small_constant':
        t +=1e-5
    elif method=='exp':
        t = t.exp()
    return t


def train(data, watermark_loss_coefficient, epsilon, node_classifier_kwargs, subgraph_kwargs, densify_method=None):
    densify_features = True if densify_method is not None else False
    if densify_features==True:
        data.x = densify(data.x,method=densify_method)



    history = {'losses':[],'losses_primary':[],'losses_watermark':[],'betas':[],'beta_similarities':[],'train_accs':[],'val_accs':[],'L':[], 'K':[], 'K_bar':[], 'L_bar':[]}
    
    [d_reduce, num_hops, node_index_to_watermark] = list(subgraph_kwargs.values())

    node_classifier = GAT(input_dim=data.num_features, hidden_dim=16, output_dim=7)
    optimizer = optim.Adam(node_classifier.parameters(), lr=node_classifier_kwargs['lr'])
    node_classifier.train()

    data_sub, _, subgraph_node_idx = generate_subgraph(data, num_hops, node_index_to_watermark=node_index_to_watermark, d_reduce=d_reduce, show=False)
    watermark = generate_watermark(data_sub, p_neg_ones=0.5, d_reduce=d_reduce)#, make_sparse=use_sparse_watermark)

    for epoch in tqdm(range(node_classifier_kwargs['epochs'])):

        optimizer.zero_grad()

        log_logits = node_classifier(data.x, data.edge_index)
        loss_primary = F.nll_loss(log_logits[data.train_mask], data.y[data.train_mask])

        acc_trn = copy.deepcopy(accuracy(log_logits[data.train_mask].clone().detach(), data.y[data.train_mask].clone().detach()))
        acc_val = copy.deepcopy(accuracy(log_logits[data.val_mask].clone().detach(), data.y[data.val_mask].clone().detach()))


        # node_classifier.eval()
        #log_logits = node_classifier(data.x, data.edge_index)
        probas = log_logits.clone().exp()

        x_sub = data_sub.x # (n, d)
        y_sub = probas[subgraph_node_idx] # (n, classes)
        n, d = x_sub.shape


        K = __compute_kernel__(x_sub, reduce=False)  # (n, n, d)
        L = __compute_kernel__(y_sub, reduce=True)  # (n, n, 1)
        K_bar = __compute_gram_matrix__(K).reshape(n ** 2, d)
        L_bar = __compute_gram_matrix__(L).reshape(n ** 2,)

        KtK = torch.matmul(K_bar.T,K_bar)

        I = torch.eye(KtK.shape[0])
        lambda_ = 1e-1
        KtL = torch.matmul(K_bar.T,L_bar)
        beta =torch.matmul((KtK+lambda_*I).inverse(),KtL)       
        
        # ''' transform beta from sparse to dense if watermark is dense '''
        # if use_sparse_watermark==False:
        #     beta = sparse_to_dense_coeffs(beta, method=densify_beta_fn)
        
        beta = beta.reshape(-1)   
        # beta = torch.sign(beta)
        loss_watermark = watermark_loss_coefficient*torch.sum(torch.clamp(epsilon-beta*watermark,0))
        loss=loss_primary+loss_watermark
        loss.backward()


        history['losses_primary'].append(loss_primary.clone().detach())
        history['train_accs'].append(acc_trn.clone().detach())
        history['val_accs'].append(acc_val.clone().detach())
        history['betas'].append(beta.clone().detach())
        history['losses_watermark'].append(loss_watermark.clone().detach())
        history['beta_similarities'].append(torch.sum(beta*watermark).clone().detach())
        history['K_bar'].append(K_bar.clone().detach())
        history['L_bar'].append(torch.sum(L_bar).clone().detach())
        history['K'].append(K.clone().detach())
        history['L'].append(torch.sum(L).clone().detach())
        history['losses'].append(loss.clone().detach())


        optimizer.step()

        print('Epoch: {:3d}, loss_primary = {:.3f}, loss_watermark = {:.3f}, B*W = {:.5f}, train acc = {:.3f}, val acc = {:.3f}'.format(epoch, loss_primary, loss_watermark, torch.sum(beta*watermark), acc_trn, acc_val))
    
    return node_classifier, watermark, history, data_sub

