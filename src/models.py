from   config import *
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, GCNConv, GraphConv, SAGEConv
from torch_geometric.nn.conv import SGConv


class GAT(nn.Module):
    def __init__(self, inDim, hDim, outDim,
                 heads_1=8, heads_2=1, attDrop=0, inDrop=0, nLayers=2, skip_connections=False, activation_fn=F.elu):
        super(GAT, self).__init__()
        self.attDrop = attDrop
        self.inDrop = inDrop
        self.nLayers=nLayers
        self.skip_connections=skip_connections
        self.activation_fn = activation_fn
        self.convs = nn.ModuleList()

        # First layer
        conv1 = GATConv(in_channels=inDim,
                        out_channels=hDim // heads_1,
                        heads=heads_1,
                        concat=True,
                        dropout=attDrop)
        self.convs.append(conv1)

        # Intermediate layers
        for l in range(nLayers - 2):
            self.convs.append(GATConv(in_channels=hDim,
                                      out_channels=hDim // heads_2,
                                      heads=heads_2,
                                      concat=True,
                                      dropout=attDrop))

        # Final layer
        self.convs.append(GATConv(in_channels=hDim,
                                  out_channels=outDim,
                                  heads=heads_2,
                                  concat=False,
                                  dropout=attDrop))

    def forward(self, x, edge_index):
        intermediate_outputs = []
        for l in range(self.nLayers):
            x = self.convs[l](x,edge_index)
            x = self.activation_fn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            intermediate_outputs.append(x)
        if self.skip_connections == True:
            x = torch.cat(intermediate_outputs, dim=-1)
            x = self.lin(x)
        return F.log_softmax(x, dim=1)




class GCN(torch.nn.Module):
    def __init__(self, inDim, hDim, outDim, nLayers=2, dropout=0,skip_connections=False, activation_fn=F.relu):

        super(GCN, self).__init__()
        self.nLayers=nLayers
        self.dropout = dropout
        self.skip_connections=skip_connections
        self.activation_fn=activation_fn
        self.convs = nn.ModuleList()

        # First layer
        conv1 = GCNConv(in_channels=inDim,
                        out_channels=hDim)
        self.convs.append(conv1)

        # Intermediate layers
        for l in range(nLayers - 2):
            self.convs.append(GCNConv(in_channels=hDim,
                                      out_channels=hDim))

        # Final layer
        self.convs.append(GCNConv(in_channels=hDim,
                                  out_channels=outDim))
        
    def forward(self, x, edge_index):
        intermediate_outputs = []
        for l in range(self.nLayers):
            x = self.convs[l](x,edge_index)
            x = self.activation_fn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            intermediate_outputs.append(x)
        if self.skip_connections == True:
            x = torch.cat(intermediate_outputs, dim=-1)
            x = self.lin(x)
        return F.log_softmax(x, dim=1)
    


class GCN_special(torch.nn.Module):
    def __init__(self, inDim, hDim, hDim_subgraphs, outDim, nLayers=2, nLayers_subgraphs=3, dropout=0, dropout_subgraphs=0, skip_connections=False, activation_fn=F.relu):

        super(GCN_special, self).__init__()
        self.nLayers=nLayers
        self.dropout = dropout
        self.skip_connections=skip_connections
        self.activation_fn=activation_fn
        
        self.convs = nn.ModuleList()
        conv1 = GCNConv(in_channels=inDim,out_channels=hDim)
        self.convs.append(conv1)
        for l in range(nLayers - 2):
            self.convs.append(GCNConv(in_channels=hDim,out_channels=hDim))
        self.convs.append(GCNConv(in_channels=hDim,out_channels=outDim))

        self.convs_subgraphs = nn.ModuleList()
        conv1 = GCNConv(in_channels=inDim,out_channels=hDim_subgraphs)
        self.conv_subgraphs.append(conv1)
        for l in range(nLayers_subgraphs- 2):
            self.convs_subgraphs.append(GCNConv(in_channels=hDim_subgraphs,out_channels=hDim_subgraphs))
        self.convs_subgraphs.append(GCNConv(in_channels=hDim_subgraphs,out_channels=outDim))
        
    def forward(self, x, edge_index, subgraphs=False):
        nLayers = self.nLayers if subgraphs==False else self.nLayers_subgraphs
        dropout = self.dropout if subgraphs == False else self.dropout_subgraphs
        convs = self.convs if subgraphs==False else self.convs_subgraphs
        intermediate_outputs = []
        for l in range(nLayers):
            x = convs[l](x,edge_index)
            x = self.activation_fn(x)
            x = F.dropout(x, p=dropout, training=self.training)
            intermediate_outputs.append(x)
        if self.skip_connections == True:
            x = torch.cat(intermediate_outputs, dim=-1)
            x = self.lin(x)
        return F.log_softmax(x, dim=1)

class GraphConv_(torch.nn.Module):
    def __init__(self, inDim, hDim, outDim, nLayers=2, dropout=0, skip_connections=False, activation_fn=F.relu):

        super(GraphConv_, self).__init__()
        self.nLayers=nLayers
        self.dropout = dropout
        self.skip_connections = skip_connections
        self.activation_fn=activation_fn
        self.convs = nn.ModuleList()

        # First conv layer
        self.convs.append(GraphConv(in_channels=inDim, out_channels=hDim))
        # Intermediate conv layer
        for l in range(nLayers - 2):
            self.convs.append(GraphConv(in_channels=hDim,out_channels=hDim))
        # Final conv layer
        self.convs.append(GraphConv(in_channels=hDim, out_channels=outDim))
        
        self.lin = torch.nn.Linear(self.nLayers * hDim, outDim)
        
    def forward(self, x, edge_index):
        intermediate_outputs = []
        for l in range(self.nLayers):
            x = self.convs[l](x,edge_index)
            x = self.activation_fn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            intermediate_outputs.append(x)
        if self.skip_connections == True:
            x = torch.cat(intermediate_outputs, dim=-1)
            x = self.lin(x)
        return F.log_softmax(x, dim=1)

    


class SAGE_special(torch.nn.Module):
    def __init__(self, inDim, hDim, hDim_subgraphs, outDim, nLayers=2, nLayers_subgraphs=3, dropout=0, dropout_subgraphs=0, skip_connections=False, activation_fn=F.elu, conv_fn=SAGEConv):
        super(SAGE_special, self).__init__()
        self.nLayers = nLayers
        self.dropout = dropout
        self.dropout_subgraphs = dropout_subgraphs
        self.skip_connections = skip_connections
        self.activation_fn = activation_fn
        
        self.convs = nn.ModuleList()
        self.convs.append(conv_fn(in_channels=inDim, out_channels=hDim))
        for l in range(nLayers - 2):
            self.convs.append(conv_fn(in_channels=hDim, out_channels=hDim))
        self.convs.append(conv_fn(in_channels=hDim, out_channels=outDim))

        self.subgraph_convs = nn.ModuleList()
        self.subgraph_convs.append(conv_fn(in_channels=inDim, out_channels=hDim_subgraphs))
        for l in range(nLayers_subgraphs - 2):
            self.subgraph_convs.append(conv_fn(in_channels=hDim_subgraphs, out_channels=hDim_subgraphs))
        self.subgraph_convs.append(conv_fn(in_channels=hDim_subgraphs, out_channels=outDim))
        if skip_connections:
            self.lin = nn.Linear(hDim * nLayers, outDim)
    
    def forward(self, x, edge_index, subgraphs=False):
        convs = self.subgraph_convs if subgraphs else self.convs
        dropout_rate = self.dropout_subgraphs if subgraphs else self.dropout  # Different dropout rates
        intermediate_outputs = []
        for l in range(self.nLayers):
            x = convs[l](x, edge_index)
            x = self.activation_fn(x)
            x = F.dropout(x, p=dropout_rate, training=self.training)
            intermediate_outputs.append(x)
        if self.skip_connections:
            x = torch.cat(intermediate_outputs, dim=-1)
            x = self.lin(x)
        return F.log_softmax(x, dim=1)



class SAGE(torch.nn.Module):
    def __init__(self, inDim, hDim, outDim, nLayers=2, dropout=0, skip_connections=False, activation_fn=F.elu, conv_fn=SAGEConv):

        super(SAGE, self).__init__()
        self.nLayers=nLayers
        self.dropout = dropout
        self.skip_connections=skip_connections
        self.activation_fn=activation_fn
        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(in_channels=inDim, out_channels=hDim))
        # Intermediate layers
        for l in range(nLayers - 2):
            self.convs.append(SAGEConv(in_channels=hDim, out_channels=hDim))
        # Final layer
        self.convs.append(SAGEConv(in_channels=hDim, out_channels=outDim))
        
    def forward(self, x, edge_index):
        intermediate_outputs = []
        for l in range(self.nLayers):
            x = self.convs[l](x,edge_index)
            x = self.activation_fn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            intermediate_outputs.append(x)
        if self.skip_connections == True:
            x = torch.cat(intermediate_outputs, dim=-1)
            x = self.lin(x)
        return F.log_softmax(x, dim=1)



class Net(torch.nn.Module):
    def __init__(self, **model_kwargs):
        super(Net, self).__init__()
        inDim = model_kwargs['inDim'] 
        hDim = model_kwargs['hDim']
        outDim = model_kwargs['outDim']
        self.dropout = model_kwargs['dropout']
        self.activation_fn = {'elu': F.elu, 'relu': F.relu}[model_kwargs['activation']]
        self.nLayers = model_kwargs['nLayers'] 


        conv_fn = {'GAT': GATConv, 'GCN': GCNConv, 'GraphConv': GraphConv, 'SAGE': SAGEConv, 'SGC': SGConv}[model_kwargs['arch']]
        self.convs = nn.ModuleList()
        if model_kwargs['arch']=='GAT':
            heads_1 = model_kwargs['heads_1']
            heads_2 = model_kwargs['heads_2']
            self.convs.append(conv_fn(in_channels=inDim, out_channels=hDim // heads_1, heads=heads_1, concat=True, dropout=self.dropout)) # First conv layer
            for l in range(self.nLayers - 2): # Intermediate conv layers
                self.convs.append(conv_fn(in_channels=hDim, out_channels=hDim // heads_2, heads=heads_2, concat=True, dropout=self.dropout))
            self.convs.append(conv_fn(in_channels=hDim, out_channels=outDim, heads=heads_2, concat=False, dropout=self.dropout)) # Final conv layer
            
        elif model_kwargs['arch']!='GAT':
            self.convs.append(conv_fn(in_channels=inDim, out_channels=hDim)) # First conv layer
            for l in range(self.nLayers - 2): # Intermediate conv layers
                self.convs.append(conv_fn(in_channels=hDim, out_channels=hDim))
            self.convs.append(conv_fn(in_channels=hDim, out_channels=outDim)) # Final conv layer
        
        self.skip_connections = model_kwargs['skip_connections']
        if self.skip_connections:
            self.lin = torch.nn.Linear(((self.nLayers-1)*hDim)+outDim, outDim)


        self.feature_weights = torch.zeros(inDim)  # To track feature weights

        
    def forward(self, x, edge_index, dropout=0):
        intermediate_outputs = []
        for l in range(self.nLayers):
            x = self.convs[l](x,edge_index)
            x = self.activation_fn(x)
            x = F.dropout(x, p=dropout, training=self.training)
            intermediate_outputs.append(x)
        if self.skip_connections == True:
            x = torch.cat(intermediate_outputs, dim=-1)
            x = self.lin(x)
        return F.log_softmax(x, dim=1)

    def update_feature_weights(self, x):
        grad = x.grad.abs().mean(dim=0)
        self.feature_weights += grad



''' consider this later for Flickr if still struggling '''
# import argparse
# import os.path as osp

# import torch_sparse
# import torch.nn.functional as F

# from torch_geometric.datasets import Flickr
# from torch_geometric.loader import GraphSAINTRandomWalkSampler
# from torch_geometric.nn import GraphConv
# from torch_geometric.typing import WITH_TORCH_SPARSE
# from torch_geometric.utils import degree

# if not WITH_TORCH_SPARSE:
#     quit("This example requires 'torch-sparse'")

# # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
# # dataset = Flickr(path)
# data = dataset[0]
# row, col = data.edge_index
# data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.

# # parser = argparse.ArgumentParser()
# # parser.add_argument('--use_normalization', action='store_true')
# # args = parser.parse_args()

# loader = GraphSAINTRandomWalkSampler(data, batch_size=6000, walk_length=2,
#                                      num_steps=5, sample_coverage=100,
#                                      save_dir=dataset.processed_dir,
#                                      num_workers=4)


# class Net(torch.nn.Module):
#     def __init__(self, hidden_channels):
#         super().__init__()
#         in_channels = dataset.num_node_features
#         out_channels = dataset.num_classes
#         self.conv1 = GraphConv(in_channels, hidden_channels)
#         self.conv2 = GraphConv(hidden_channels, hidden_channels)
#         self.conv3 = GraphConv(hidden_channels, hidden_channels)
#         self.lin = torch.nn.Linear(3 * hidden_channels, out_channels)

#     def set_aggr(self, aggr):
#         self.conv1.aggr = aggr
#         self.conv2.aggr = aggr
#         self.conv3.aggr = aggr

#     def forward(self, x0, edge_index, edge_weight=None):
#         x1 = F.relu(self.conv1(x0, edge_index, edge_weight))
#         x1 = F.dropout(x1, p=0.2, training=self.training)
#         x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
#         x2 = F.dropout(x2, p=0.2, training=self.training)
#         x3 = F.relu(self.conv3(x2, edge_index, edge_weight))
#         x3 = F.dropout(x3, p=0.2, training=self.training)
#         x = torch.cat([x1, x2, x3], dim=-1)
#         x = self.lin(x)
#         return x.log_softmax(dim=-1)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net(hidden_channels=256).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# use_normalization = True
# def train():
#     model.train()
#     model.set_aggr('add' if use_normalization else 'mean')

#     total_loss = total_examples = 0
#     for data in loader:
#         data = data.to(device)
#         optimizer.zero_grad()

#         if use_normalization:
#             edge_weight = data.edge_norm * data.edge_weight
#             out = model(data.x, data.edge_index, edge_weight)
#             loss = F.nll_loss(out, data.y, reduction='none')
#             loss = (loss * data.node_norm)[data.train_mask].sum()
#         else:
#             out = model(data.x, data.edge_index)
#             loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * data.num_nodes
#         total_examples += data.num_nodes
#     return total_loss / total_examples


# @torch.no_grad()
# def test():
#     model.eval()
#     model.set_aggr('mean')

#     out = model(data.x.to(device), data.edge_index.to(device))
#     pred = out.argmax(dim=-1)
#     correct = pred.eq(data.y.to(device))

#     accs = []
#     for _, mask in data('train_mask', 'val_mask', 'test_mask'):
#         accs.append(correct[mask].sum().item() / mask.sum().item())
#     return accs


# for epoch in range(1, 51):
#     loss = train()
#     accs = test()
#     print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {accs[0]:.4f}, '
#           f'Val: {accs[1]:.4f}, Test: {accs[2]:.4f}')


import torch



import torch
from torch.optim import Optimizer
import torch.nn.functional as F


import torch
from torch.optim import Optimizer

class SAM(Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho value: {rho}"
        defaults = dict(rho=rho, **kwargs)
        params = list(params)  # Convert the generator to a list

        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(params, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

        # Set rho for each parameter group
        for group in self.param_groups:
            group.setdefault('rho', rho)
        
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)

            for p in group['params']:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.sub_(2 * p.grad * group['rho'])  # go back to the original point "w - e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    def step(self, closure=None):
        assert closure is not None, "SAM requires closure, but it was not provided"

        closure = torch.enable_grad()(closure)
        closure()
        self.first_step(zero_grad=True)
        closure()
        self.second_step()
        self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm