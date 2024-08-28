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


from config import *
from general_utils import *
from models import *
from subgraph_utils import *
from transform_functions import *
from watermark_utils import *

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


    def explain_node_pytorch(self, data, node_idx, train=False, d_reduce=None, num_hops=2, seed=0, **kwargs):
#        probas = self.__init_predict__(data.x, data.edge_index, train=train, **kwargs)
        if train==False:
            with torch.no_grad():
                log_logits = self.model(x=data.x, edge_index=data.edge_index)
                probas = log_logits.exp()
        else:
            log_logits = self.model(x=data.x, edge_index=data.edge_index)
            probas = log_logits.exp()
        self.cached_result = probas
        
        data_sub, _, subgraph_node_idx = generate_subgraph(data, num_hops, node_index_to_watermark=node_idx, d_reduce=d_reduce, show=False, seed=seed)

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

def get_complement_subgraph(data, subgraph_dict):
    all_node_indices = torch.cat([subgraph_dict[k]['nodeIndices'] for k in subgraph_dict]).tolist()
    complement_indices = torch.tensor([i for i in range(len(data.x)) if i not in all_node_indices])

    num_nodes = data.num_nodes
    comp_sub_edge_index, _ = subgraph(complement_indices, data.edge_index, relabel_nodes=True, num_nodes=num_nodes)
    data_comp_sub = Data(
        x=data.x[complement_indices] if data.x is not None else None,
        edge_index=comp_sub_edge_index,
        y=data.y[complement_indices] if data.y is not None else None,
        train_mask=data.train_mask[complement_indices] if data.train_mask is not None else None,
        test_mask=data.test_mask[complement_indices] if data.test_mask is not None else None,
        val_mask=data.val_mask[complement_indices] if data.val_mask is not None else None,
    )
    return data_comp_sub, complement_indices

def normalize_features(features):
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True) + 1e-10  # to avoid division by zero
    normalized_features = (features - mean) / std
    return normalized_features

def filter_out_zero_features_from_unimportant_indices(unimportant_indices, subgraph_dict, watermark_kwargs):
    # Step 2: Filter out indices that are in zero_features_across_subgraphs
    sample_data = subgraph_dict[list(subgraph_dict.keys())[0]]['subgraph']
    num_features = sample_data.x.shape[1]
    features_all_subgraphs = torch.vstack([subgraph_dict[subgraph_central_node]['subgraph'].x for subgraph_central_node in subgraph_dict.keys()]).squeeze()
    zero_features_across_subgraphs = torch.where(torch.sum(features_all_subgraphs, dim=0) == 0)[0]

    filtered_unimportant_indices = [i for i in unimportant_indices if i not in zero_features_across_subgraphs]

    # Step 3: Ensure the number of unimportant indices remains the same
    num_unimportant_indices_needed = len(unimportant_indices)
    if len(filtered_unimportant_indices) < num_unimportant_indices_needed:
        remaining_indices = [i for i in range(num_features) if i not in filtered_unimportant_indices and i not in zero_features_across_subgraphs]
        additional_unimportant_indices = np.random.choice(remaining_indices, num_unimportant_indices_needed - len(filtered_unimportant_indices), replace=False)
        filtered_unimportant_indices = np.concatenate((filtered_unimportant_indices, additional_unimportant_indices))
    
    filtered_unimportant_indices = torch.tensor(filtered_unimportant_indices)
    return filtered_unimportant_indices




def select_indices_of_present_features(current_indices, num_indices, zero_features):
    indices = []
    i=0
    while len(indices)<num_indices:
        if current_indices[i] not in zero_features:
            try:
                indices.append(current_indices[i].item())
            except:
                indices.append(current_indices[i])
        i +=1 
    return torch.tensor(indices)


def compute_feature_variability_weights(data_objects):
    variablities = []
    for data_obj in data_objects:
        std_devs = data_obj.x.std(dim=0)
        variablity = std_devs#.mean()
        variablities.append(variablity)
    variablities = torch.vstack(variablities)
    weights = 1 / (variablities + 1e-10)
    return weights


# a method for the Trainer class
def test_perturb_x(self):
    node_classifier = copy.deepcopy(self.node_classifier)
    optimizer = copy.deepcopy(self.optimizer)
    subgraph_dict = copy.deepcopy(self.subgraph_dict)
    betas_dict = copy.deepcopy(self.betas_dict)
    beta_similarities_dict = copy.deepcopy(self.beta_similarities_dict)
    debug_multiple_subgraphs = False 
    beta_weights = copy.deepcopy(self.beta_weights)
    percent_matches = [[]]*config.subgraph_kwargs['numSubgraphs']
    optimizer.zero_grad()
    x = self.x.clone()
    x = x.requires_grad_(True)
    log_logits = node_classifier(x, self.edge_index, config.node_classifier_kwargs['dropout'])
    probas = log_logits.clone().exp()
    if config.optimization_kwargs['separate_forward_passes_per_subgraph']==True:
        probas_dict = self.separate_forward_passes_per_subgraph()
    else:
        probas_dict={}
    subgraph_dict = self.apply_watermark_()
    for _ in range(100):
        optimizer.zero_grad()
        log_logits = node_classifier(x, self.edge_index, config.node_classifier_kwargs['dropout'])
        probas = log_logits.clone().exp()
        if config.optimization_kwargs['separate_forward_passes_per_subgraph']==True:
            probas_dict = self.separate_forward_passes_per_subgraph()
        else:
            probas_dict={}
        node_classifier.eval()
        optimizer.zero_grad()
        loss_watermark,percent_matches = self.optimize_watermark_and_update_dicts(probas, probas_dict, subgraph_dict, betas_dict, 
                                                                                    beta_similarities_dict, False,
                                                                                    debug_multiple_subgraphs, beta_weights, 
                                                                                    [])
        print("percent matches:",percent_matches)
        x_grad = torch.autograd.grad(loss_watermark, x, retain_graph=True)[0]
        grad_norms = []
        for sig in subgraph_dict:
            indices = subgraph_dict[sig]['nodeIndices']
            x_grad_narrow = x_grad[indices]
            grad_norm = np.round(torch.norm(x_grad_narrow).item(),5)
            grad_norms.append(grad_norm)
        print('grad_norms:',grad_norms)
        print('loss_watermark:',loss_watermark)
        x = self.perturb_x(x, x_grad)
        self.perturbed_x = x
        for i, sig in enumerate(subgraph_dict.keys()):
            node_indices = subgraph_dict[sig]['nodeIndices']
            subgraph_dict[sig]['subgraph'].x = x[node_indices]

# a method for the Trainer class
def perturb_x(self, x, this_grad):
    perturbation = torch.zeros_like(self.data.x)
    perturbed_indices = self.all_subgraph_indices
    perturbation[perturbed_indices] = -config.optimization_kwargs['perturb_lr']*this_grad[perturbed_indices]
    x = x + perturbation
    return x



# a method for the Trainer class
def graphSAINTsampler_option_draft(self, print_every=10):
    if config.optimization_kwargs['graphSAINTsampling']==True:
        sampler = GraphSAINTRandomWalkSampler(self.data, batch_size=2000, num_steps=30, walk_length=2, method='node')
        
        for epoch in tqdm(range(self.epochs)):
            self.epoch=epoch
            for sampled_data in sampler:
                self.edge_index, self.x, self.y    = augment_data(sampled_data, self.node_aug, self.edge_aug, self.train_nodes_to_consider, self.all_subgraph_indices, sampling_used=True)
                wmk_optimization_condition_met_op1 = config.watermark_kwargs['watermark_type']=='basic' or config.watermark_kwargs['watermark_type']=='most_represented'
                wmk_optimization_condition_met_op2 = config.watermark_kwargs['watermark_type']=='unimportant' and self.epoch>=config.watermark_kwargs['unimportant_selection_kwargs']['clf_only_epochs']
                wmk_optimization_condition_met = wmk_optimization_condition_met_op1 or wmk_optimization_condition_met_op2

                ## this is where i left off -- see how sampling in this way works under the closure_watermark() function (specifically optimize_and_update () within it).
                ## might not be so straightforward.

                if not wmk_optimization_condition_met:
                    self.percent_matches = [0]*(len(self.subgraph_signatures))
                    closure = self.closure_primary
                elif wmk_optimization_condition_met:
                    # if config.optimization_kwargs['freeze_params_before_wmk']==True:
                        # self.instantiate_optimizer(freeze=True)
                    self.coefWmk = self.wmk_coef_schedule_dict[epoch]
                    closure = self.closure_watermark
                if config.optimization_kwargs['use_sam']==True:
                    self.optimizer.step(closure)
                else:
                    closure()
                    self.optimizer.step()          
            
            self.history = update_history_one_epoch(self.history, self.loss, self.loss_dict, self.acc_trn, self.acc_val, self.percent_matches, self.x)
            if self.epoch%print_every==0:
                print_epoch_status(self.epoch, self.loss_primary_weighted, self.acc_trn, self.acc_val, wmk_optimization_condition_met, self.loss_watermark_weighted, self.beta_similarity, False)
            gc.collect()
    else:
        for epoch in tqdm(range(self.epochs)):
            self.epoch=epoch
            self.edge_index, self.x, self.y    = augment_data(self.data, self.node_aug, self.edge_aug, self.train_nodes_to_consider, self.all_subgraph_indices)
            wmk_optimization_condition_met_op1 = config.watermark_kwargs['watermark_type']=='basic' or config.watermark_kwargs['watermark_type']=='most_represented'
            wmk_optimization_condition_met_op2 = config.watermark_kwargs['watermark_type']=='unimportant' and self.epoch>=config.watermark_kwargs['unimportant_selection_kwargs']['clf_only_epochs']
            wmk_optimization_condition_met = wmk_optimization_condition_met_op1 or wmk_optimization_condition_met_op2
            if not wmk_optimization_condition_met:
                self.percent_matches = [0]*(len(self.subgraph_signatures))
                closure = self.closure_primary
            elif wmk_optimization_condition_met:
                # if config.optimization_kwargs['freeze_params_before_wmk']==True:
                    # self.instantiate_optimizer(freeze=True)
                self.coefWmk = self.wmk_coef_schedule_dict[epoch]
                closure = self.closure_watermark
            if config.optimization_kwargs['use_sam']==True:
                self.optimizer.step(closure)
            else:
                closure()
                self.optimizer.step()          
            self.history = update_history_one_epoch(self.history, self.loss, self.loss_dict, self.acc_trn, self.acc_val, self.percent_matches, self.x)
            if self.epoch%print_every==0:
                print_epoch_status(self.epoch, self.loss_primary_weighted, self.acc_trn, self.acc_val, wmk_optimization_condition_met, self.loss_watermark_weighted, self.beta_similarity, False)
            gc.collect()
            

# a method for the Trainer class
def shift_subgraph_(self, p_to_swap, subgraph_node_indices):
    train_nodes_to_consider = self.train_nodes_to_consider
    data = self.data
    return shift_subgraph(p_to_swap, subgraph_node_indices, train_nodes_to_consider, data)

def shift_subgraph(p_to_swap, subgraph_node_indices, train_nodes_to_consider, data, seed):
    num_to_swap = int(p_to_swap*len(subgraph_node_indices))
    torch.manual_seed(seed)
    random_indices = torch.randperm(len(subgraph_node_indices))
    subgraph_node_indices = subgraph_node_indices[random_indices[:len(subgraph_node_indices)-num_to_swap]]
    filtered_tensor = train_nodes_to_consider[~train_nodes_to_consider.unsqueeze(1).eq(subgraph_node_indices).any(dim=1)]
    random_index = torch.randint(0, filtered_tensor.size(0), (num_to_swap,))
    random_element = filtered_tensor[random_index]
    subgraph_node_indices = torch.concatenate([subgraph_node_indices, random_element])

    sub_edge_index, _ = subgraph(subgraph_node_indices, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    shifted_subgraph = Data(
        x          = data.x[subgraph_node_indices]          if data.x is not None else None,
        edge_index = sub_edge_index,
        y          = data.y[subgraph_node_indices]          if data.y is not None else None,
        train_mask = data.train_mask[subgraph_node_indices] if data.train_mask is not None else None,
        test_mask  = data.test_mask[subgraph_node_indices]  if data.test_mask is not None else None,
        val_mask   = data.val_mask[subgraph_node_indices]   if data.val_mask is not None else None)
    return shifted_subgraph, subgraph_node_indices


def test_robustness_to_node_similar_subgraphs(node_classifier, data, subgraph_dict, verbose=True):
    all_subgraph_node_indices = []
    for sig in subgraph_dict.keys():
        nodeIndices = subgraph_dict[sig]['nodeIndices'].tolist()
        all_subgraph_node_indices += nodeIndices
    all_subgraph_indices = torch.tensor(all_subgraph_node_indices)
    sacrifice_method = config.optimization_kwargs['sacrifice_kwargs']['method']
    size_dataset = data.x.shape[0]
    train_nodes_to_use_mask = get_train_nodes_to_consider(data, all_subgraph_indices, sacrifice_method, size_dataset, train_with_test_set=False)
    averages = []
    for p_to_swap in [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]:
        percent_matches=[]
        for s, sig in enumerate(subgraph_dict.keys()):
            subgraph_node_indices = subgraph_dict[sig]['nodeIndices']
            shifted_subgraph, _ = shift_subgraph(p_to_swap, subgraph_node_indices,train_nodes_to_use_mask, data)
            ignore_zeros_from_subgraphs=False
            this_watermark = subgraph_dict[sig]['watermark']
            x_sub, edge_index_sub = shifted_subgraph.x, shifted_subgraph.edge_index
            node_classifier.eval()
            log_logits = node_classifier(x_sub,edge_index_sub)
            y_sub = log_logits.exp()
            _,not_omit_indices = get_omit_indices(x_sub, this_watermark,ignore_zeros_from_subgraphs=ignore_zeros_from_subgraphs) #indices where watermark is 0
            this_raw_beta = solve_regression(x_sub, y_sub, config.regression_kwargs['lambda'])
            watermark_non_zero   = this_watermark[not_omit_indices]
            this_sign_beta       = torch.sign(this_raw_beta[not_omit_indices])
            this_matches = len(torch.where(this_sign_beta==watermark_non_zero)[0])
            this_percent_match = 100*this_matches/len(watermark_non_zero)
            percent_matches.append(this_percent_match)
        if verbose:
            print('p:',p_to_swap,'avg % match:',np.mean(percent_matches))
        averages.append(np.mean(percent_matches))
    return averages


def backward(losses, optimizer, type_='primary', epoch=0, verbose=False, retain_graph=False, gn=None):
    use_pcgrad   = config.optimization_kwargs['use_pcgrad']
    loss = sum(losses)

    if use_pcgrad==True:
        optimizer.pc_backward(losses)
        if verbose==True:
            print(f"Epoch {epoch}: PCGrad backpropagation for multiple losses")
    elif use_pcgrad==False:
        loss.backward(retain_graph=retain_graph)
        if verbose==True:
            print(f"Epoch {epoch}: Regular backpropagation for multiple losses")



def give_subgraph_example(dataset_name, graph_to_watermark, numHops, compare_to_full=False, max_degree=None, seed=0):
    if dataset_name=='PubMed':
        node_indices_to_watermark = [18745, 18728, 18809]
    else:
        ranked_nodes = rank_training_nodes_by_degree(dataset_name, graph_to_watermark, max_degree=max_degree)
        node_indices_to_watermark = ranked_nodes[:1]

    for node_index_to_watermark in node_indices_to_watermark:
        print(node_index_to_watermark)
        data_sub, _, subgraph_node_idx = generate_subgraph(graph_to_watermark, dataset_name, numHops, node_index_to_watermark=node_index_to_watermark, show=True, seed=seed)
        if compare_to_full==True:
            subgraph_node_idx, subgraph_edge_idx, _, _ = k_hop_subgraph(node_index_to_watermark, numHops, edge_index=graph_to_watermark.edge_index, num_nodes=graph_to_watermark.num_nodes, relabel_nodes=True)
            data_sub = Data(x=graph_to_watermark.x[subgraph_node_idx], edge_index=subgraph_edge_idx, y=graph_to_watermark.y[subgraph_node_idx])
            G_sub = to_networkx(data_sub, to_undirected=True)
            plt.figure(figsize=(5, 3))
            nx.draw_networkx(G_sub, with_labels=False,  node_color = 'blue', node_size=30)
            plt.title(f'{numHops}-hop subgraph centered at node {node_index_to_watermark} -- training mask not applied')
            plt.show()

def generate_basic_watermark(k, p_neg_ones=0.5, p_remove=0.75):
    j = int(p_neg_ones*k)
    watermark = torch.ones(k)
    watermark_neg_1_indices = torch.randperm(k)[:j]
    watermark[watermark_neg_1_indices] = -1

    j_0 = int(p_remove*k)
    watermark_remove_indices = torch.randperm(k)[:j_0]
    watermark[watermark_remove_indices] = 0
    return watermark

    
def name_compare_dict(dataset_name, optimization_kwargs, node_classifier_kwargs, watermark_kwargs, subgraph_kwargs, augment_kwargs, watermark_loss_kwargs):
    folder_name = get_results_folder_name(dataset_name, optimization_kwargs, node_classifier_kwargs, watermark_kwargs, subgraph_kwargs, augment_kwargs, watermark_loss_kwargs)
    compare_dict_name = f"compare_dicts_{folder_name}"
    return compare_dict_name



def compute_watermark_loss(subgraph_dict, probas, beta_weights):
    watermark_loss_kwargs = config.watermark_loss_kwargs
    regression_kwargs = config.regression_kwargs
    optimization_kwargs = config.optimization_kwargs

    loss_watermark = torch.tensor(0.0)
    for s, sig in enumerate(subgraph_dict.keys()):
        this_watermark, data_sub, subgraph_node_indices = [subgraph_dict[sig][k] for k in ['watermark','subgraph','nodeIndices']]
        x_sub, y_sub = data_sub.x, probas[subgraph_node_indices]
        ''' epoch condtion: epoch==epoch-1'''
        omit_indices,not_omit_indices = get_omit_indices(x_sub, this_watermark,ignore_zeros_from_subgraphs=False) #indices where watermark is 0
        raw_beta            = solve_regression(x_sub, y_sub, regression_kwargs['lambda'])
        beta                = process_beta(raw_beta, omit_indices)
        B_x_W = (beta*this_watermark).clone()
        B_x_W = B_x_W[not_omit_indices]

        beta_weights_ = beta_weights[s]
        beta_weights_ = beta_weights_[not_omit_indices]
        loss_watermark = loss_watermark+torch.mean(torch.clamp(watermark_loss_kwargs['epsilon']-B_x_W, min=0)*beta_weights_)


    loss_watermark = loss_watermark/len(subgraph_dict)
    loss_watermark_scaled = loss_watermark*optimization_kwargs['coefWmk_kwargs']['coefWmk']
    return loss_watermark_scaled