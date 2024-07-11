import copy
import gc
import grafog.transforms as T
import itertools
import numpy as np
import numpy as np 
import pandas as pd
from   pcgrad.pcgrad import PCGrad # from the following: https://github.com/WeiChengTseng/Pytorch-PCGrad. Renamed to 'pcgrad' and moved to site-packages folder.
import pickle
import random
from   scipy import stats
from   sklearn.model_selection import train_test_split
import textwrap
from   tqdm.notebook import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F


import torch
import torch.nn as nn
import torchviz
from torchviz import make_dot


from pyHSICLasso import HSICLasso


from torch_geometric.data import Data  
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx, k_hop_subgraph, subgraph
from torch_geometric.transforms import BaseTransform, NormalizeFeatures, Compose


import config
from general_utils import *
from models import *
from regression_utils import *
from subgraph_utils import *
from transform_functions import *
from watermark_utils import *

torch.manual_seed(2)


import torch.nn.functional as F


def prep_data(dataset_name='CORA', 
              location='default', 
              batch_size='default',
              transform_list = 'default', #= NormalizeFeatures())
              train_val_test_split=[0.6,0.2,0.2]
              ):
    class_ = dataset_attributes[dataset_name]['class']
    single_or_multi_graph = dataset_attributes[dataset_name]['single_or_multi_graph']

    if location=='default':
        location = '../data' if dataset_name in ['CORA','CiteSeer','PubMed','computers','photo','PPI'] else f'../data/{dataset_name}' if dataset_name in ['Flickr','Reddit','Reddit2'] else None

    if batch_size=='default':
        batch_size = 'All'

    if transform_list=='default':
        transform_list = []
        if dataset_name in ['CORA','CiteSeer']:
            transform_list =[ChooseLargestMaskForTrain()]
        if dataset_name in ['PubMed','Flickr']:
            transform_list = [NormalizeFeatures(), ChooseLargestMaskForTrain()]
        if dataset_name in ['computers', 'photo']:
            transform_list = [CreateMaskTransform(*train_val_test_split)]
        if dataset_name in ['Reddit','Reddit2']:
            transform_list = [KHopsFractionDatasetTransform(1,3)]
    transform = Compose(transform_list)
    
    print(f'Transorms used when loading {dataset_name}: {[str(t) for t in transform_list]}')

    if single_or_multi_graph=='single':
        if dataset_name in ['Reddit', 'Reddit2','Flickr']:
            dataset = class_(location, transform)
        else:
            dataset = class_(location, dataset_name, transform=transform)
        dataset = add_indices(dataset)
        print("train_mask:", torch.sum(dataset[0].train_mask).item())
        print("test_mask:",  torch.sum(dataset[0].test_mask).item())
        print("val_mask:",   torch.sum(dataset[0].val_mask).item())
        return dataset
    
    elif single_or_multi_graph=='multi':
        train_dataset = class_(location, split='train', transform=transform)
        train_dataset = add_indices(train_dataset)
        val_dataset   = class_(location, split='val',   transform=transform)
        val_dataset = add_indices(val_dataset)
        test_dataset  = class_(location, split='test',  transform=transform)
        test_dataset = add_indices(test_dataset)

        train_dataset.y = torch.argmax(train_dataset.y, dim=1)
        val_dataset.y   = torch.argmax(val_dataset.y,   dim=1)
        test_dataset.y  = torch.argmax(test_dataset.y,  dim=1)

        batch_size = len(train_dataset) if batch_size=='All' else batch_size

        train_loader = DataLoader(train_dataset,    batch_size=batch_size,  shuffle=True)
        val_loader   = DataLoader(val_dataset,      batch_size=2,           shuffle=False)
        test_loader  = DataLoader(test_dataset,     batch_size=2,           shuffle=False)
        
        return [train_dataset, val_dataset, test_dataset], [train_loader, val_loader, test_loader]
    


def accuracy(output, labels, verbose=False):
    output, labels = output.clone().detach(), labels.clone().detach()
    _, pred = output.max(dim=1)
    correct = pred.eq(labels).double()
    if verbose==True:
        print('correct:',correct.sum(),'output:',len(output),'labels:',len(labels))
    correct = copy.deepcopy(correct.sum()) # i don't think the copy.deepcopy is necessary
    return correct / len(labels)

def sacrifice_node_indices(train_node_indices,method,p_sacrifice,subgraph_node_indices=None):
    if method=='subgraph_node_indices':
        assert subgraph_node_indices is not None
        group = subgraph_node_indices
        print(f'Sacrificing {100*p_sacrifice}% of subgraph nodes from node classification training')
    elif method=='train_node_indices':
        group=train_node_indices
        print(f'Sacrificing {100*p_sacrifice}% of train set nodes from node classification training')

    num_sacrifice = int(p_sacrifice * len(group))
    rand_perm_sacrifice = torch.randperm(len(group))[:num_sacrifice]
    sacrifice_these = group[rand_perm_sacrifice]
    train_node_indices_sans_sacrificed = train_node_indices[~torch.isin(train_node_indices, sacrifice_these)]
    assert torch.all(torch.isin(sacrifice_these,group))
    return train_node_indices_sans_sacrificed 


def get_eval_acc(model, loader):
    acc=0
    model.eval()
    
    for i, batch in enumerate(loader):
        log_logits = model(batch.x, batch.edge_index)
        
        if len(batch.y.shape)>1:
            batch.y = torch.argmax(batch.y,dim=1) # convert from one-hot encoding for F.nll_loss
        
        batch_acc   = accuracy(log_logits, batch.y)
        acc         += batch_acc.clone().detach()

    return acc


def extract_results_random_subgraphs(data, dataset_name, fraction, numSubgraphs, alpha, watermark, probas, node_classifier, subgraph_kwargs, watermark_loss_kwargs, regression_kwargs, use_train_mask=False):
    subgraph_kwargs['numSubgraphs']=numSubgraphs
    subgraph_kwargs['fraction']=fraction
    if subgraph_kwargs['method']=='khop':
        num_nodes = data.x.shape[0]
        node_indices_to_watermark = random.sample(list(range(num_nodes)),numSubgraphs) 
        subgraph_kwargs['khop_kwargs']['nodeIndices'] = node_indices_to_watermark
    elif subgraph_kwargs['method']=='random':
        pass

    subgraph_dict, all_subgraph_indices = collect_subgraphs_within_single_graph_for_watermarking(data, dataset_name, use_train_mask, subgraph_kwargs)
    betas_dict  = {k:[] for k in subgraph_dict.keys()}
    beta_similarities_dict  = {k:None for k in subgraph_dict.keys()}

    if probas is None:
        node_classifier.eval()
        log_logits = node_classifier(data.x, data.edge_index)
        probas = log_logits.clone().exp()
    else:
        pass
    for sig in subgraph_dict.keys():

        data_sub = subgraph_dict[sig]['subgraph']    
        subgraph_node_indices = subgraph_dict[sig]['nodeIndices']
    
        x_sub = data_sub.x
        y_sub = probas[subgraph_node_indices]

        omit_indices,not_omit_indices = get_omit_indices(x_sub, watermark,ignore_zeros_from_subgraphs=False)
        beta = process_beta(solve_regression(x_sub, y_sub, regression_kwargs['lambda']), alpha, omit_indices, watermark_loss_kwargs['scale_beta_method'])
        betas_dict[sig].append(beta.clone().detach())
        beta_similarities_dict[sig] = torch.sum(beta*watermark)

    return betas_dict, beta_similarities_dict



def collect_augmentations():#augment_kwargs, outDim):
    augment_kwargs = config.augment_kwargs
    outDim = config.node_classifier_kwargs['outDim']
    node_augs = []
    if augment_kwargs['nodeDrop']['use']==True:
        node_augs.append(T.NodeDrop(augment_kwargs['nodeDrop']['p']))
    if augment_kwargs['nodeMixUp']['use']==True:
        node_augs.append(T.NodeMixUp(lamb=augment_kwargs['nodeMixUp']['lambda'], classes=outDim))
    if augment_kwargs['nodeFeatMask']['use']==True:
        node_augs.append(T.NodeFeatureMasking(p=augment_kwargs['nodeFeatMask']['p']))
    edge_augs = []
    if augment_kwargs['edgeDrop']['use']==True:
        edge_augs.append(T.NodeDrop(augment_kwargs['edgeDrop']['p']))
    node_aug = T.Compose(node_augs) if len(node_augs)>0 else None
    edge_aug = T.Compose(edge_augs) if len(edge_augs)>0 else None
    return node_aug, edge_aug


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



def get_omit_indices(x_sub, watermark, ignore_zeros_from_subgraphs=True):
    if ignore_zeros_from_subgraphs==True:
        zero_features_within_subgraph = torch.where(torch.sum(x_sub, dim=0) == 0)
    else:
        zero_features_within_subgraph = torch.tensor([[]])
    zero_indices_within_watermark = torch.where(watermark==0)
    omit_indices = torch.tensor(list(set(zero_features_within_subgraph[0].tolist() + zero_indices_within_watermark[0].tolist())))
    not_omit_indices = torch.tensor([i for i in range(x_sub.shape[1]) if i not in omit_indices])
    return omit_indices, not_omit_indices



def process_beta(beta, alpha, omit_indices, scale_beta_method='clip'):
    if scale_beta_method=='tanh':
        beta = torch.tanh(alpha*beta)
    elif scale_beta_method=='tan':
        beta = torch.tan(alpha*beta)
    elif scale_beta_method=='clip':
        beta = torch.clip(beta,min=-1,max=1)
    elif scale_beta_method==None:
        pass
    beta = beta.clone()  # Avoid in-place operation
    if omit_indices is not None and len(omit_indices)>0:
        beta[omit_indices] = 0 # zero out non-contributing indices
    return beta

def get_one_minus_B_x_W(beta, watermark, omit_indices):
    one_minus_B_x_W = 1-beta*watermark
    one_minus_B_x_W = one_minus_B_x_W.clone() # Avoid in-place operation
    if omit_indices is not None and len(omit_indices)>0:
        one_minus_B_x_W[omit_indices] = 0 # zero out non-contributing indices
    return one_minus_B_x_W


def augment_data(data, node_aug, edge_aug, train_nodes_to_consider, all_subgraph_indices):
    new_data = copy.deepcopy(data)
    new_data.x = new_data.x.detach().clone()  # Use clone instead of deepcopy

    if config.augment_kwargs['separate_trainset_from_subgraphs'] == True:
        trn_minus_subgraph_nodes = torch.tensor(list(set(train_nodes_to_consider.tolist())-set(all_subgraph_indices)))
        train_minus_subgraph_data = get_subgraph_from_node_indices(copy.deepcopy(data), trn_minus_subgraph_nodes)
        if node_aug is not None:
            train_minus_subgraph_data = node_aug(train_minus_subgraph_data)
        if edge_aug is not None:
            train_minus_subgraph_data = edge_aug(train_minus_subgraph_data)
        edge_index_trn, x_trn, y_trn = train_minus_subgraph_data.edge_index, train_minus_subgraph_data.x, train_minus_subgraph_data.y
        new_data.x[trn_minus_subgraph_nodes] = x_trn
        new_data.y[trn_minus_subgraph_nodes] = y_trn
        mask_trn = torch.isin(new_data.edge_index[0], trn_minus_subgraph_nodes) & torch.isin(new_data.edge_index[1], trn_minus_subgraph_nodes)
        new_data.edge_index[:, mask_trn] = edge_index_trn
        if config.augment_kwargs['ignore_subgraphs']==False:
            subgraph_data = get_subgraph_from_node_indices(copy.deepcopy(data), all_subgraph_indices)
            if node_aug is not None:
               subgraph_data = node_aug(subgraph_data)
            if edge_aug is not None:
               subgraph_data = edge_aug(subgraph_data)
            edge_index_sub, x_sub, y_sub = subgraph_data.edge_index, subgraph_data.x, subgraph_data.y
            new_data.x[all_subgraph_indices] = x_sub
            new_data.y[all_subgraph_indices] = y_sub
            mask_sub = torch.isin(new_data.edge_index[0], all_subgraph_indices) & torch.isin(new_data.edge_index[1], all_subgraph_indices)
            new_data.edge_index[:, mask_sub] = edge_index_sub
        edge_index, x, y = new_data.edge_index, new_data.x, new_data.y
    elif config.augment_kwargs['separate_trainset_from_subgraphs'] == False:
        if node_aug is not None:
            new_data = node_aug(new_data)
        if edge_aug is not None:
            new_data = edge_aug(new_data)
        edge_index, x, y = new_data.edge_index, new_data.x, new_data.y
    return edge_index, x, y


def initialize_training():
    lr = config.optimization_kwargs['lr']
    node_classifier_kwargs = config.node_classifier_kwargs
    node_classifier = Net(**node_classifier_kwargs)
    optimizer = optim.Adam(node_classifier.parameters(), lr=lr)
    node_classifier.train()
    return node_classifier, optimizer

def setup_history(clf_only=False, subgraph_signatures=None):
    if clf_only==False:
        assert subgraph_signatures is not None
    history = {
        'losses': [], 
        'losses_primary': [], 'losses_watermark': [], 'regs':[], 
        'losses_primary_weighted': [], 'losses_watermark_weighted': [], 'regs_weighted':[], 
        'betas': [], 'beta_similarities': [], 'train_accs': [], 'val_accs': [], 'percent_matches': [], 'xs_perturbed': []
    }
    betas_dict = {sig: [] for sig in subgraph_signatures} if clf_only==False else {}
    beta_similarities_dict = {sig: None for sig in subgraph_signatures} if clf_only==False else {}
    return history, betas_dict, beta_similarities_dict

def setup_subgraph_dict(data, dataset_name):
    print('setup_subgraph_dict')
    optimization_kwargs=config.optimization_kwargs
    subgraph_kwargs = config.subgraph_kwargs
    watermark_kwargs = config.watermark_kwargs
    subgraph_dict, all_subgraph_indices = collect_subgraphs_within_single_graph_for_watermarking(data, dataset_name, use_train_mask=True, subgraph_kwargs=subgraph_kwargs)

    if config.optimization_kwargs['use_summary_beta']==True:
        subgraph_dict = add_summary_subgraph(subgraph_dict, data)
        all_subgraph_node_indices = []
        for sig in subgraph_dict.keys():
            nodeIndices = subgraph_dict[sig]['nodeIndices'].tolist()
            all_subgraph_node_indices += nodeIndices
        all_subgraph_node_indices = torch.tensor(all_subgraph_node_indices)
    subgraph_signatures = list(subgraph_dict.keys())
    if watermark_kwargs['watermark_type']=='basic':
        subgraph_dict = apply_basic_watermark(data, subgraph_dict, watermark_kwargs)

    return subgraph_dict, subgraph_signatures, all_subgraph_indices

def print_epoch_status(epoch, loss_primary, acc_trn, acc_val, condition_met=False, loss_watermark=None, beta_similarity=None, clf_only=True):
    if clf_only==True:
        epoch_printout = f'Epoch: {epoch:3d}, loss_primary = {loss_primary:.3f}, train acc = {acc_trn:.3f}, val acc = {acc_val:.3f}'
    elif clf_only==False:
        if condition_met:
           epoch_printout = f'Epoch: {epoch:3d}, loss_primary = {loss_primary:.3f}, loss_watermark = {loss_watermark:.3f}, B*W = {beta_similarity:.5f}, train acc = {acc_trn:.3f}, val acc = {acc_val:.3f}'
        else:
          epoch_printout = f'Epoch: {epoch:3d}, loss_primary = {loss_primary:.3f}, loss_watermark = n/a, B*W = n/a, train acc = {acc_trn:.3f}, val acc = {acc_val:.3f}'
    print(epoch_printout)
 


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
        beta                = process_beta(raw_beta, watermark_loss_kwargs['alpha'], omit_indices, watermark_loss_kwargs['scale_beta_method'])
        B_x_W = (beta*this_watermark).clone()
        B_x_W = B_x_W[not_omit_indices]

        balanced_beta_weights = beta_weights[s]
        balanced_beta_weights = balanced_beta_weights[not_omit_indices]
        loss_watermark = loss_watermark+torch.mean(torch.clamp(watermark_loss_kwargs['epsilon']-B_x_W, min=0)*balanced_beta_weights)


    loss_watermark = loss_watermark/len(subgraph_dict)
    loss_watermark_scaled = loss_watermark*optimization_kwargs['coefWmk']
    return loss_watermark_scaled



def optimize_watermark(probas,
                       subgraph_dict,sig,
                       #betas_dict, 
                       #beta_similarities_dict, 
                       #epoch_condition, 
                       ignore_zeros_from_subgraphs=False, 
                       debug=False,
                       balanced_beta_weights=None):
    
    watermark_loss_kwargs = config.watermark_loss_kwargs
    regression_kwargs = config.regression_kwargs
    this_watermark, data_sub, subgraph_node_indices = [subgraph_dict[sig][k] for k in ['watermark','subgraph','nodeIndices']]
    x_sub, y_sub = data_sub.x, probas[subgraph_node_indices]
    ''' epoch condtion: epoch==epoch-1'''
    omit_indices,not_omit_indices = get_omit_indices(x_sub, this_watermark,ignore_zeros_from_subgraphs=ignore_zeros_from_subgraphs) #indices where watermark is 0
    this_raw_beta = solve_regression(x_sub, y_sub, regression_kwargs['lambda'])
    # print('raw beta:',raw_beta)
    # hsic_lasso = HSICLasso()
    # hsic_lasso.input(x_sub.detach().numpy(), y_sub.detach().numpy(),kernelX='Gaussian', kernelY='Gaussian')
    # hsic_lasso.classification(x_sub.shape[1])
    # b = hsic_lasso.get_index_score()
    # print('b:',b)

    beta                = process_beta(this_raw_beta, watermark_loss_kwargs['alpha'], omit_indices, watermark_loss_kwargs['scale_beta_method'])
    B_x_W = (beta*this_watermark).clone()
    B_x_W = B_x_W[not_omit_indices]

    balanced_beta_weights = balanced_beta_weights[not_omit_indices]
    this_loss_watermark = torch.mean(torch.clamp(watermark_loss_kwargs['epsilon']-B_x_W, min=0)*balanced_beta_weights)
    this_beta_similarity = torch.mean(B_x_W)
    # loss_watermark  = loss_watermark+this_loss_watermark
    # beta_similarity += this_beta_similarity
    if debug:
        print(f"Subgraph: Loss Watermark: {this_loss_watermark.item()}, Beta Similarity: {this_beta_similarity.item()}")
    # if epoch_condition:
    #     beta_similarities_dict[sig] = this_beta_similarity.clone().detach()
    # betas_dict[sig].append(raw_beta)
    watermark_non_zero   = this_watermark[not_omit_indices]
    this_sign_beta       = torch.sign(this_raw_beta[not_omit_indices])
    this_matches = len(torch.where(this_sign_beta==watermark_non_zero)[0])
    this_percent_match = 100*this_matches/len(watermark_non_zero)

    # this_raw_beta_detached = raw_beta.clone().detach()

    return this_loss_watermark, this_percent_match, this_beta_similarity, this_raw_beta #beta_similarities_dict, raw_beta#_detached

def get_reg_term(betas_from_every_subgraph):
    regularization_type = config.optimization_kwargs['regularization_type']
    lambda_l2 = config.optimization_kwargs['lambda_l2']
    if regularization_type==None:
        return None
    else:
        if regularization_type=='L2':
            reg = sum(torch.norm(betas_from_every_subgraph[i]) for i in range(len(betas_from_every_subgraph)))
            reg = reg*lambda_l2
        elif regularization_type=='beta_var':
            inter_tensor_variance = torch.std(betas_from_every_subgraph, dim=0, unbiased=False)
            reg = torch.sum(inter_tensor_variance)
        return reg


def compute_feature_variability_weights(data_objects):
    variablities = []
    for data_obj in data_objects:
        std_devs = data_obj.x.std(dim=0)
        variablity = std_devs#.mean()
        variablities.append(variablity)
    variablities = torch.vstack(variablities)
    weights = 1 / (variablities + 1e-10)
    return weights

def compute_overall_feature_representation(population_features):
    overall_feature_sums = population_features.sum(dim=0)
    overall_feature_representations = overall_feature_sums / overall_feature_sums.sum()
    return overall_feature_representations

def compute_subgraph_feature_representation(x_sub):
    subgraph_feature_sums = x_sub.sum(dim=0)
    subgraph_feature_representations = subgraph_feature_sums / subgraph_feature_sums.sum()
    return subgraph_feature_representations

def compute_balanced_feature_weights(overall_representations, subgraph_representations):
    weights = overall_representations / (subgraph_representations + 1e-10)
    return weights


def get_balanced_beta_weights(subgraphs):
    all_subgraph_features = torch.vstack([subgraph.x for subgraph in subgraphs])
    overall_feature_representations = compute_overall_feature_representation(all_subgraph_features)
    subgraph_feature_representations = torch.vstack([compute_subgraph_feature_representation(subgraph.x) for subgraph in subgraphs])
    balanced_weights = compute_balanced_feature_weights(overall_feature_representations, subgraph_feature_representations)
    return balanced_weights


def get_subgraph_from_node_indices(data, node_indices):
    sub_edge_index, _ = subgraph(node_indices, data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes)
    sub_data = Data(
        x=data.x[node_indices] if data.x is not None else None,
        edge_index=sub_edge_index,
        y=data.y[node_indices] if data.y is not None else None,
        train_mask=data.train_mask[node_indices] if data.train_mask is not None else None,
        test_mask=data.test_mask[node_indices] if data.test_mask is not None else None,
        val_mask=data.val_mask[node_indices] if data.val_mask is not None else None)
    
    return sub_data

def add_summary_subgraph(subgraph_dict, data):
    all_indices = torch.concat([subgraph_dict[k]['nodeIndices'] for k in subgraph_dict.keys()])
    sub_data = get_subgraph_from_node_indices(data, all_indices)
    # sub_edge_index, _ = subgraph(all_indices, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    # sub_data = Data(
    #     x=data.x[all_indices] if data.x is not None else None,
    #     edge_index=sub_edge_index,
    #     y=data.y[all_indices] if data.y is not None else None,
    #     train_mask=data.train_mask[all_indices] if data.train_mask is not None else None,
    #     test_mask=data.test_mask[all_indices] if data.test_mask is not None else None,
    #     val_mask=data.val_mask[all_indices] if data.val_mask is not None else None)
    sig = '_'.join([str(s) for s in all_indices.tolist()])
    subgraph_dict[sig]={}
    subgraph_dict[sig]['subgraph']=sub_data
    subgraph_dict[sig]['nodeIndices']=all_indices
    return subgraph_dict

def get_train_nodes_to_consider(data, subgraph_dict, all_subgraph_indices, sacrifice_method, size_dataset):
    train_mask = data.train_mask
    edge_index = data.edge_index
    
    train_nodes_not_neighboring_subgraphs_mask = copy.deepcopy(train_mask)
    if config.optimization_kwargs['ignore_subgraph_neighbors']==True:# and optimization_kwargs['skip_clf']==False:
        all_neighbors=set()
        subgraph_mask = torch.zeros_like(train_mask, dtype=torch.bool)
        for sig in subgraph_dict.keys():
            nodeIndices = subgraph_dict[sig]['nodeIndices']
            subgraph_mask[nodeIndices] = True
            for node in nodeIndices:
                neighbors = edge_index[1][edge_index[0] == node]
                all_neighbors.update(neighbors.tolist())
        all_neighbors = torch.tensor(list(all_neighbors))
        all_neighbors_mask = torch.zeros_like(train_mask, dtype=torch.bool)
        all_neighbors_mask[all_neighbors] = True
        train_nodes_not_neighboring_subgraphs_mask = train_mask & all_neighbors_mask & ~subgraph_mask

    train_nodes_not_sacrificed_mask = copy.deepcopy(train_mask)
    if sacrifice_method is not None:
        train_node_indices = torch.arange(size_dataset)[train_mask]
        p_sacrifice = config.optimization_kwargs['sacrifice_kwargs']['percentage']
        train_nodes_not_sacrificed = sacrifice_node_indices(train_node_indices,sacrifice_method,p_sacrifice,all_subgraph_indices)
        train_nodes_not_sacrificed_mask = torch.zeros_like(train_mask, dtype=torch.bool)
        train_nodes_not_sacrificed_mask[train_nodes_not_sacrificed] = True

    train_nodes_to_use_mask = train_nodes_not_neighboring_subgraphs_mask & train_nodes_not_sacrificed_mask

    # out_mask = train_mask

    # optimization_kwargs = config.optimization_kwargs
    # train_nodes_not_neighboring_subgraphs = torch.tensor([])
    # if optimization_kwargs['ignore_subgraph_neighbors']==True and optimization_kwargs['skip_clf']==False:
    #     all_neighbors=set()
    #     subgraph_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    #     for sig in subgraph_dict.keys():
    #         nodeIndices = subgraph_dict[sig]['nodeIndices']
    #         subgraph_mask[nodeIndices] = True
    #         for node in nodeIndices:
    #             neighbors = edge_index[1][edge_index[0] == node]
    #             all_neighbors.update(neighbors.tolist())

    #     all_neighbors = torch.tensor(list(all_neighbors))
    #     train_neighbors = all_neighbors[train_mask[all_neighbors]]
    #     train_nodes_not_neighboring_subgraphs = train_neighbors[~subgraph_mask[train_neighbors]]

    #     out_mask = torch.concat([out_mask,train_nodes_not_neighboring_subgraphs]).unique().int()


    # train_nodes_not_sacrificed = torch.tensor([])
    # if sacrifice_method is not None:
    #     train_node_indices = torch.arange(size_dataset)[train_mask]
    #     p_sacrifice = config.optimization_kwargs['sacrifice_kwargs']['percentage']
    #     train_nodes_not_sacrificed = sacrifice_node_indices(train_node_indices,sacrifice_method,p_sacrifice,all_subgraph_indices)

    #     out_mask = torch.concat([out_mask,train_nodes_not_sacrificed]).unique().int()

    return train_nodes_to_use_mask


def train_clf_only(data, dataset_name, save=True, print_every=1,separate_nodemix=True,ignore_subgraphs_nodemix=True):
    loss=None
    optimization_kwargs = config.optimization_kwargs
    sacrifice_method = optimization_kwargs['sacrifice_kwargs']['method']
   # train_nodes_to_consider = get_train_nodes_to_consider(None, sacrifice_method, data.x.shape[0], data.train_mask)
    train_nodes_to_consider_mask = get_train_nodes_to_consider(data, None, None, sacrifice_method, data.x.shape[0])
    validate_kwargs()
    node_aug, edge_aug         = collect_augmentations()
    history, _, _              = setup_history(clf_only=optimization_kwargs['clf_only'])
    loss_dict                  = setup_loss_dict()
    node_classifier, optimizer = initialize_training()
    if optimization_kwargs['use_pcgrad']==True:
        print('Defaulting to regular Adam optimizer since only one learning task (node classification).')

    epochs = optimization_kwargs['epochs']
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()

        edge_index, x, y    = augment_data(data, node_aug, edge_aug)
        log_logits          = node_classifier(x, edge_index)
        loss_primary        = F.nll_loss(log_logits[train_nodes_to_consider_mask], y[train_nodes_to_consider_mask])
        loss = loss_primary
        loss.backward()
        optimizer.step()

        acc_trn = accuracy(log_logits[data.train_mask], y[data.train_mask],verbose=False)
        acc_val = accuracy(log_logits[data.val_mask],   y[data.val_mask], verbose=False)
        loss_dict['loss_primary']=loss_primary



        history = update_history_one_epoch(history, #[loss_primary,None,None], [None,None,None], 
                                           loss,
                                           loss_dict,
                                           acc_trn, acc_val, 
                                           None, None)
        if epoch%print_every==0:
            print_epoch_status(epoch, loss_primary, acc_trn, acc_val, clf_only=True)

    history['betas'], history['beta_similarities'] = {},{} ## include for consistency with watermarking outputs

    if save==True:
        save_results(dataset_name, node_classifier, history)

    return node_classifier, history



def get_beta_weights(subgraph_dict, num_features):
    if config.watermark_loss_kwargs['balance_beta_weights'] == True:
        beta_weights = get_balanced_beta_weights([subgraph_dict[sig]['subgraph'] for sig in subgraph_dict.keys()])
    elif config.watermark_loss_kwargs['balance_beta_weights'] == False:
        beta_weights = torch.ones(len(subgraph_dict),num_features)
    return beta_weights


    

def optimize_and_update(probas,subgraph_dict, betas_dict, beta_similarities_dict, is_last_epoch,
                        debug_multiple_subgraphs, beta_weights, percent_matches):
    loss_watermark, beta_similarity = torch.tensor(0.0, requires_grad=True), torch.tensor(0.0)
    betas_from_every_subgraph = []
    for s, sig in enumerate(subgraph_dict.keys()):
        this_loss_watermark, this_percent_match, this_beta_similarity,this_raw_beta = optimize_watermark(
                                                                        #loss_watermark, beta_similarity, 
                                                                       probas, subgraph_dict, sig, 
                                                                       #betas_dict, 
                                                                       #beta_similarities_dict, 
                                                                        #is_last_epoch, 
                                                                        ignore_zeros_from_subgraphs=False, 
                                                                        debug=debug_multiple_subgraphs,
                                                                        balanced_beta_weights=beta_weights[s])
        loss_watermark = loss_watermark + this_loss_watermark
        beta_similarity = beta_similarity + this_beta_similarity
        if is_last_epoch:
            beta_similarities_dict[sig] = this_beta_similarity.clone().detach()
        percent_matches.append(this_percent_match)
        betas_dict[sig].append(this_raw_beta)
        betas_from_every_subgraph.append(this_raw_beta)

    betas_from_every_subgraph = torch.vstack(betas_from_every_subgraph)
    loss_watermark = loss_watermark/len(subgraph_dict)
    beta_similarity = beta_similarity/len(subgraph_dict)
    return loss_watermark, beta_similarity, betas_from_every_subgraph, betas_dict, beta_similarities_dict, percent_matches

def check_grads(node_classifier, epoch, tag='A'):
    grad_norm = 0
    for param in node_classifier.parameters():
        if param.grad is not None:
            grad_norm += param.grad.norm().item()
    print(f"Epoch {epoch} {tag}: Gradient norm = {grad_norm}")

def torch_add_not_None(list_):
    sum_ = torch.tensor(0)
    for item_ in list_:
        if item_ is not None:
            sum_ = sum_ + item_
    return sum_

def setup_loss_dict():
    loss_dict = {'loss_primary':torch.tensor(0.0),   'loss_watermark': torch.tensor(0.0),   'reg':torch.tensor(0.0),
                 'loss_primary_weighted':torch.tensor(0.0),  'loss_watermark_weighted':torch.tensor(0.0),     'reg_weighted':torch.tensor(0.0)}
    return loss_dict

def get_weighted_losses(type_='primary',
                        loss_dict={},
                        loss_primary=None, loss_watermark=None, 
                        reg=None, gn=None):

    loss_dict['loss_primary']=loss_primary
    loss_dict['loss_watermark']=loss_watermark
    loss_dict['reg']=reg

    use_gradnorm = config.optimization_kwargs['use_gradnorm']
    coef_wmk     = config.optimization_kwargs['coefWmk']


    assert type_ in ['primary','combined']
    if type_=='primary':
        assert loss_primary is not None
        loss_dict['loss_primary_weighted']=loss_primary
    elif type_=='combined':
        assert loss_watermark is not None
        if use_gradnorm==False:
            assert coef_wmk is not None
            loss_watermark_weighted = loss_watermark*coef_wmk 
            loss_dict['loss_primary_weighted'] = loss_primary # weight not changed by this process
            loss_dict['loss_watermark_weighted'] = loss_watermark_weighted
            loss_dict['reg_weighted'] = reg # weight not changed by this process

        elif use_gradnorm==True:
            assert gn is not None
            weighted_losses = gn.get_weighted_losses_gradnorm(loss_primary, loss_watermark*coef_wmk, reg)
            loss_dict['loss_primary_weighted'] = weighted_losses[0]
            loss_dict['loss_watermark_weighted'] = weighted_losses[1]
            try:
                loss_dict['reg_weighted'] = weighted_losses[2]
            except:
                loss_dict['reg_weighted'] = None

    unweighted_total = torch_add_not_None([loss_dict[k] for k in ['loss_primary','loss_watermark','reg']])
    weighted_total = torch_add_not_None([loss_dict[k] for k in ['loss_primary_weighted','loss_watermark_weighted','reg_weighted']])

    return loss_dict, unweighted_total, weighted_total


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



def replace_Nones_with_0s(values):
    if isinstance(values[0],torch.Tensor) or isinstance(values[-1],torch.Tensor):
        values = [v if v is not None else torch.tensor(0.0) for v in values ]
    if isinstance(values[0],(int, float, np.integer, np.floating)) or isinstance(values[-1],(int, float, np.integer, np.floating)):
        values = [v if v is not None else 0.0 for v in values]
    return values

def replace_history_Nones(history):
    for k in history.keys():
        if 'loss' in k or 'reg' in k or 'acc' in k or 'match' in k:
            history[k]=replace_Nones_with_0s(history[k])
    return history

def model_updated(initial_params, model):
    for name, param in model.named_parameters():
        if not torch.equal(initial_params[name], param.data):
            return True
    return False

# Store initial model parameters
def get_initial_params(model):
    return {name: param.data.clone() for name, param in model.named_parameters()}

def initialize_gn():
    if config.optimization_kwargs['use_gradnorm']==True:
        gn = GradNorm(config.optimization_kwargs['gradnorm_alpha'])
    else:
        gn=None
    return gn

def update_gn(gn, weighted_losses, shared_paramters, verbose=False):
    use_gradnorm = config.optimization_kwargs['use_gradnorm']
    if use_gradnorm==True:
        gn.update_loss_weights(weighted_losses, shared_paramters)
        if verbose==True:
            print(f"Using gradnorm to update loss term weights")


def flip_categorical_x_values(x_narrowed,grad_narrowed):
    grad_sign = torch.sign(grad_narrowed).int()
    x_narrowed -= grad_sign
    #grad_abs  = torch.abs(grad_narrowed)
    #max_grad_abs_args = torch.argmax(grad_abs, dim=1)
    #for i in range(grad_narrowed.size(0)):
    #    x_narrowed[i, max_grad_abs_args[i]] -= grad_sign[i, max_grad_abs_args[i]]
    x_narrowed = torch.clamp(x_narrowed,0,1)
    return x_narrowed




def perturb_x(data, x, this_grad, wmk_loss, perturb_lr, optimizer, subgraph_dict, all_watermark_indices, all_subgraph_indices):
    # x_copy,edge_index_copy = copy.deepcopy(data).x, copy.deepcopy(data).edge_index
    # x_copy = copy.deepcopy(data.x).requires_grad_(True)
    # log_logits_copy          = node_classifier(x_copy, edge_index_copy)
    # probas_copy = log_logits_copy.clone().exp()
    # loss_watermark_scaled_copy = compute_watermark_loss(subgraph_dict, probas_copy, beta_weights)
    # wmk_loss = loss_watermark_scaled_copy+reg_copy if reg_copy is not None else loss_watermark_scaled_copy
    
    # optimizer.zero_grad()
    # wmk_loss.backward()
    # # # print(f'{j}/20: {wmk_loss:3f}',end='\r')

    # this_grad = x_copy.grad.clone()
    # this_grad = torch.autograd.grad(wmk_loss, x)[0]
    #this_grad_narrowed = this_grad[all_subgraph_indices]
    #x_narrowed = x[all_subgraph_indices]
    for i, sig in enumerate(subgraph_dict.keys()):
        node_indices = subgraph_dict[sig]['nodeIndices']
        wmk_indices  = all_watermark_indices[i]
        x_narrowed = x[node_indices][:,wmk_indices]
        grad_narrowed = this_grad[node_indices][:,wmk_indices]
        #x_narrowed = flip_categorical_x_values(x_narrowed,grad_narrowed).detach().clone()
        data.x[node_indices][:,wmk_indices] -= perturb_lr*grad_narrowed
    # x_copy = x.clone().detach()

    # for s,sig in enumerate(subgraph_dict.keys()):
    #     node_indices = subgraph_dict[sig]['nodeIndices']
    #     watermark_indices = all_watermark_indices[s]
    #     grad_at_this_idx = this_grad[node_indices][:,watermark_indices]
    #     values = data.x[node_indices][:,watermark_indices]-perturb_lr*grad_at_this_idx
    #     data.x[node_indices][:,watermark_indices] = torch.sigmoid(values)

    # these_grad = this_grad[all_subgraph_indices]
    # this_grad[all_subgraph_indices] = (these_grad-these_grad.min())/(these_grad.max()-these_grad.min())
    # x_grad_wmk_loss = torch.zeros_like(data.x)
    # x_grad_wmk_loss[all_subgraph_indices] = this_grad.clone()[all_subgraph_indices]
    # data.x = data.x - perturb_lr*x_grad_wmk_loss
    # watermark = subgraph_dict[list(subgraph_dict.keys())[0]]['watermark']
    # for s in all_subgraph_indices:
    #     data.x[s] = torch.zeros_like(data.x[s])
    #     data.x[s][all_watermark_indices[0]]=watermark[all_watermark_indices[0]]

    optimizer.zero_grad()

    return data

def separate_forward_passes_per_subgraph(node_classifier, subgraph_dict):
    probas_dict = {}
    for sig in subgraph_dict.keys():
        print(sig)
        subgraph = subgraph_dict[sig]['subgraph']
        log_logits_ = node_classifier(subgraph.x, subgraph.edge_index)
        probas_ = log_logits_.clone().exp()
        print('subgraph.x shape:',subgraph.x.shape, 'subgraph.edge_index shape:',subgraph.edge_index.shape, 'probas shape:',probas_.shape)
        probas_dict[sig]= probas_
    return probas_dict



def train(data, dataset_name, 
          debug_multiple_subgraphs=True, 
          save=True,
          print_every=10,
          perturb=False,
          perturb_lr=1e-3):

    torch.autograd.set_detect_anomaly(True)


    if config.optimization_kwargs['clf_only']==True:
        node_classifier, history = train_clf_only(data, dataset_name, save=True, print_every=1)
        return node_classifier, history, None, None, None, None
    

    loss, loss_primary, loss_watermark, loss_primary_weighted, loss_watermark_weighted = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

    optimization_kwargs     = config.optimization_kwargs
    watermark_kwargs        = config.watermark_kwargs

    validate_kwargs()

    num_features = data.x.shape[1]
    node_aug, edge_aug                                       = collect_augmentations()
    loss_dict                                                = setup_loss_dict()
    subgraph_dict, subgraph_signatures, all_subgraph_indices = setup_subgraph_dict(data, dataset_name)
    history, betas_dict, beta_similarities_dict              = setup_history(subgraph_signatures=subgraph_signatures)
    beta_weights                                             = get_beta_weights(subgraph_dict, num_features)
    node_classifier, optimizer                               = initialize_training()
    gn = initialize_gn()

    if optimization_kwargs['use_pcgrad']==True:
        optimizer = PCGrad(optimizer)

    sacrifice_method        = optimization_kwargs['sacrifice_kwargs']['method']
    train_nodes_to_consider_mask = get_train_nodes_to_consider(data, subgraph_dict, all_subgraph_indices, sacrifice_method, data.x.shape[0])
    train_nodes_to_consider = torch.where(train_nodes_to_consider_mask==True)[0]
    epochs                  = optimization_kwargs['epochs']
    all_feature_importances, all_watermark_indices, probas = None, None, None
    x_grad_wmk_loss = torch.zeros_like(data.x)
    x_original_zeros_mask = data.x==0
    wmk_type = watermark_kwargs['watermark_type']
    loss_watermark, beta_similarity = torch.tensor(0.0), torch.tensor(0.0)
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()

        edge_index, x, y    = augment_data(data, node_aug, edge_aug, train_nodes_to_consider, all_subgraph_indices)
        # edge_index_copy = edge_index.clone()
        x = x.requires_grad_(True)
        x_copy = x.clone()

        log_logits          = node_classifier(x, edge_index)
        wmk_optimization_condition_met = (wmk_type=='basic') or (wmk_type=='fancy' and epoch>=watermark_kwargs['fancy_selection_kwargs']['clf_only_epochs']) 
        
        percent_matches = []
        if not wmk_optimization_condition_met:

            loss_primary        = F.nll_loss(log_logits[train_nodes_to_consider_mask], y[train_nodes_to_consider_mask])
            loss_dict, unweighted_total, _ = get_weighted_losses(type_='primary', loss_dict=loss_dict, loss_primary=loss_primary)
            loss = loss_primary = loss_primary_weighted = unweighted_total
            # unweighted_losses = [loss]
            # loss = backward(optimizer, type_='primary', loss_primary=loss_primary, epoch=epoch)
            backward([loss], optimizer, type_='primary', epoch=epoch, verbose=False)
            # loss_save = loss.clone().detach()

            for _ in subgraph_signatures:
                percent_matches.append(0)
            dot = make_dot(loss, params=dict(node_classifier.named_parameters()))
            dot.format = 'png'
            dot.render('computation_graph_clf_stage')

        elif wmk_optimization_condition_met:
            loss_primary        = F.nll_loss(log_logits[train_nodes_to_consider_mask], y[train_nodes_to_consider_mask])
            # loss_watermark_scaled=None
            ''' otherwise if not doing a fancy watermark or if the epoch has been reached, optimize watermark '''

            probas = log_logits.clone().exp()
            print('probas:',probas)
            # probas_copy = probas.clone()
            probas_dict = separate_forward_passes_per_subgraph(node_classifier, subgraph_dict)
            # if config.optimization_kwargs['separate_forward_passes_per_subgraph']==True:
                # probas_dict = separate_forward_passes_per_subgraph(node_classifier, subgraph_dict)
            # else:
                # probas_dict=None

            if wmk_type=='fancy' and epoch==watermark_kwargs['fancy_selection_kwargs']['clf_only_epochs']:
                subgraph_dict, all_watermark_indices, all_feature_importances = apply_fancy_watermark(num_features, subgraph_dict, probas, probas_dict)

            is_last_epoch = epoch==epoch-1

            loss_watermark, beta_similarity, betas_from_every_subgraph, \
                betas_dict, beta_similarities_dict, percent_matches = optimize_and_update(probas, subgraph_dict, betas_dict, 
                                                                                          beta_similarities_dict, is_last_epoch,
                                                                                          debug_multiple_subgraphs, beta_weights, 
                                                                                          percent_matches)
            reg = get_reg_term(betas_from_every_subgraph)

            # loss = backward(optimizer, type_='combined', loss_primary=loss_primary, loss_watermark_scaled=loss_watermark_scaled, reg=reg, epoch=epoch, verbose=False, retain_graph=True, gn=gn)
            # weighted_losses = []
            # losses = [loss_primary, loss_watermark,reg]
            loss_dict, unweighted_total, weighted_total = get_weighted_losses(type_='combined',
                                                            loss_dict=loss_dict,
                                                            loss_primary=loss_primary, 
                                                            loss_watermark=loss_watermark, 
                                                            reg=reg,
                                                            gn=gn)
            loss = weighted_total

            unweighted_losses = [loss_dict[k] for k in ['loss_primary','loss_watermark','reg']]
            weighted_losses = [loss_dict[k] for k in ['loss_primary_weighted','loss_watermark_weighted','reg_weighted']]
            loss_primary_weighted, loss_watermark_weighted = weighted_losses[:2]
            weighted_losses_backward = weighted_losses[:2] if weighted_losses[2] is None else weighted_losses

            x_grad = torch.autograd.grad(loss_watermark_weighted, x, retain_graph=True)[0]

            backward(weighted_losses_backward, optimizer, type_='combined', epoch=epoch, verbose=False, retain_graph=True, gn=gn)
            update_gn(gn, weighted_losses, list(node_classifier.parameters()), verbose=False) # will only run if use_gradnorm=True


            loss_save = loss.clone().detach()

            dot = make_dot(loss, params=dict(node_classifier.named_parameters()))
            dot.format = 'png'
            dot.render('computation_graph_watermarking_stage')


        optimizer.step()

        if wmk_optimization_condition_met and perturb==True:
            data =  perturb_x(data, copy.deepcopy(x), x_grad, loss_watermark, perturb_lr, optimizer, subgraph_dict, all_watermark_indices, all_subgraph_indices)

            # pass
            #this_grad_narrowed = x_grad[all_subgraph_indices]
            #x_narrowed = copy.deepcopy(x)[all_subgraph_indices]
            #x_narrowed = flip_categorical_x_values(x_narrowed,this_grad_narrowed)


            # x_narrowed = x_narrowed.detach().clone()
            # data.x[all_subgraph_indices] = x_narrowed
            # x_narrowed = perturb_x(data, x_copy, x_grad, loss_watermark, perturb_lr, optimizer, subgraph_dict, all_watermark_indices, all_subgraph_indices)


        # if wmk_optimization_condition_met and perturb==True:
            # pass # perturb_x()
            # x_copy,edge_index_copy = copy.deepcopy(data).x, copy.deepcopy(data).edge_index
            # x_copy = x_copy.requires_grad_(True)
            # log_logits_copy          = node_classifier(x_copy, edge_index_copy)
            # probas_copy = log_logits_copy.clone().exp()
            # loss_watermark_scaled_copy = compute_watermark_loss(subgraph_dict, probas_copy, beta_weights)
            # wmk_loss = loss_watermark_scaled_copy+reg_copy if reg_copy is not None else loss_watermark_scaled_copy
            
            # optimizer.zero_grad()
            # wmk_loss.backward()
            # # # print(f'{j}/20: {wmk_loss:3f}',end='\r')

            # this_grad = x_copy.grad.clone()
        #     this_grad_ = torch.zeros_like(this_grad)
        #     for s,sig in enumerate(subgraph_dict.keys()):
        #         node_indices = subgraph_dict[sig]['nodeIndices']
        #         watermark_indices = all_watermark_indices[s]
        #         grad_at_this_idx = this_grad[node_indices][:,watermark_indices]
        #         values = data.x[node_indices][:,watermark_indices]-perturb_lr*grad_at_this_idx
        #         data.x[node_indices][:,watermark_indices] = torch.sigmoid(values)



            # this_grad[all_subgraph_indices] = x_copy.grad.clone()[all_subgraph_indices]
            # this_grad[all_subgraph_indices] = (this_grad[all_subgraph_indices]-this_grad[all_subgraph_indices].min())/(this_grad[all_subgraph_indices].max()-this_grad[all_subgraph_indices].min())
            # x_grad_wmk_loss = torch.zeros_like(data.x)
            # x_grad_wmk_loss[all_subgraph_indices] = this_grad.clone()[all_subgraph_indices]
            # data.x = data.x - perturb_lr*x_grad_wmk_loss
           # watermark = subgraph_dict[list(subgraph_dict.keys())[0]]['watermark']
            #for s in all_subgraph_indices:
                # data.x[s] = torch.zeros_like(data.x[s])
             #   data.x[s][all_watermark_indices[0]]=watermark[all_watermark_indices[0]]

            # optimizer.zero_grad()


        acc_trn = accuracy(log_logits[data.train_mask], y[data.train_mask],verbose=False)
        acc_val = accuracy(log_logits[data.val_mask],   y[data.val_mask],verbose=False)
        # history = update_history_one_epoch(history, loss_save, loss_primary, loss_watermark, acc_trn, acc_val, percent_matches, x)
        # history = update_history_one_epoch(history, losses, weighted_losses, None, acc_trn, acc_val, None, None)
        history = update_history_one_epoch(history, #unweighted_losses, weighted_losses, 
                                           loss,
                                           loss_dict,
                                           acc_trn, acc_val, percent_matches, x)


        if epoch%print_every==0:
            print_epoch_status(epoch, loss_primary_weighted, acc_trn, acc_val, wmk_optimization_condition_met, loss_watermark_weighted, beta_similarity, False)
        gc.collect()

    history['betas']=betas_dict
    history['beta_similarities'] = beta_similarities_dict
    history = replace_history_Nones(history)


    if save==True:
        save_results(dataset_name, node_classifier, history, subgraph_dict, all_feature_importances, all_watermark_indices, probas)
    return node_classifier, history, subgraph_dict, all_feature_importances, all_watermark_indices, probas



# def update_history_one_epoch(history, loss, loss_primary, loss_watermark, acc_trn, acc_val, percent_matches, x_perturbed):
def update_history_one_epoch(history, #losses, weighted_losses, 
                             loss,
                             loss_dict,
                             acc_trn, acc_val, percent_matches, x_perturbed):
    # loss = losses.sum()
   # [loss_primary, loss_watermark, reg] = losses
   # [loss_primary_weighted, loss_watermark_weighted, reg_weighted] = weighted_losses

    try:
        history['losses'].append(loss.clone().detach())
    except:
        history['losses'].append(loss)
    try:
        history['losses_primary'].append(loss_dict['loss_primary'].clone().detach())
    except:
        history['losses_primary'].append(loss_dict['loss_primary'])
    try:
        history['losses_watermark'].append(loss_dict['loss_watermark'].clone().detach())
    except:
        history['losses_watermark'].append(loss_dict['loss_watermark'])
    try:
        history['regs'].append(loss_dict['reg'].clone().detach())
    except:
        history['regs'].append(loss_dict['reg'])
    try:
        history['losses_primary_weighted'].append(loss_dict ['loss_primary_weighted'].clone().detach())
    except:
        history['losses_primary_weighted'].append(loss_dict ['loss_primary_weighted'])
    try:
        history['losses_watermark_weighted'].append(loss_dict ['loss_watermark_weighted'].clone().detach())
    except:
        history['losses_watermark_weighted'].append(loss_dict ['loss_watermark_weighted'])
    try:
        history['regs_weighted'].append(loss_dict ['reg_weighted'].clone().detach())
    except:
        history['regs_weighted'].append(loss_dict ['reg_weighted'])

    try:
        history['xs_perturbed'].append(x_perturbed.clone().detach())
    except:
        history['xs_perturbed'].append(x_perturbed)

    history['train_accs'].append(acc_trn)
    history['val_accs'].append(acc_val)
    history['percent_matches'].append(percent_matches)
    return history


def gather_random_subgraphs_for_testing(data, dataset_name, 
                                        max_degrees_choices=[20,50,100], 
                                        frac_choices = [0.001,0.004,0.005,0.01], 
                                        restart_prob_choices = [0,0.1,0.2], 
                                        nHops_choices=[1,2,3],
                                        overrule_size_info=False, explicit_size_choice=10):
    

    subgraph_kwargs_ =   {'method': 'random',  'fraction': None,  'numSubgraphs': 1,
                          'khop_kwargs':      {'autoChooseSubGs': True,   'nodeIndices':  None,   'numHops': None,   'max_degree': None},
                          'random_kwargs':    {},
                          'rwr_kwargs':       {'restart_prob':None,       'max_steps':1000}}

    if dataset_name=='computers':
        max_degrees_choices = [40]

    use_train_mask=True
    num_ = 50
    avoid_indices = []
    subgraphs = []
    for i in range(num_):
        subgraph_kwargs_['method'] = np.random.choice(['khop','random','random_walk_with_restart'])
        if overrule_size_info==False:
            fraction = subgraph_kwargs_['fraction'] = np.random.choice(frac_choices)
            num_watermarked_nodes = int(fraction*sum(data.train_mask))
        elif overrule_size_info==True:
            fraction = subgraph_kwargs_['fraction'] = 0
            num_watermarked_nodes = explicit_size_choice

        print(f'Forming subgraph {i+1} of {num_}: {subgraph_kwargs_['method']}')
        if subgraph_kwargs_['method']=='khop':
            subgraph_kwargs_['khop_kwargs']['numHops'] = np.random.choice(nHops_choices)
            maxDegree = subgraph_kwargs_['khop_kwargs']['maxDegree'] = np.random.choice(max_degrees_choices)
            random.seed(2575)
            ranked_nodes = torch.tensor(rank_training_nodes_by_degree(dataset_name, data, max_degree=maxDegree)[:50])
            idxs = torch.randperm(len(ranked_nodes))
            ranked_nodes = ranked_nodes[idxs]
            node_indices_to_watermark = ranked_nodes[:num_watermarked_nodes]
            central_node = node_indices_to_watermark[0]
        elif subgraph_kwargs_['method']=='random_walk_with_restart':
            maxDegree = np.random.choice(max_degrees_choices)
            subgraph_kwargs_['rwr_kwargs']['restart_prob'] = np.random.choice(restart_prob_choices)
            ranked_nodes = torch.tensor(rank_training_nodes_by_degree(dataset_name, data, max_degree=maxDegree)[:50])
            idxs = torch.randperm(len(ranked_nodes))
            ranked_nodes = ranked_nodes[idxs]
            node_indices_to_watermark = ranked_nodes[:num_watermarked_nodes]
            print('node_indices_to_watermark:',node_indices_to_watermark)
            central_node = node_indices_to_watermark[0]
        elif subgraph_kwargs_['method']=='random':
            central_node=None

        data_sub, _, subgraph_node_indices = generate_subgraph(data, dataset_name, subgraph_kwargs_, central_node, avoid_indices, use_train_mask, show=False,overrule_size_info=overrule_size_info,explicit_size_choice=explicit_size_choice)
        subgraphs.append((data_sub,subgraph_node_indices))
        try:
            avoid_indices += [node_index.item() for node_index in subgraph_node_indices]
        except:
            avoid_indices += [node_index.item() for node_index in subgraph_node_indices]

    return subgraphs




def get_performance_trends(history, subgraph_dict):
    primary_loss_curve = history['losses_primary'] 
    primary_acc_curve = history['train_accs']
    train_acc = np.round(history['train_accs'][-1],3)
    val_acc = np.round(history['val_accs'][-1],3)
    train_acc = np.round(history['train_accs'][-1],3)
    val_acc = np.round(history['val_accs'][-1],3)

    if config.optimization_kwargs['clf_only']==True:
        return primary_loss_curve, None, None, None, None, None, None, primary_acc_curve, None, train_acc, val_acc
    else:
        watermark_loss_curve = history['losses_watermark']
        final_betas, watermarks, percent_matches = [], [], []
        for subgraph_sig in subgraph_dict.keys():
            this_watermark = subgraph_dict[subgraph_sig]['watermark']
            this_nonzero_indices = torch.where(this_watermark!=0)[0]
            watermark_non_zero   = this_watermark[this_nonzero_indices]
            this_final_beta      = history['betas'][subgraph_sig][-1].clone().detach()
            this_sign_beta       = torch.sign(this_final_beta[this_nonzero_indices])
            this_matches = len(torch.where(this_sign_beta==watermark_non_zero)[0])
            percent_matches.append(100*this_matches/len(watermark_non_zero))
            final_betas.append(this_final_beta)
            watermarks.append(this_watermark)
        percent_match_mean, percent_match_std = np.round(np.mean(percent_matches),1), np.round(np.std(percent_matches),3)
        watermark_acc_curve =  history['percent_matches']
    

    return primary_loss_curve, watermark_loss_curve, final_betas, watermarks, percent_matches, percent_match_mean, percent_match_std, primary_acc_curve, watermark_acc_curve, train_acc, val_acc

               
def update_and_save_df(all_dfs, train_acc, val_acc, percent_matches, final_betas, watermarks, filename='results_df.pkl'):
    merged_dict = merge_kwargs_dicts()
    merged_dict['Train Acc']=train_acc
    merged_dict['Val Acc']=val_acc
    merged_dict['Match Rates']=str(percent_matches)
    df_final_betas = str([b.tolist() for b in final_betas]) if final_betas is not None else str(final_betas)
    df_watermarks = str(watermarks[0].tolist()) if watermarks is not None else str(watermarks)
    merged_dict['Final Betas']=df_final_betas
    merged_dict['Watermark']=df_watermarks

    this_df = pd.DataFrame(merged_dict,index=[0])
    all_dfs = pd.concat([all_dfs,this_df])
    with open(filename,'wb') as f:
        pickle.dump(all_dfs,f)
    return all_dfs


def final_plot_clf_only(history, display_title, primary_loss_curve, train_acc):
    primary_acc_curve = history['train_accs']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axs = plt.subplots(1,2, figsize=(10, 4))
    wrapped_display_title = wrap_title(f'{display_title}.\nTrain Acc={train_acc}', width=80)
    fig.suptitle(wrapped_display_title)
    axs[0].plot(range(len(primary_loss_curve)), primary_loss_curve, label='Primary loss',color=colors[0])
    axs[0].set_xlabel('Epochs');axs[0].set_ylabel('Loss')

    axs[1].plot(range(len(primary_acc_curve)), [100*p for p in primary_acc_curve], label='Clf Acc',color=colors[0])
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend(); plt.show()
    print(); print()

def final_plot(history, display_title, percent_matches, primary_loss_curve, watermark_loss_curve, train_acc):
    if config.optimization_kwargs['clf_only']==True:
        final_plot_clf_only(history, display_title, primary_loss_curve, train_acc)
    else:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        coef_wmk = config.optimization_kwargs['coefWmk']
        percent_match_mean, percent_match_std = np.round(np.mean(percent_matches),1), np.round(np.std(percent_matches),3)
        primary_acc_curve, watermark_acc_curve = history['train_accs'], history['percent_matches']
        means = np.asarray([np.mean(w) for w in watermark_acc_curve])
        stds = np.asarray([np.std(w) for w in watermark_acc_curve])

        fig, axs = plt.subplots(1,2, figsize=(10, 4))
        wrapped_display_title = wrap_title(f'{display_title}.\nTrain Acc={train_acc}%\nMatch rate (mean,std): {percent_match_mean,percent_match_std}', width=80)
        fig.suptitle(wrapped_display_title)
        axs[0].plot(range(len(primary_loss_curve)), primary_loss_curve, label='Primary loss',color=colors[0])
        axs[0].plot(range(len(watermark_loss_curve)), [coef_wmk*l for l in watermark_loss_curve], label='Watermark loss',color=colors[1])
        axs[0].set_xlabel('Epochs');axs[0].set_ylabel('Loss')

        df = pd.DataFrame({"value": means, "std": stds})# np.asarray([np.std(w) for w in watermark_acc_curve])})
        x = range(len(df))
        axs[1].plot(range(len(primary_acc_curve)), [100*p for p in primary_acc_curve], label='Clf Acc',color=colors[0])
        axs[1].plot(x, df["value"], label='Watermark Accuracy',color=colors[1])
        axs[1].fill_between(x, df["value"] - df["std"], df["value"] + df["std"],alpha=0.2,color=colors[1])
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend(); plt.show()
        print(); print()


def get_betas_wmk_and_random(subgraph_dict_wmk, random_subgraphs, probas):
    betas_wmk_raw__ = []
    betas_wmk__ = []
    for subgraph_sig in subgraph_dict_wmk.keys():
        x_sub = subgraph_dict_wmk[subgraph_sig]['subgraph'].x
        y_sub = probas[subgraph_dict_wmk[subgraph_sig]['nodeIndices']]
        beta = solve_regression(x_sub, y_sub).clone().detach()
        betas_wmk_raw__.append(beta)
        betas_wmk__.append(torch.sign(beta))

    betas_random_raw__ = []
    betas_random__ = []
    for (subgraph,node_indices) in random_subgraphs:
        x_sub = subgraph.x
        y_sub = probas[node_indices]
        beta = solve_regression(x_sub, y_sub).clone().detach()
        betas_random_raw__.append(beta)
        betas_random__.append(torch.sign(beta))

    return [betas_wmk_raw__, betas_wmk__], [betas_random_raw__, betas_random__]

def compute_likelihood_of_matches(subgraph_dict_wmk, random_subgraphs, betas_wmk__, betas_random__, sample_size=1000):
    n = len(subgraph_dict_wmk)
    print(f'Computing # matches in groups of {n} beta tensors...')

    ''' random subgraphs '''
    n_tuplets = list(itertools.combinations(range(len(random_subgraphs)), n))
    n_tuplets = random.sample(n_tuplets, sample_size)
    match_counts = []
    for i in range(len(n_tuplets)):
        print(f'{i}/{len(n_tuplets)}',end='\r')
        bs = torch.vstack([betas_random__[j] for j in n_tuplets[i]])
        match_counts.append(count_matches(bs))
    sample_mean = np.mean(match_counts)
    sample_std = np.std(match_counts)

    ''' watermarked-subgraphs '''
    n_tuplets = list(itertools.combinations(range(len(subgraph_dict_wmk)), n))
    bs = torch.vstack(betas_wmk__)
    test_value = count_matches(bs[:n])                 
    t_score = (test_value - sample_mean)/(sample_std)
    p_value = 1 - stats.t.cdf(t_score, df=sample_size-1)

    print(f'\nPopulation Mean, Std: {np.round(sample_mean,3)}, {np.round(sample_std,3)}')
    print(f'# Matches among the {n} watermarked betas: {test_value}\n')
    print(f't_score = {np.round(t_score,3)}, p-value = {np.round(p_value,5)}')

    return test_value, t_score, p_value, match_counts


def show_beta_matches_wmk_and_random(subgraph_dict_wmk, betas_wmk__, betas_random__):
    sig = list(subgraph_dict_wmk.keys())[0]
    watermark = subgraph_dict_wmk[sig]['watermark']
    n = len(torch.where(watermark!=0)[0])
    betas_list = [betas_wmk__, betas_random__[:len(betas_wmk__)]]
    for (type_,betas) in zip(['Watermarked','Random'],betas_list):
        fig,axs = plt.subplots(1, len(betas),figsize=(12,1),sharey=False)
        axs[0].set_ylabel('Match')
        for j,beta in enumerate(betas):
            nonzero_indices = torch.where(watermark!=0)[0]
            watermark_non_zero = watermark[nonzero_indices]
            sign_beta = torch.sign(beta[nonzero_indices])
            matches = len(torch.where(sign_beta==watermark_non_zero)[0])
            percent_match = np.round(100*matches/len(watermark_non_zero),2)

            axs[j].bar(range(len(sign_beta[:n])),           sign_beta[:n].numpy(),          label='sign(Beta)', alpha=0.5)
            axs[j].bar(range(len(watermark_non_zero[:n])),  watermark_non_zero[:n].numpy(), label='Watermark',  alpha=0.5)
            axs[j].set_xlabel('Watermark Indices')
            axs[j].set_ylim(-1,1)
            axs[j].set_yticks([-1,0,1])
            title = f'{type_} subgraph {j+1}:  {percent_match}% match'
            title = wrap_title(title, width=18)
            axs[j].set_title(title,fontsize=8)

        axs[j].legend(fontsize=7,loc='lower right')
        plt.tight_layout()
        plt.show()


def step(optimizer):
    if config.optimization_kwargs['SAM']==True:
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()


def dynamic_grid_search_inner(data, dataset_name, debug_multiple_subgraphs, all_dfs, clf_only, variables, order, current_combination, count_only, count):
    if len(order) == 0:
        process_grid_combination(data, dataset_name, debug_multiple_subgraphs, all_dfs, current_combination, clf_only, count_only)
        return
    first_var = order[0]
    remaining_order = order[1:]
    for val in variables[first_var]:
        new_combination = current_combination.copy()
        new_combination[first_var] = val
        if remaining_order:
            count, [node_classifier, history, subgraph_dict, all_feature_importances, all_watermark_indices, probas] = dynamic_grid_search_inner(data, dataset_name, debug_multiple_subgraphs, all_dfs, clf_only, variables, remaining_order, new_combination, count_only, count)
        else:
            count, [node_classifier, history, subgraph_dict, all_feature_importances, all_watermark_indices, probas] = process_grid_combination(data, dataset_name, debug_multiple_subgraphs, all_dfs, new_combination, clf_only, count_only, count)
    return count, [node_classifier, history, subgraph_dict, all_feature_importances, all_watermark_indices, probas]

def dynamic_grid_search(data, dataset_name, debug_multiple_subgraphs, all_dfs, clf_only, variables, order, count_only=False):
    if len(order) == 0:
        return
    first_var = order[0]
    remaining_order = order[1:]
    count=0
    for val in variables[first_var]:
        if remaining_order:
            count, [node_classifier, history, subgraph_dict, all_feature_importances, all_watermark_indices, probas] = dynamic_grid_search_inner(data, dataset_name, debug_multiple_subgraphs, all_dfs, clf_only, variables, remaining_order, {first_var: val}, count_only, count)
        else:
            count, [node_classifier, history, subgraph_dict, all_feature_importances, all_watermark_indices, probas] = process_grid_combination(data, dataset_name, debug_multiple_subgraphs, all_dfs, {first_var: val}, clf_only, count_only, count)
    
    return count, [node_classifier, history, subgraph_dict, all_feature_importances, all_watermark_indices, probas]

def process_grid_combination(data, dataset_name, debug_multiple_subgraphs, all_dfs, combination, clf_only, count_only, count):
    def set_aug(bool_dict):
        config.augment_kwargs['separate_trainset_from_subgraphs']=bool_dict['separate_trainset_from_subgraphs']
        config.augment_kwargs['ignore_subgraphs']=bool_dict['ignore_subgraphs']
        config.augment_kwargs['nodeDrop']['use']=bool_dict['nodeDrop']
        config.augment_kwargs['nodeMixUp']['use']=bool_dict['nodeMixUp']
        config.augment_kwargs['nodeFeatMask']['use']=bool_dict['nodeFeatMask']
        config.augment_kwargs['edgeDrop']['use']=bool_dict['edgeDrop']
    def set_sacrifice_kwargs(method_and_percentage_pair,training_clf_only):
        continue_=True
        method, percentage = method_and_percentage_pair[0], method_and_percentage_pair[1]
        settings_not_applicable_to_clf_only = (method =='train_node_indices' and percentage ==1) or (method=='subgraph_node_indices')
        if training_clf_only and settings_not_applicable_to_clf_only:
            continue_=False
        else:
            config.optimization_kwargs['sacrifice_kwargs']['method']= method
            config.optimization_kwargs['sacrifice_kwargs']['percentage']= percentage
            continue_=True
        return continue_
    def set_selection_kwargs(evaluate_individually, multi_subg_strategy, selection_strategy):
        config.watermark_kwargs['fancy_selection_kwargs']['evaluate_individually']=evaluate_individually
        config.watermark_kwargs['fancy_selection_kwargs']['multi_subg_strategy']=multi_subg_strategy
        config.watermark_kwargs['fancy_selection_kwargs']['selection_strategy']=selection_strategy
    def set_pcgrad(boolean_value):
        config.optimization_kwargs['use_pcgrad']=boolean_value
    def set_subraph_method(method,regenerate):
        config.subgraph_kwargs['method']=method
        config.subgraph_kwargs['regenerate']=regenerate
    def set_reg_kwargs(reg_kwargs):
        config.optimization_kwargs = update_dict(config.optimization_kwargs,['regularization_type', 'lambda_l2'],reg_kwargs)
    def set_perc(perc):
        if config.watermark_kwargs['watermark_type']=='fancy':
            config.watermark_kwargs['fancy_selection_kwargs']['percent_of_features_to_watermark']=perc
        elif config.watermark_kwargs['watermark_type']=='basic':
            config.watermark_kwargs['p_remove']=1-0.01*perc
    def set_clf_epochs(clf_epochs):
        config.watermark_kwargs['fancy_selection_kwargs']['clf_only_epochs']=clf_epochs
    def set_frac(frac):
        config.subgraph_kwargs['fraction']=frac
    def set_coef_wmk(coef_wmk):
        config.optimization_kwargs['coefWmk']=coef_wmk
    def set_num_subraphs(num_subgraphs):
        config.subgraph_kwargs['numSubgraphs']=num_subgraphs
    def set_balance_beta_weights(boolean_value):
        config.watermark_loss_kwargs['balance_beta_weights']=boolean_value
    def set_ignore_subgraph_neighbors(boolean_value):
        config.optimization_kwargs['ignore_subgraph_neighbors']=boolean_value
    ########
    ########
    continue_=True
    subgraph_type, beta_type, reg_type, sacrifice_percentage, use_pcgrad, balance_beta_weights_choice = '', '', '', 0, False, False
    if clf_only==True:
        watermark_only_keys = ['balance_beta_weights','beta_selection','clf_epochs','frac','ignore_subgraph_neighbors','num_subgraphs','perc','reg','sacrifice_kwargs','subgraph_method','use_PCgrad']
        combination_ = {k:v for (k,v) in combination.items() if k not in watermark_only_keys}
    else:
        combination_ = combination
    for key, value in combination_.items():
        if key=='augment':
            set_aug(value)
        elif key=='coef_wmk':
            set_coef_wmk(value)
        elif key=='sacrifice_kwargs':
            continue_ = set_sacrifice_kwargs(value,clf_only)
            sacrifice_percentage = value[1]
        elif key=='beta_selection':
            set_selection_kwargs(value[0],value[1],value[2])
            beta_type ='Random' if value[2]=='random' else ' '.join(value[1:]).capitalize()
            beta_type += ' beta'
        elif key=='use_PCgrad':
            set_pcgrad(value)
            use_pcgrad=value
        elif key=='subgraph_method':
            set_subraph_method(value[0],value[1])
            subgraph_type = value[0] + ' subgraph'
        elif key=='frac':
            set_frac(value)
        elif key=='reg':
            set_reg_kwargs(value)
            reg_type = 'No' if value[0]==None else ' '.join([value[0],f', lambda={value[1]}'])
            reg_type += ' regularization'
        elif key=='perc':
            set_perc(value)
        elif key=='clf_epochs':
            set_clf_epochs(value)
        elif key=='num_subgraphs':
            set_num_subraphs(value)
        elif key=='balance_beta_weights':
            set_balance_beta_weights(value)
            balance_beta_weights_choice = value
        elif key=='ignore_subgraph_neighbors':
            set_ignore_subgraph_neighbors(value)
        else:
            print(f'key "{key}" not accounted for!')
            break
    
    
    if continue_==True:
        count += 1
        if count_only==True:
            return count, [None, None, None, None, None, None]
        else:
            if clf_only==True:
                try:
                    display_title = f'Classification Only -- Sacrifice {100*sacrifice_percentage}% Training Nodes, PCGrad={use_pcgrad}'
                except:
                    display_title = f'Classification Only -- Sacrifice {sacrifice_percentage} Training Nodes, PCGrad={use_pcgrad}'
            else:
                display_title = f'{beta_type}, {subgraph_type}, {reg_type}, balanced beta weights = {balance_beta_weights_choice}'
            print(display_title)
            gc.collect()                        

            node_classifier, history, subgraph_dict, all_feature_importances, \
                all_watermark_indices, probas = train(data, dataset_name, debug_multiple_subgraphs=debug_multiple_subgraphs,
                                                        save=True, print_every=1)
            

            primary_loss_curve, watermark_loss_curve, final_betas, watermarks, \
                percent_matches, percent_match_mean, percent_match_std, \
                    primary_acc_curve, watermark_acc_curve, train_acc, val_acc = get_performance_trends(history, subgraph_dict)

            all_dfs = update_and_save_df(all_dfs, train_acc, val_acc, percent_matches, final_betas, watermarks, filename='results_df.pkl')
            final_plot(history, display_title, percent_matches, primary_loss_curve, watermark_loss_curve, train_acc)                                                    
            config.subgraph_kwargs['regenerate']=False

            return count, [node_classifier, history, subgraph_dict, all_feature_importances, all_watermark_indices, probas]
        
    return all_dfs


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# class GradNorm:
#     def __init__(self, alpha=0.5):
#         self.alpha = alpha
#         self.weight_primary = torch.tensor(1.0, requires_grad=True)
#         self.weight_watermark = torch.tensor(1.0, requires_grad=True)
#         self.weight_reg = torch.tensor(1.0, requires_grad=True)

#     def get_weighted_losses_gradnorm(self, loss_primary, loss_watermark, reg):
#         assert loss_primary is not None and loss_watermark is not None
#         loss_primary_wtd = self.weight_primary * loss_primary
#         loss_watermark_wtd = self.weight_watermark * loss_watermark
#         weighted_losses = [loss_primary_wtd, loss_watermark_wtd] if reg is None else [loss_primary_wtd, loss_watermark_wtd, self.weight_reg * reg]
#         return weighted_losses

#     def update_loss_weights(self, weighted_losses, shared_parameters):
#         [loss_primary_weighted, loss_watermark_weighted] = weighted_losses[:2]
#         reg_weighted=weighted_losses[2] if len(weighted_losses)==3 else None

#         with torch.no_grad():
#             norms = []
#             g_primary = torch.autograd.grad(loss_primary_weighted, shared_parameters, retain_graph=True)
#             norm_primary = torch.norm(g_primary[0])
#             norms.append(norm_primary)

#             g_watermark = torch.autograd.grad(loss_watermark_weighted, shared_parameters, retain_graph=True)
#             norm_watermark = torch.norm(g_watermark[0])
#             norms.append(norm_watermark)

#             if reg_weighted is not None:
#                 g_reg = torch.autograd.grad(reg_weighted, shared_parameters, retain_graph=True)
#                 norm_reg = torch.norm(g_reg[0])
#                 norms.append(norm_reg)

#             mean_norm = torch.mean(torch.tensor(norms))
            
#             self.weight_primary = self.weight_primary*(norm_primary/mean_norm)**self.alpha
#             self.weight_watermark = self.weight_watermark*(norm_watermark/mean_norm)**self.alpha
#             if reg_weighted is not None:
#                 self.weight_reg = self.weight_reg*(norm_reg/mean_norm)**self.alpha

import torch

class GradNorm:
    def __init__(self, alpha=1, epsilon=1e-8):
        self.alpha = alpha
        self.epsilon = epsilon
        self.weight_primary = torch.tensor(1.0, requires_grad=True)
        self.weight_watermark = torch.tensor(1.0, requires_grad=True)
        self.weight_reg = torch.tensor(1.0, requires_grad=True)

    def get_weighted_losses_gradnorm(self, loss_primary, loss_watermark, reg):
        assert loss_primary is not None and loss_watermark is not None
        loss_primary_wtd = self.weight_primary * loss_primary
        loss_watermark_wtd = self.weight_watermark * loss_watermark
        weighted_losses = [loss_primary_wtd, loss_watermark_wtd]
        if reg is not None:
            weighted_losses.append(self.weight_reg * reg)
        return weighted_losses

    def update_loss_weights(self, weighted_losses, shared_parameters):
        loss_primary_weighted = weighted_losses[0]
        loss_watermark_weighted = weighted_losses[1]
        reg_weighted = weighted_losses[2] if len(weighted_losses) == 3 else None

        norms = []

        # Compute gradients and their norms for the primary loss
        g_primary = torch.autograd.grad(loss_primary_weighted, shared_parameters, retain_graph=True)
        norm_primary = torch.norm(g_primary[0])
        print('norm_primary:',norm_primary)
        norms.append(norm_primary)

        # Compute gradients and their norms for the watermark loss
        g_watermark = torch.autograd.grad(loss_watermark_weighted, shared_parameters, retain_graph=True)
        norm_watermark = torch.norm(g_watermark[0])
        print('norm_watermark:',norm_watermark)
        norms.append(norm_watermark)

        # Compute gradients and their norms for the regularization loss (if applicable)
        if reg_weighted is not None:
            g_reg = torch.autograd.grad(reg_weighted, shared_parameters, retain_graph=True)
            norm_reg = torch.norm(g_reg[0])
            print('norm_reg:',norm_reg)
            norms.append(norm_reg)

        # mean_norm = torch.mean(torch.stack(norms))

        # with torch.no_grad():
        #     self.weight_primary *= (norm_primary / (mean_norm + self.epsilon)) ** self.alpha
        #     self.weight_watermark *= (norm_watermark / (mean_norm + self.epsilon)) ** self.alpha
        #     if reg_weighted is not None:
        #         self.weight_reg *= (norm_reg / (mean_norm + self.epsilon)) ** self.alpha

        #     # Normalize the weights to prevent them from exploding
        #     total_weight = self.weight_primary + self.weight_watermark + (self.weight_reg if reg_weighted is not None else 0)
        #     self.weight_primary /= (total_weight + self.epsilon)
        #     self.weight_watermark /= (total_weight + self.epsilon)
        #     if reg_weighted is not None:
        #         self.weight_reg /= (total_weight + self.epsilon)
