import copy
import gc
import grafog.transforms as T
# import itertools
# import matplotlib
import numpy as np
import numpy as np 
import pandas as pd
from   pcgrad.pcgrad import PCGrad # from the following: https://github.com/WeiChengTseng/Pytorch-PCGrad. Renamed to 'pcgrad' and moved to site-packages folder.
import os
import pickle
import random
from   scipy import stats
from   scipy.stats import norm
from   sklearn.model_selection import train_test_split
# import textwrap
from   tqdm.notebook import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune



import torch
# import torch.nn as nn
# import torchviz
from torchviz import make_dot


# from pyHSICLasso import HSICLasso


from torch_geometric.data import Data  
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph
from torch_geometric.transforms import NormalizeFeatures, Compose


import config
from general_utils import *
from models import *
from regression_utils import *
from subgraph_utils import *
from transform_functions import *
from watermark_utils import *
import matplotlib


import torch.nn.functional as F

from torch_geometric.loader import GraphSAINTRandomWalkSampler

import sys
def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if 'ZMQInteractiveShell' in shell:
            return True  # Jupyter notebook or qtconsole
        elif 'TerminalInteractiveShell' in shell:
            return False  # Terminal running IPython
        else:
            return False  # Other type (e.g., running from terminal)
    except NameError:
        return False  # Probably standard Python interpreter

def prep_data(dataset_name='CORA', 
              location='default', 
              batch_size='default',
              transform_list = 'default', #= NormalizeFeatures())
              train_val_test_split=[0.6,0.2,0.2],
              seed=0
              ):
    train_ratio, val_ratio, test_ratio = train_val_test_split
    class_ = dataset_attributes[dataset_name]['class']
    print('class:',class_)
    single_or_multi_graph = dataset_attributes[dataset_name]['single_or_multi_graph']

    if location=='default':
        location = '../data' if dataset_name in ['CORA','CiteSeer','PubMed','computers','photo','PPI','NELL','TWITCH_EN'] else f'../data/{dataset_name}' if dataset_name in ['Flickr','Reddit','Reddit2'] else None

    if batch_size=='default':
        batch_size = 'All'

    if transform_list=='default':
        transform_list = []
        if dataset_name in ['CORA','CiteSeer']:
            transform_list =[ChooseLargestMaskForTrain()]
        if dataset_name in ['Flickr']:
            transform_list = [NormalizeFeatures(), ChooseLargestMaskForTrain()]
        if dataset_name == 'CS':
            transform_list =[ChooseLargestMaskForTrain()]
        if dataset_name in ['CS','PubMed']:
            transform_list = [CreateMaskTransform(seed=seed)]
        if dataset_name == 'NELL':
            transform_list = [CreateMaskTransform(seed=seed), SparseToDenseTransform()]
        if dataset_name in ['computers', 'photo']:
            transform_list = [CreateMaskTransform(train_ratio, val_ratio, test_ratio, seed)]
        if dataset_name in ['Reddit','Reddit2']:
            transform_list = [KHopsFractionDatasetTransform(1,3, seed)]
        if dataset_name in ['RelLinkPredDataset']:
            transform_list = [CreateMaskTransform(seed=seed)]
        if dataset_name in ['Twitch_EN']:
            transform_list = [CreateMaskTransform(seed=seed)]
    transform = Compose(transform_list)
    
    print(f'Transorms used when loading {dataset_name}: {[str(t) for t in transform_list]}')

    if single_or_multi_graph=='single':
        if dataset_name in ['Reddit', 'Reddit2','Flickr','NELL']:
            dataset = class_(location, transform)
        elif dataset_name in ['RelLinkPredDataset']:
            dataset = class_(location, 'FB15k-237', transform=transform)
        elif dataset_name in ['Twitch_EN']:
            dataset = class_(location, 'EN', transform=transform)
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

def sacrifice_node_indices(train_node_indices,method,p_sacrifice,subgraph_node_indices=None, verbose=False, seed=0):
    if method=='subgraph_node_indices':
        assert subgraph_node_indices is not None
        group = subgraph_node_indices
        if verbose==True:
            print(f'Sacrificing {100*p_sacrifice}% of subgraph nodes from node classification training')
    elif method=='train_node_indices':
        group=train_node_indices
        if verbose==True:
            print(f'Sacrificing {100*p_sacrifice}% of train set nodes from node classification training')

    num_sacrifice = int(p_sacrifice * len(group))
    torch.manual_seed(seed)
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

def extract_results_random_subgraphs(data, dataset_name, sub_size_as_fraction, numSubgraphs, alpha, watermark, probas, node_classifier, subgraph_kwargs, watermark_loss_kwargs, regression_kwargs, use_train_mask=False, seed=0):
    subgraph_kwargs['numSubgraphs']=numSubgraphs
    subgraph_kwargs['subgraph_size_as_fraction']=sub_size_as_fraction
    if subgraph_kwargs['method']=='khop':
        num_nodes = data.x.shape[0]
        node_indices_to_watermark = random.sample(list(range(num_nodes)),numSubgraphs) 
        subgraph_kwargs['khop_kwargs']['nodeIndices'] = node_indices_to_watermark
    elif subgraph_kwargs['method']=='random':
        pass

    subgraph_dict, _ = collect_subgraphs_within_single_graph_for_watermarking(data, dataset_name, use_train_mask, subgraph_kwargs, seed=seed)
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
        beta = process_beta(solve_regression(x_sub, y_sub, regression_kwargs['lambda']), omit_indices)
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




def get_omit_indices(x_sub, watermark, ignore_zeros_from_subgraphs=True):
    if ignore_zeros_from_subgraphs==True:
        print('ommitting zeros from subgraphs')
        zero_features_within_subgraph = torch.where(torch.sum(x_sub, dim=0) == 0)
    else:
        zero_features_within_subgraph = torch.tensor([[]])
    zero_indices_within_watermark = torch.where(watermark==0)
    omit_indices = torch.tensor(list(set(zero_features_within_subgraph[0].tolist() + zero_indices_within_watermark[0].tolist())))
    not_omit_indices = torch.tensor([i for i in range(x_sub.shape[1]) if i not in omit_indices])
    return omit_indices, not_omit_indices



def process_beta(beta, omit_indices):
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



def augment_data(data, node_aug, edge_aug, train_nodes_to_consider, all_subgraph_indices, sampling_used=False, seed=0):
    p = config.augment_kwargs['p']
    new_data = copy.deepcopy(data)
    new_data.x = new_data.x.detach().clone()

    def apply_augmentations(data_subset, aug_fns):
        for aug_fn in aug_fns:
            if aug_fn is not None:
                data_subset = aug_fn(data_subset)
        return data_subset
    
    def update_data(data, indices, augmented_data):
        data.x[indices] = augmented_data.x
        data.y[indices] = augmented_data.y
        mask = torch.isin(data.edge_index[0], indices) & torch.isin(data.edge_index[1], indices)
        data.edge_index[:, mask] = augmented_data.edge_index

    def select_random_indices(indices, p):
        n = len(indices)
        torch.manual_seed(seed)
        random_order = torch.randperm(n)
        num_keep = int(p*n)
        keep_indices = indices[random_order[:num_keep]]
        return keep_indices

    if config.augment_kwargs['separate_trainset_from_subgraphs'] == True and config.optimization_kwargs['clf_only']==False:
        if sampling_used==True:
            original_node_indices = new_data.node_idx
            original_to_new_node_mapping = {original_idx.item():new_idx for (new_idx,original_idx) in zip(range(len(original_node_indices)), original_node_indices)}
            train_nodes_to_consider = torch.tensor([original_to_new_node_mapping[original_idx] for original_idx in train_nodes_to_consider])
            all_subgraph_indices    = torch.tensor([original_to_new_node_mapping[original_idx] for original_idx in all_subgraph_indices])

        trn_minus_subgraph_nodes = torch.tensor(list(set(train_nodes_to_consider.tolist())-set(all_subgraph_indices)))
        trn_minus_subgraph_nodes_keep = select_random_indices(trn_minus_subgraph_nodes, p)
        train_minus_subgraph_data = get_subgraph_from_node_indices(copy.deepcopy(data), trn_minus_subgraph_nodes_keep)
        train_minus_subgraph_data = apply_augmentations(train_minus_subgraph_data, [node_aug, edge_aug])
        update_data(new_data, trn_minus_subgraph_nodes_keep, train_minus_subgraph_data)
        if config.augment_kwargs['ignore_subgraphs']==False:
            all_subgraph_indices_keep = select_random_indices(all_subgraph_indices,p)
            subgraph_data = get_subgraph_from_node_indices(copy.deepcopy(data), all_subgraph_indices_keep)
            subgraph_data = apply_augmentations(subgraph_data, [node_aug, edge_aug])
            update_data(new_data, all_subgraph_indices_keep, subgraph_data)

    elif config.augment_kwargs['separate_trainset_from_subgraphs'] == False or config.optimization_kwargs['clf_only']==True:
        new_data = apply_augmentations(new_data, [node_aug, edge_aug])
    edge_index, x, y = new_data.edge_index, new_data.x, new_data.y
    return edge_index, x, y


def setup_history(clf_only=False, subgraph_signatures=None):
    if clf_only==False:
        assert subgraph_signatures is not None
    history = {
        'losses': [], 
        'losses_primary': [], 'losses_watermark': [], 'regs':[], 
        'losses_primary_weighted': [], 'losses_watermark_weighted': [], 'regs_weighted':[], 
        'betas': [], 'beta_similarities': [], 'train_accs': [], 'val_accs': [], 'watermark_percent_matches': [], 'match_counts': [], 'match_count_confidence': []
    }
    betas_dict = {sig: [] for sig in subgraph_signatures} if clf_only==False else {}
    beta_similarities_dict = {sig: None for sig in subgraph_signatures} if clf_only==False else {}
    return history, betas_dict, beta_similarities_dict

def setup_subgraph_dict(data, dataset_name, not_watermarked=False, seed=0):
    subgraph_kwargs = config.subgraph_kwargs
    subgraph_dict, all_subgraph_indices = collect_subgraphs_within_single_graph_for_watermarking(data, dataset_name, use_train_mask=True, subgraph_kwargs=subgraph_kwargs, not_watermarked=not_watermarked, seed=seed)
    subgraph_signatures = list(subgraph_dict.keys())
    return subgraph_dict, subgraph_signatures, all_subgraph_indices

def print_epoch_status(epoch, loss_primary, acc_trn, acc_val, condition_met=False, loss_watermark=None, observed_match_count=None, observed_match_confidence=None, observed_match_count_not_watermarked=None, observed_match_confidence_not_watermarked=None, clf_only=True):
    if clf_only==True:
        epoch_printout = f'Epoch: {epoch:3d}, loss_primary = {loss_primary:.3f}, train acc = {acc_trn:.3f}, val acc = {acc_val:.3f}'
    elif clf_only==False:
        if condition_met:
           epoch_printout = f'Epoch: {epoch:3d}, loss_primary = {loss_primary:.3f}, loss_watermark = {loss_watermark:.3f}, train acc = {acc_trn:.3f}, val acc = {acc_val:.3f}, match_count = {observed_match_count}, match_count_confidence = {observed_match_confidence:.3f}, match_count_not_watermarked = {observed_match_count_not_watermarked}, match_count_confidence_not_watermarked = {observed_match_confidence_not_watermarked:.3f}'
        else:
          epoch_printout = f'Epoch: {epoch:3d}, loss_primary = {loss_primary:.3f}, loss_watermark = n/a, B*W = n/a, train acc = {acc_trn:.3f}, val acc = {acc_val:.3f}'
    print(epoch_printout)
 



def get_watermark_performance_single_subgraph(probas, probas_dict,
                       subgraph_dict,sig,
                       ignore_zeros_from_subgraphs=False, 
                       debug=False,
                       beta_weights=None,
                       similar_subgraph=False,
                       watermark_loss_kwargs=None,
                       regression_kwargs=None,
                       optimization_kwargs=None,
                       this_watermark=None):
    
    
    if watermark_loss_kwargs==None:
        watermark_loss_kwargs = config.watermark_loss_kwargs
    if regression_kwargs==None:
        regression_kwargs = config.regression_kwargs
    if optimization_kwargs==None:
        optimization_kwargs = config.optimization_kwargs

    if similar_subgraph==False:
        data_sub, subgraph_node_indices = [subgraph_dict[sig][k] for k in ['subgraph','nodeIndices']]
    elif similar_subgraph==True:
        data_sub, subgraph_node_indices = [subgraph_dict[sig][k] for k in ['subgraph_shifted','nodeIndices_shifted']]


    # in case you are testing watermark matches with random/non-watermarked betas (to compare against watermarked performance)
    if this_watermark is None:
        this_watermark = subgraph_dict[sig]['watermark']


    x_sub = data_sub.x
    if optimization_kwargs['separate_forward_passes_per_subgraph']==True:
        y_sub = probas_dict[sig]
    else:
        y_sub = probas[subgraph_node_indices]
    ''' epoch condtion: epoch==epoch-1'''
    omit_indices,not_omit_indices = get_omit_indices(x_sub, this_watermark,ignore_zeros_from_subgraphs=ignore_zeros_from_subgraphs)
    this_raw_beta = solve_regression(x_sub, y_sub, regression_kwargs['lambda'])
    beta          = process_beta(this_raw_beta, omit_indices)
    B_x_W = (beta*this_watermark).clone()
    B_x_W = B_x_W[not_omit_indices]
    beta_weights = beta_weights[not_omit_indices]
    this_loss_watermark = torch.mean(torch.clamp(watermark_loss_kwargs['epsilon']-B_x_W, min=0)*beta_weights)
    this_beta_similarity = torch.mean(B_x_W)
    if debug:
        print(f"Subgraph: Loss Watermark: {this_loss_watermark.item()}, Beta Similarity: {this_beta_similarity.item()}")
    watermark_non_zero   = this_watermark[not_omit_indices]
    this_sign_beta       = torch.sign(this_raw_beta[not_omit_indices])
    this_matches = len(torch.where(this_sign_beta==watermark_non_zero)[0])
    this_percent_match = 100*this_matches/len(watermark_non_zero)
    return this_loss_watermark, this_percent_match, this_beta_similarity, this_raw_beta


# def get_watermark_performance_single_subgraph_2(probas, probas_dict,
#                        subgraph_dict,sig,
#                        ignore_zeros_from_subgraphs=False, 
#                        debug=False,
#                        beta_weights=None,
#                        similar_subgraph=False,
#                        watermark_loss_kwargs=None,
#                        regression_kwargs=None,
#                        optimization_kwargs=None):
#     if watermark_loss_kwargs==None:
#         watermark_loss_kwargs = config.watermark_loss_kwargs
#     if regression_kwargs==None:
#         regression_kwargs = config.regression_kwargs
#     if optimization_kwargs==None:
#         optimization_kwargs = config.optimization_kwargs

#     if similar_subgraph==False:
#         data_sub, subgraph_node_indices = [subgraph_dict[sig][k] for k in ['watermark','subgraph','nodeIndices']]
#     elif similar_subgraph==True:
#         data_sub, subgraph_node_indices = [subgraph_dict[sig][k] for k in ['watermark','subgraph_shifted','nodeIndices_shifted']]


#     x_sub = data_sub.x
#     if optimization_kwargs['separate_forward_passes_per_subgraph']==True:
#         y_sub = probas_dict[sig]
#     else:
#         y_sub = probas[subgraph_node_indices]
#     ''' epoch condtion: epoch==epoch-1'''
#     omit_indices,not_omit_indices = get_omit_indices(x_sub, this_watermark,ignore_zeros_from_subgraphs=ignore_zeros_from_subgraphs) #indices where watermark is 0
#     this_raw_beta = solve_regression(x_sub, y_sub, regression_kwargs['lambda'])
#     beta          = process_beta(this_raw_beta, omit_indices)
#     B_x_W = (beta*this_watermark).clone()
#     B_x_W = B_x_W[not_omit_indices]
#     beta_weights = beta_weights[not_omit_indices]
#     this_loss_watermark = torch.mean(torch.clamp(watermark_loss_kwargs['epsilon']-B_x_W, min=0)*beta_weights)
#     this_beta_similarity = torch.mean(B_x_W)
#     if debug:
#         print(f"Subgraph: Loss Watermark: {this_loss_watermark.item()}, Beta Similarity: {this_beta_similarity.item()}")
#     watermark_non_zero   = this_watermark[not_omit_indices]
#     this_sign_beta       = torch.sign(this_raw_beta[not_omit_indices])
#     this_matches = len(torch.where(this_sign_beta==watermark_non_zero)[0])
#     this_percent_match = 100*this_matches/len(watermark_non_zero)

#     return this_raw_beta



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
    sig = '_'.join([str(s) for s in all_indices.tolist()])
    subgraph_dict[sig]={}
    subgraph_dict[sig]['subgraph']=sub_data
    subgraph_dict[sig]['nodeIndices']=all_indices
    return subgraph_dict

def get_train_nodes_to_consider(data, all_subgraph_indices, sacrifice_method, size_dataset, train_with_test_set=False, seed=0):
    if train_with_test_set==False:
        train_mask = data.train_mask
    else:
        train_mask = data.test_mask
    train_nodes_not_neighboring_subgraphs_mask = copy.deepcopy(train_mask)
    train_nodes_not_sacrificed_mask = copy.deepcopy(train_mask)
    if sacrifice_method is not None:
        train_node_indices = torch.arange(size_dataset)[train_mask]
        p_sacrifice = config.optimization_kwargs['sacrifice_kwargs']['percentage']
        train_nodes_not_sacrificed = sacrifice_node_indices(train_node_indices,sacrifice_method,p_sacrifice,all_subgraph_indices, seed=seed)
        train_nodes_not_sacrificed_mask = torch.zeros_like(train_mask, dtype=torch.bool)
        train_nodes_not_sacrificed_mask[train_nodes_not_sacrificed] = True
    train_nodes_to_use_mask = train_nodes_not_neighboring_subgraphs_mask & train_nodes_not_sacrificed_mask
    return train_nodes_to_use_mask


def get_beta_weights(subgraph_dict, num_features):
    beta_weights = torch.ones(len(subgraph_dict),num_features)        
    return beta_weights


    

def get_watermark_performance(probas, probas_dict, subgraph_dict, betas_dict, beta_similarities_dict, is_last_epoch,
                        debug_multiple_subgraphs, beta_weights, penalize_similar_subgraphs=False, shifted_subgraph_loss_coef=0):
    loss_watermark, beta_similarity = torch.tensor(0.0, requires_grad=True), torch.tensor(0.0)
    betas_from_every_subgraph = []
    percent_matches=[]
    sign_betas = []
    for s, sig in enumerate(subgraph_dict.keys()):
        this_loss_watermark, this_percent_match, this_beta_similarity,this_raw_beta = get_watermark_performance_single_subgraph(probas, probas_dict, subgraph_dict, sig, 
                                                                                                         ignore_zeros_from_subgraphs=False, 
                                                                                                         debug=debug_multiple_subgraphs,
                                                                                                         beta_weights=beta_weights[s],similar_subgraph=False)
                                                                                                         
        
        if penalize_similar_subgraphs==True:
            similar_subgraph_penalty, _, _, _ = get_watermark_performance_single_subgraph(probas, probas_dict, subgraph_dict, sig, 
                                                                                                         ignore_zeros_from_subgraphs=False, 
                                                                                                         debug=debug_multiple_subgraphs,
                                                                                                         beta_weights=beta_weights[s],similar_subgraph=True)
            this_loss_watermark = this_loss_watermark - shifted_subgraph_loss_coef*similar_subgraph_penalty

        loss_watermark  = loss_watermark + this_loss_watermark
        beta_similarity = beta_similarity + this_beta_similarity
        if is_last_epoch:
            beta_similarities_dict[sig] = this_beta_similarity.clone().detach()
        percent_matches.append(this_percent_match)
        betas_dict[sig].append(this_raw_beta)
        sign_betas.append(torch.sign(this_raw_beta))
        betas_from_every_subgraph.append(this_raw_beta)


    ###
    sign_betas = torch.vstack(sign_betas)
    observed_match_count = count_matches(sign_betas)
    n_features = sign_betas[0].shape[0]
    mu_natural, sigma_natural = get_natural_match_distribution(n_features, len(subgraph_dict))
    observed_matches_confidence = get_confidence_observed_matches(observed_match_count, mu_natural, sigma_natural)
    ###

    betas_from_every_subgraph = torch.vstack(betas_from_every_subgraph)
    loss_watermark  = loss_watermark/len(subgraph_dict)
    beta_similarity = beta_similarity/len(subgraph_dict)
    return loss_watermark, beta_similarity, betas_from_every_subgraph, betas_dict, beta_similarities_dict, percent_matches, observed_match_count, observed_matches_confidence

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

def get_initial_params(model):
    return {name: param.data.clone() for name, param in model.named_parameters()}


class Trainer():
    def __init__(self, data, dataset_name, node_classifier=None):

        self.data = data
        self.dataset_name = dataset_name
        self.num_features = data.x.shape[1]
        self.train_mask = data.train_mask
        self.val_mask = data.val_mask
        self.node_aug, self.edge_aug = collect_augmentations()
        self.sacrifice_method        = config.optimization_kwargs['sacrifice_kwargs']['method']
        self.use_pcgrad=config.optimization_kwargs['use_pcgrad']
        self.lr = config.optimization_kwargs['lr']
        self.epochs = config.optimization_kwargs['epochs']
        self.loss_dict = setup_loss_dict()
        self.coefWmk = config.optimization_kwargs['coefWmk_kwargs']['coefWmk']
        self.node_classifier = Net(**config.node_classifier_kwargs)
        self.instantiate_optimizer()
        validate_kwargs()



        if config.optimization_kwargs['clf_only']==True:
            self.history, _, _                = setup_history(clf_only=True)
            self.train_nodes_to_consider_mask = get_train_nodes_to_consider(self.data, None, self.sacrifice_method, self.data.x.shape[0], train_with_test_set=False, seed=config.seed)
            self.train_nodes_to_consider = torch.where(self.train_nodes_to_consider_mask==True)[0]
            if config.optimization_kwargs['use_pcgrad']==True:
                print('Defaulting to regular Adam optimizer since only one learning task (node classification).')
                config.optimization_kwargs['use_pcgrad']=False
                self.use_pcgrad=False

        else:
            # subgraphs being watermarked
            self.subgraph_dict, self.subgraph_signatures, self.all_subgraph_indices = setup_subgraph_dict(data, dataset_name, not_watermarked=False, seed=config.seed)
            self.history, self.betas_dict, self.beta_similarities_dict              = setup_history(subgraph_signatures=self.subgraph_signatures)
            self.beta_weights                                                       = get_beta_weights(self.subgraph_dict, self.num_features)
            # random subgraphs to test against
            self.subgraph_dict_not_watermarked, self.subgraph_signatures_not_watermarked, self.all_subgraph_indices_not_watermarked = setup_subgraph_dict(data, dataset_name, not_watermarked=True, seed=config.random_seed)
            self.history_not_watermarked, self.betas_dict_not_watermarked, self.beta_similarities_dict_not_watermarked              = setup_history(subgraph_signatures=self.subgraph_signatures_not_watermarked)
            self.beta_weights_not_watermarked                                                                                       = get_beta_weights(self.subgraph_dict_not_watermarked, self.num_features)

            self.loss = torch.tensor(0.0)
            self.loss_primary = torch.tensor(0.0)
            self.loss_watermark = torch.tensor(0.0)
            self.loss_primary_weighted = torch.tensor(0.0)
            self.loss_watermark_weighted = torch.tensor(0.0)
            self.loss_watermark = torch.tensor(0.0)
            self.beta_similarity = torch.tensor(0.0)
            self.train_nodes_to_consider_mask = get_train_nodes_to_consider(self.data, self.all_subgraph_indices, self.sacrifice_method, self.data.x.shape[0], train_with_test_set=False, seed=config.seed)
            self.train_nodes_to_consider = torch.where(self.train_nodes_to_consider_mask==True)[0]
            self.each_subgraph_feature_importances, self.each_subgraph_watermark_indices, self.probas = None, None, None
            self.x_grad = None

        
        return

    def instantiate_optimizer(self):
        lr = self.lr
        node_classifier = Net(**config.node_classifier_kwargs)
        
        if config.optimization_kwargs['use_sam']==True:
            optimizer = SAM(node_classifier.parameters(), optim.SGD, lr=lr, momentum=config.optimization_kwargs['sam_momentum'],rho = config.optimization_kwargs['sam_rho'])
        else:
            optimizer = optim.Adam(node_classifier.parameters(), lr=lr)
        
        if self.use_pcgrad==True:
            optimizer = PCGrad(optimizer)

        self.optimizer = optimizer
        self.node_classifier = node_classifier
        self.lr = lr
        
        node_classifier.train()
        return node_classifier, optimizer

    def build_wmk_coef_sched(self, num_changes=3):
        if config.watermark_kwargs['watermark_type']=='unimportant':
            wmk_start_epoch = config.watermark_kwargs['unimportant_selection_kwargs']['clf_only_epochs']
        else:
            wmk_start_epoch = 0
        if config.optimization_kwargs['coefWmk_kwargs']['schedule_coef_wmk']==True:
            min_coef = config.optimization_kwargs['coefWmk_kwargs']['min_coefWmk_scheduled']
            max_coef = self.coefWmk
            reach_max_by = config.optimization_kwargs['coefWmk_kwargs']['reach_max_coef_wmk_by_epoch']
            coef_intervals = list(np.linspace(min_coef, max_coef, num=num_changes, dtype=int))
            epoch_intervals = list(np.linspace(wmk_start_epoch, reach_max_by, num=num_changes, dtype=int))
            self.wmk_coef_schedule_dict = {}#
            i=0
            for epoch in range(wmk_start_epoch, self.epochs):
                try:
                    next_epoch_change = epoch_intervals[i+1]
                    if epoch>=next_epoch_change:
                        i += 1
                    self.wmk_coef_schedule_dict[epoch]=coef_intervals[i]
                except:
                    self.wmk_coef_schedule_dict[epoch]=coef_intervals[-1]
        else:
            self.wmk_coef_schedule_dict = {}#
            for epoch in range(wmk_start_epoch, self.epochs):
                self.wmk_coef_schedule_dict[epoch]=self.coefWmk


    def compute_accuracy(self, log_logits, y, verbose=False):
        self.acc_trn = accuracy(log_logits[self.train_mask], y[self.train_mask],verbose=verbose)
        self.acc_val = accuracy(log_logits[self.val_mask],   y[self.val_mask],verbose=verbose)

    def train(self, debug_multiple_subgraphs=True, save=True, print_every=10):
        print('config.seed:',config.seed)
        self.debug_multiple_subgraphs = debug_multiple_subgraphs
        if config.optimization_kwargs['clf_only']==True:
            self.node_classifier, self.history =self.train_clf_only(save=save, print_every=print_every)
            return self.node_classifier, self.history, None, None, None, None
        else:
            self.build_wmk_coef_sched(num_changes=3) # builds even if coefWmk is constant
            mu_natural, sigma_natural = get_natural_match_distribution(self.data.x.shape[1], len(self.subgraph_dict))
            print(f'\n\nNatural match distribution across {len(self.subgraph_dict)} tensors of length {self.data.x.shape[1]}: mu={mu_natural:.3f}, sigma={sigma_natural:.3f}\n')
            augment_seed = config.seed
            for epoch in tqdm(range(self.epochs)):
                augment_seed=update_seed(augment_seed)
                self.epoch=epoch
                self.edge_index, self.x, self.y    = augment_data(self.data, self.node_aug, self.edge_aug, self.train_nodes_to_consider, self.all_subgraph_indices, seed=augment_seed)
                wmk_optimization_condition_met_op1 = config.watermark_kwargs['watermark_type']=='basic' or config.watermark_kwargs['watermark_type']=='most_represented'
                wmk_optimization_condition_met_op2 = config.watermark_kwargs['watermark_type']=='unimportant' and self.epoch>=config.watermark_kwargs['unimportant_selection_kwargs']['clf_only_epochs']
                wmk_optimization_condition_met = wmk_optimization_condition_met_op1 or wmk_optimization_condition_met_op2
                if not wmk_optimization_condition_met:
                    self.watermark_percent_matches = [0]*(len(self.subgraph_signatures))
                    closure = self.closure_primary
                elif wmk_optimization_condition_met:
                    self.coefWmk = self.wmk_coef_schedule_dict[epoch]
                    closure = self.closure_watermark
                if config.optimization_kwargs['use_sam']==True:
                    self.optimizer.step(closure)
                else:
                    closure()
                    self.optimizer.step()
                self.history = update_history_one_epoch(self.history, self.loss, self.loss_dict, self.acc_trn, self.acc_val, self.watermark_percent_matches, self.observed_match_count, self.observed_matches_confidence)
                if self.epoch%print_every==0:
                    print_epoch_status(self.epoch, self.loss_primary_weighted, self.acc_trn, self.acc_val, wmk_optimization_condition_met, self.loss_watermark_weighted, self.observed_match_count, self.observed_matches_confidence, self.observed_match_count_not_watermarked, self.observed_matches_confidence_not_watermarked, False)

                gc.collect()


        self.history['betas']=self.betas_dict
        self.history['beta_similarities'] = self.beta_similarities_dict
        self.history = replace_history_Nones(self.history)
        if save==True:
            print('***')
            save_results(self.dataset_name, self.node_classifier, self.history, self.subgraph_dict, self.each_subgraph_feature_importances, self.each_subgraph_watermark_indices, self.probas)
        return self.node_classifier, self.history, self.subgraph_dict, self.each_subgraph_feature_importances, self.each_subgraph_watermark_indices, self.probas
                    
    def train_clf_only(self, save=True, print_every=1):
        augment_seed=config.seed
        for epoch in tqdm(range(self.epochs)):
            augment_seed=update_seed(augment_seed)
            self.epoch=epoch            
            self.edge_index, self.x, self.y    = augment_data(self.data, self.node_aug, self.edge_aug, self.train_nodes_to_consider_mask, None,seed=augment_seed)
            self.x = self.x.requires_grad_(True)
            closure = self.closure_primary
            if config.optimization_kwargs['use_sam']==True:
                self.optimizer.step(closure)
            else:
                closure()
                self.optimizer.step()
            self.history = update_history_one_epoch(self.history, self.loss, self.loss_dict, self.acc_trn, self.acc_val, None, None, None)
            if self.epoch%print_every==0:
                print_epoch_status(self.epoch, self.loss_primary, self.acc_trn, self.acc_val, True, True, None, None, None, None, None, True)
            gc.collect()
        self.history['betas'], self.history['beta_similarities'] = {},{} ## include for consistency with watermarking outputs
        if save==True:
            print('****')
            save_results(self.dataset_name, self.node_classifier, self.history)
        return self.node_classifier, self.history

    def forward(self, x, edge_index, dropout):
        log_logits = self.node_classifier(x, edge_index, dropout)
        return log_logits
    
    def separate_forward_passes_per_subgraph_(self):
        return separate_forward_passes_per_subgraph(self.subgraph_dict, self.node_classifier)
    
    def closure_primary(self):
        self.optimizer.zero_grad()
        log_logits          = self.forward(self.x, self.edge_index, dropout=config.node_classifier_kwargs['dropout'])
        self.loss_primary   = F.nll_loss(log_logits[self.train_nodes_to_consider_mask], self.y[self.train_nodes_to_consider_mask])
        self.loss_dict, self.unweighted_total, _ = self.get_weighted_losses('primary', self.loss_primary)
        self.loss = self.loss_primary = self.loss_primary_weighted = self.unweighted_total
        self.compute_accuracy(log_logits, self.y, verbose=False)
        self.backward([self.loss], verbose=False, retain_graph=False)
        return self.loss
    
    def closure_watermark(self):
        self.optimizer.zero_grad()
        log_logits          = self.forward(self.x, self.edge_index, dropout=config.node_classifier_kwargs['dropout'])
        self.probas = log_logits.clone().exp()
        self.probas_dict = self.separate_forward_passes_per_subgraph_()
        ##
        self.probas_dict_not_watermarked = separate_forward_passes_per_subgraph(self.subgraph_dict_not_watermarked, self.node_classifier)


        if config.watermark_kwargs['watermark_type']=='unimportant' and self.epoch==config.watermark_kwargs['unimportant_selection_kwargs']['clf_only_epochs']:
            self.subgraph_dict = self.apply_watermark_()
        elif config.watermark_kwargs['watermark_type']!='unimportant' and self.epoch==0:
            self.subgraph_dict = self.apply_watermark_()
        self.loss_primary = F.nll_loss(log_logits[self.train_nodes_to_consider_mask], self.y[self.train_nodes_to_consider_mask])
        self.compute_accuracy(log_logits, self.y, verbose=False)
        is_last_epoch = self.epoch==self.epochs-1
        self.get_watermark_performance_(is_last_epoch)
        self.reg = get_reg_term(self.betas_from_every_subgraph)
        self.loss_dict, self.unweighted_total, self.weighted_total = self.get_weighted_losses('combined', self.loss_primary, self.loss_watermark, self.reg)
        self.loss = self.weighted_total
        self.unweighted_losses = [self.loss_dict[k] for k in ['loss_primary','loss_watermark','reg']]
        self.weighted_losses   = [self.loss_dict[k] for k in ['loss_primary_weighted','loss_watermark_weighted','reg_weighted']]
        self.loss_primary_weighted, self.loss_watermark_weighted = self.weighted_losses[:2]
        self.weighted_losses_backward = self.weighted_losses[:2] if self.weighted_losses[2] is None else self.weighted_losses
        self.backward(self.weighted_losses_backward, verbose=False, retain_graph=True)

    def apply_watermark_(self):
        watermark_type = config.watermark_kwargs['watermark_type']
        len_watermark = int(config.watermark_kwargs['percent_of_features_to_watermark']*self.num_features/100)
        subgraph_x_concat = torch.concat([self.subgraph_dict[k]['subgraph'].x for k in self.subgraph_dict.keys()])
        subgraph_dict, self.each_subgraph_watermark_indices, self.each_subgraph_feature_importances, self.watermarks = apply_watermark(watermark_type, self.num_features, len_watermark, self.subgraph_dict, subgraph_x_concat, self.probas, self.probas_dict, config.watermark_kwargs, seed=config.seed)
        for i, subgraph_sig in enumerate(self.subgraph_dict_not_watermarked.keys()):
            self.subgraph_dict_not_watermarked[subgraph_sig]['watermark']=self.watermarks[i]
        return subgraph_dict

    def backward(self, losses, verbose=False, retain_graph=False):
        self.loss = sum(losses)
        if self.use_pcgrad==True:
            self.optimizer.pc_backward(losses)
            if verbose==True:
                print(f"Epoch {self.epoch}: PCGrad backpropagation for multiple losses")
        elif self.use_pcgrad==False:
            self.loss.backward(retain_graph=retain_graph)
            if verbose==True:
                print(f"Epoch {self.epoch}: Regular backpropagation for multiple losses")


    def get_weighted_losses(self, type_='primary', loss_primary=None, loss_watermark=None, reg=None):
        self.loss_dict['loss_primary']=loss_primary
        self.loss_dict['loss_watermark']=loss_watermark
        self.loss_dict['reg']=reg
        assert type_ in ['primary','combined']
        if type_=='primary':
            assert loss_primary is not None
            self.loss_dict['loss_primary_weighted']=loss_primary
        elif type_=='combined':
            assert loss_watermark is not None
            ##
            assert self.coefWmk is not None
            loss_watermark_weighted = loss_watermark*self.coefWmk 
            self.loss_dict['loss_primary_weighted'] = loss_primary 
            self.loss_dict['loss_watermark_weighted'] = loss_watermark_weighted
            self.loss_dict['reg_weighted'] = reg 
        unweighted_total = torch_add_not_None([self.loss_dict[k] for k in ['loss_primary','loss_watermark','reg']])
        weighted_total = torch_add_not_None([self.loss_dict[k] for k in ['loss_primary_weighted','loss_watermark_weighted','reg_weighted']])
        return self.loss_dict, unweighted_total, weighted_total


    def get_watermark_performance_(self, is_last_epoch):
        if config.optimization_kwargs['penalize_similar_subgraphs']==True:
            for sig in self.subgraph_signatures:
                subgraph_node_indices = self.subgraph_dict[sig]['nodeIndices']
                shifted_subgraph, shifted_subgraph_node_indices = self.shift_subgraph_(config.optimization_kwargs['p_swap'], subgraph_node_indices)
                self.subgraph_dict[sig]['subgraph_shifted']=shifted_subgraph
                self.subgraph_dict[sig]['nodeIndices_shifted']=shifted_subgraph_node_indices
        # subgraphs that are being watermarked
        self.loss_watermark, self.beta_similarity, self.betas_from_every_subgraph, \
            self.betas_dict, self.beta_similarities_dict, self.watermark_percent_matches, self.observed_match_count, self.observed_matches_confidence = get_watermark_performance(self.probas, 
                                                                                                                                                                            self.probas_dict, 
                                                                                                                                                                            self.subgraph_dict, 
                                                                                                                                                                            self.betas_dict, 
                                                                                                                                                                            self.beta_similarities_dict, 
                                                                                                                                                                            is_last_epoch,
                                                                                                                                                                            self.debug_multiple_subgraphs, 
                                                                                                                                                                            self.beta_weights,
                                                                                                                                                                            penalize_similar_subgraphs=config.optimization_kwargs['penalize_similar_subgraphs'],
                                                                                                                                                                            shifted_subgraph_loss_coef=config.optimization_kwargs['shifted_subgraph_loss_coef'])
        # compare against random subgraphs
        self.loss_watermark_not_watermarked, self.beta_similarity_not_watermarked, self.betas_from_every_subgraph_not_watermarked, \
            self.betas_dict_not_watermarked, self.beta_similarities_dict_not_watermarked, self.watermark_percent_matches_not_watermarked, \
                self.observed_match_count_not_watermarked, self.observed_matches_confidence_not_watermarked = get_watermark_performance(self.probas, 
                                                                                                                                    self.probas_dict_not_watermarked, 
                                                                                                                                    self.subgraph_dict_not_watermarked, 
                                                                                                                                    self.betas_dict_not_watermarked, 
                                                                                                                                    self.beta_similarities_dict_not_watermarked, 
                                                                                                                                    is_last_epoch,
                                                                                                                                    self.debug_multiple_subgraphs, 
                                                                                                                                    self.beta_weights_not_watermarked,
                                                                                                                                    penalize_similar_subgraphs=config.optimization_kwargs['penalize_similar_subgraphs'],
                                                                                                                                    shifted_subgraph_loss_coef=config.optimization_kwargs['shifted_subgraph_loss_coef'])


def update_history_one_epoch(history, loss, loss_dict, acc_trn, acc_val, watermark_percent_matches, observed_match_count, observed_matches_confidence):
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
    history['train_accs'].append(acc_trn)
    history['val_accs'].append(acc_val)
    history['watermark_percent_matches'].append(watermark_percent_matches)
    history['match_counts'].append(observed_match_count)
    history['match_count_confidence'].append(observed_matches_confidence)

    return history


def gather_random_subgraphs_not_watermarked(data, dataset_name, 
                                        subgraph_method_choices = ['khop','random','random_walk_with_restart'],
                                        max_degrees_choices=[20,50,100], 
                                        frac_choices = [0.001,0.004,0.005,0.01], 
                                        restart_prob_choices = [0,0.1,0.2], 
                                        nHops_choices=[1,2,3],
                                        overrule_size_info=False, 
                                        explicit_size_choice=10, 
                                        seed=0):

    subgraph_kwargs_ =   {'method': 'random',  'subgraph_size_as_fraction': None,  'numSubgraphs': 1,
                          'khop_kwargs':      {'autoChooseSubGs': True,   'nodeIndices':  None,   'numHops': None,   'max_degree': None},
                          'random_kwargs':    {},
                          'rwr_kwargs':       {'restart_prob':None,       'max_steps':1000}}

    if dataset_name=='computers':
        max_degrees_choices = [40]

    use_train_mask=True
    num_ = 50
    avoid_indices = []
    subgraphs = []
    new_graph_seed = seed
    for i in range(num_):
        new_graph_seed=update_seed(new_graph_seed)
        np.random.seed(new_graph_seed)
        subgraph_kwargs_['method'] = np.random.choice(subgraph_method_choices)
        if overrule_size_info==False:
            sub_size_as_fraction = subgraph_kwargs_['subgraph_size_as_fraction'] = np.random.choice(frac_choices)
            num_watermarked_nodes = int(sub_size_as_fraction*sum(data.train_mask)*subgraph_kwargs_['numSubgraphs'])
        elif overrule_size_info==True:
            num_watermarked_nodes = explicit_size_choice

        print(f'Forming subgraph {i+1} of {num_}: {subgraph_kwargs_['method']}')
        if subgraph_kwargs_['method']=='khop':
            subgraph_kwargs_['khop_kwargs']['numHops'] = np.random.choice(nHops_choices)
            maxDegree = subgraph_kwargs_['khop_kwargs']['maxDegree'] = np.random.choice(max_degrees_choices)
            ranked_nodes = torch.tensor(rank_training_nodes_by_degree(dataset_name, data, max_degree=maxDegree)[:50])
            torch.manual_seed(new_graph_seed)
            idxs = torch.randperm(len(ranked_nodes))
            ranked_nodes = ranked_nodes[idxs]
            node_indices_to_watermark = ranked_nodes[:num_watermarked_nodes]
            central_node = node_indices_to_watermark[0]
        elif subgraph_kwargs_['method']=='random_walk_with_restart':
            maxDegree = np.random.choice(max_degrees_choices)
            subgraph_kwargs_['rwr_kwargs']['restart_prob'] = np.random.choice(restart_prob_choices)
            ranked_nodes = torch.tensor(rank_training_nodes_by_degree(dataset_name, data, max_degree=maxDegree)[:50])
            torch.manual_seed(new_graph_seed)
            idxs = torch.randperm(len(ranked_nodes))
            ranked_nodes = ranked_nodes[idxs]
            node_indices_to_watermark = ranked_nodes[:num_watermarked_nodes]
            print('node_indices_to_watermark:',node_indices_to_watermark)
            central_node = node_indices_to_watermark[0]
        elif subgraph_kwargs_['method']=='random':
            central_node=None

        data_sub, _, subgraph_node_indices = generate_subgraph(data, dataset_name, subgraph_kwargs_, central_node, avoid_indices, use_train_mask, show=False,overrule_size_info=overrule_size_info,explicit_size_choice=explicit_size_choice, seed=seed)
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
    match_counts = history['match_counts'][-1]
    match_count_confidence = np.round(history['match_count_confidence'][-1],3)


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
        watermark_acc_curve =  history['watermark_percent_matches']
    
    return primary_loss_curve, watermark_loss_curve, final_betas, watermarks, percent_matches, percent_match_mean, percent_match_std, primary_acc_curve, watermark_acc_curve, train_acc, val_acc, match_counts, match_count_confidence

               
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
    axs[1].legend()

    if is_notebook():
        plt.show()
    else:
        matplotlib.use('Agg')  # Use non-interactive backend only if not in a notebook
    
    print(); print()

def final_plot(history, display_title, percent_matches, primary_loss_curve, watermark_loss_curve, train_acc, plot_name=None, save=True):
    if config.optimization_kwargs['clf_only']==True:
        final_plot_clf_only(history, display_title, primary_loss_curve, train_acc)
    else:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        coef_wmk = config.optimization_kwargs['coefWmk_kwargs']['coefWmk']
        percent_match_mean, percent_match_std = np.round(np.mean(percent_matches),1), np.round(np.std(percent_matches),3)
        primary_acc_curve, watermark_acc_curve = history['train_accs'], history['watermark_percent_matches']
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
        axs[1].legend()
        if save==True:
            assert plot_name is not None
            plt.savefig(f'{plot_name}.png')
        if is_notebook():
            plt.show()
        else:
            matplotlib.use('Agg')  # Use non-interactive backend only if not in a notebook    
        print(); print()


def get_betas_wmk_and_random(trained_node_classifier, subgraph_dict_wmk, random_subgraphs):#, probas):
    betas_wmk_raw__ = []
    betas_wmk__ = []
    for subgraph_sig in subgraph_dict_wmk.keys():
        sub = subgraph_dict_wmk[subgraph_sig]['subgraph']
        x_sub, edge_index_sub = sub.x, sub.edge_index
        trained_node_classifier.eval()
        log_logits = trained_node_classifier(x_sub,edge_index_sub)
        y_sub = log_logits.exp()
        this_raw_beta = solve_regression(x_sub, y_sub, config.regression_kwargs['lambda']).clone().detach()
        this_sign_beta       = torch.sign(this_raw_beta)
        betas_wmk_raw__.append(this_raw_beta)
        betas_wmk__.append(this_sign_beta)


    betas_random_raw__ = []
    betas_random__ = []
    for (subgraph,_) in random_subgraphs:
        x_sub = subgraph.x
        edge_index = subgraph.edge_index
        trained_node_classifier.eval()
        log_logits_sub = trained_node_classifier(x_sub, edge_index)
        y_sub = log_logits_sub.clone().exp()
        beta = solve_regression(x_sub, y_sub, config.regression_kwargs['lambda']).clone().detach()
        betas_random_raw__.append(beta)
        betas_random__.append(torch.sign(beta))

    return [betas_wmk_raw__, betas_wmk__], [betas_random_raw__, betas_random__]


def separate_forward_passes_per_subgraph(subgraph_dict, node_classifier):
    if config.optimization_kwargs['separate_forward_passes_per_subgraph']==True:
        probas_dict = {}
        for sig in subgraph_dict.keys():
            subgraph = subgraph_dict[sig]['subgraph']
            log_logits_ = node_classifier(subgraph.x, subgraph.edge_index, dropout=config.node_classifier_kwargs['dropout_subgraphs'])
            probas_ = log_logits_.clone().exp()
            probas_dict[sig]= probas_
        return probas_dict
    else:
        return None



def get_matches_distribution(betas_random__, n_tuplets, verbose=False):
    if verbose:
        print(f'Computing # matches in groups of {len(n_tuplets[0])} beta tensors...')
    n = len(n_tuplets)
    match_counts = []
    for i in range(n):
        if verbose:
            print(f'{i}/{len(n_tuplets)}',end='\r')
        bs = torch.vstack([betas_random__[j] for j in n_tuplets[i]])
        match_counts.append(count_matches(bs))
    sample_mean_matches = np.mean(match_counts)
    sample_std_matches = np.std(match_counts, ddof=1)
    return match_counts, sample_mean_matches, sample_std_matches

def get_necessary_number_of_matches(sample_mean_matches, sample_std_matches, desired_p_value, n, verbose=False):
    desired_p_value = 0.05
    critical_z = stats.norm.ppf(1 - desired_p_value)
    matches_required = sample_mean_matches + (critical_z * sample_std_matches)
    matches_required = int(np.ceil(matches_required))
    if verbose:
        print(f"To obtain p-value={desired_p_value}, need {matches_required} matches needed across {n} sign(beta) tensors")
    return matches_required

def compute_likelihood_of_observed_matches(betas_wmk__, sample_mean_matches, sample_std_matches, verbose=False):
    ''' watermarked-subgraphs '''
    bs = torch.vstack(betas_wmk__)
    test_value = count_matches(bs)                 
    z_score = (test_value - sample_mean_matches)/sample_std_matches
    p_value = 1 - stats.norm.cdf(z_score)
    if verbose:
        print(f'\nPopulation Mean, Standard Error: {np.round(sample_mean_matches,3)}, {np.round(sample_std_matches,3)}')
        print(f'# Matches among the {len(bs)} watermarked betas: {test_value}\n')
        print(f'z_score = {np.round(z_score,3)}, p-value = {np.round(p_value,5)}')
    return test_value, z_score, p_value

def calculate_recommended_watermark_size(sample_mean_matches, matches_required, desired_p_value, num_features, verbose=False):
    z_desired = stats.norm.ppf(1 - desired_p_value)
    recommended_watermark_size = (matches_required - sample_mean_matches) + z_desired * np.sqrt(matches_required)
    recommended_watermark_size = int(np.ceil(recommended_watermark_size))
    if verbose==True:
        print(f'Recommended watermark size: at least {recommended_watermark_size} (i.e., {np.round(100*matches_required/num_features,3)}% of node feature indices)')
    return recommended_watermark_size


def show_beta_matches_wmk_and_random(subgraph_dict_wmk, betas_wmk__, betas_random__):
    sig = list(subgraph_dict_wmk.keys())[0]
    watermark = subgraph_dict_wmk[sig]['watermark']
    x_sub = subgraph_dict_wmk[sig]['subgraph'].x
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


def dynamic_grid_search_inner(data, dataset_name, debug_multiple_subgraphs, all_dfs, clf_only, variables, order, current_combination, count_only, count, plot_name='plot', save_fig=True):
    if len(order) == 0:
        process_grid_combination(data, dataset_name, debug_multiple_subgraphs, all_dfs, current_combination, clf_only, count_only, plot_name=plot_name, save_fig=save_fig)
        return
    first_var = order[0]
    remaining_order = order[1:]
    for val in variables[first_var]:
        new_combination = current_combination.copy()
        new_combination[first_var] = val
        if remaining_order:
            count, [node_classifier, history, subgraph_dict, all_feature_importances, all_watermark_indices, probas] = dynamic_grid_search_inner(data, dataset_name, debug_multiple_subgraphs, all_dfs, clf_only, variables, remaining_order, new_combination, count_only, count, plot_name=plot_name, save_fig=save_fig)
        else:
            count, [node_classifier, history, subgraph_dict, all_feature_importances, all_watermark_indices, probas] = process_grid_combination(data, dataset_name, debug_multiple_subgraphs, all_dfs, new_combination, clf_only, count_only, count, plot_name=plot_name, save_fig=save_fig)
    return count, [node_classifier, history, subgraph_dict, all_feature_importances, all_watermark_indices, probas]

def dynamic_grid_search(data, dataset_name, debug_multiple_subgraphs, all_dfs, clf_only, variables, order, count_only=False,  plot_name='plot', save_fig=True):
    if len(order) == 0:
        return
    first_var = order[0]
    remaining_order = order[1:]
    count=0
    for val in variables[first_var]:
        if remaining_order:
            count, [node_classifier, history, subgraph_dict, all_feature_importances, all_watermark_indices, probas] = dynamic_grid_search_inner(data, dataset_name, debug_multiple_subgraphs, all_dfs, clf_only, variables, remaining_order, {first_var: val}, count_only, count, plot_name=plot_name, save_fig=save_fig)
        else:
            count, [node_classifier, history, subgraph_dict, all_feature_importances, all_watermark_indices, probas] = process_grid_combination(data, dataset_name, debug_multiple_subgraphs, all_dfs, {first_var: val}, clf_only, count_only, count,  plot_name=plot_name, save_fig=save_fig)
    
    return count, [node_classifier, history, subgraph_dict, all_feature_importances, all_watermark_indices, probas]

def process_grid_combination(data, dataset_name, debug_multiple_subgraphs, all_dfs, combination, clf_only, count_only, count, plot_name='plot', save_fig=True):
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
    def set_selection_kwargs(evaluate_individually, multi_subg_strategy):
        config.watermark_kwargs['unimportant_selection_kwargs']['evaluate_individually']=evaluate_individually
        config.watermark_kwargs['unimportant_selection_kwargs']['multi_subg_strategy']=multi_subg_strategy
    def set_pcgrad(boolean_value):
        config.optimization_kwargs['use_pcgrad']=boolean_value
    def set_subraph_method(method,regenerate):
        config.subgraph_kwargs['method']=method
        config.subgraph_kwargs['regenerate']=regenerate
    def set_reg_kwargs(reg_kwargs):
        config.optimization_kwargs = update_dict(config.optimization_kwargs,['regularization_type', 'lambda_l2'],reg_kwargs)
    def set_perc(perc):
        config.watermark_kwargs['percent_of_features_to_watermark']=perc
    def set_clf_epochs(clf_epochs):
        config.watermark_kwargs['unimportant_selection_kwargs']['clf_only_epochs']=clf_epochs
    def set_sub_size_as_frac(set_sub_size_as_frac):
        config.subgraph_kwargs['subgraph_size_as_fraction']=set_sub_size_as_frac
    def set_coef_wmk(coef_wmk):
        config.optimization_kwargs['coefWmk_kwargs']['coefWmk']=coef_wmk
    def set_num_subraphs(num_subgraphs):
        config.subgraph_kwargs['numSubgraphs']=num_subgraphs
    ########
    ########
    continue_=True
    subgraph_type, beta_type, reg_type, sacrifice_percentage, use_pcgrad, balance_beta_weights_choice = '', '', '', 0, False, False
    if clf_only==True:
        watermark_only_keys = ['beta_selection','clf_epochs','sub_size_as_fraction','num_subgraphs','perc','reg','sacrifice_kwargs','subgraph_method','use_PCgrad']
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
        elif key=='subgraph_size_as_fraction':
            set_sub_size_as_frac(value)
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
                display_title = f'{beta_type}, {subgraph_type}, {reg_type}'
            print(display_title)
            gc.collect()                        

            node_classifier, history, subgraph_dict, all_feature_importances, \
                all_watermark_indices, probas = train(data, dataset_name, debug_multiple_subgraphs=debug_multiple_subgraphs,
                                                        save=True, print_every=1)
            

            primary_loss_curve, watermark_loss_curve, final_betas, watermarks, \
                watermark_percent_matches, percent_match_mean, percent_match_std, \
                    primary_acc_curve, watermark_acc_curve, train_acc, val_acc = get_performance_trends(history, subgraph_dict)

            all_dfs = update_and_save_df(all_dfs, train_acc, val_acc, watermark_percent_matches, final_betas, watermarks, filename='results_df.pkl')
            final_plot(history, display_title, watermark_percent_matches, primary_loss_curve, watermark_loss_curve, train_acc, plot_name=plot_name,save_fig=save_fig)                                                    
            config.subgraph_kwargs['regenerate']=False

            return count, [node_classifier, history, subgraph_dict, all_feature_importances, all_watermark_indices, probas]
        
    return all_dfs


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
    


def get_natural_match_distribution(n_features, c):
    p_natural_match = 2*(0.5**(c))
    mu_natural = p_natural_match*n_features
    var_natural = n_features*p_natural_match*(1-p_natural_match)
    sigma_natural = np.sqrt(var_natural)
    return mu_natural, sigma_natural

def get_confidence_observed_matches(observed_matches, mu_natural, sigma_natural):
    z_score = (observed_matches - mu_natural) / sigma_natural
    one_tail_confidence_level = norm.cdf(z_score)
    return one_tail_confidence_level


def find_min_n_uncertain(n_features, mu_natural, sigma_natural, c_LB, c_t, test_effective=False, verbose=False):
    z_score_dict = {0.9:1.645, 0.95:1.96, 0.99:2.576}
    z_t = z_score_dict[c_t]
    t = np.ceil(min(mu_natural +z_t*sigma_natural,n_features))


    z_LB = z_score_dict[c_LB]
    LB = max(mu_natural - z_LB*sigma_natural,0)

    net_matches_needed = max(0,min(t - LB, n_features))
    p_effective = (n_features-LB)/n_features

    recommended_watermark_length = int(np.ceil(net_matches_needed/p_effective))


    def simulate_trial(F, mu_natural, sigma_natural, n_recommended, t):
        # Step 1: Generate initial matches
        num_initial_matches = int(np.random.normal(mu_natural, sigma_natural))
        num_initial_matches = np.clip(num_initial_matches, 0, n_features)  # Ensure valid range

        # Create tensor with initial matches
        tensor = np.zeros(n_features)
        initial_match_indices = np.random.choice(n_features, num_initial_matches, replace=False)
        tensor[initial_match_indices] = 1

        # Inject additional matches at random locations
        available_indices = np.where(tensor == 0)[0]
        if len(available_indices)==0:
            pass
        else:
            additional_match_indices = np.random.choice(range(n_features), n_recommended, replace=False)
            tensor[additional_match_indices] = 1

        # Step 3: Check if the target is met
        final_num_matches = np.sum(tensor)
        return final_num_matches >= t


    if verbose:
        print('mu_natural:',mu_natural)
        print('sigma_natural:',sigma_natural)
        print(f"LB: {LB} (z_LB={z_LB})")
        print(f"t:  {t} (z_t={z_t})")
        print(f'Recommended watermark length:', recommended_watermark_length)
        if test_effective:
            successes = 0
            num_trials=1000
            for _ in range(num_trials):
                if simulate_trial(n_features, mu_natural, sigma_natural, recommended_watermark_length, t):
                    successes += 1
            success_rate = successes / num_trials
            print(f"Success rate: {success_rate * 100:.2f}%")

    return recommended_watermark_length


def calculate_sparsity(model,verbose=False):
    total_params = 0
    total_zero_params = 0
    for name, module in model.named_modules():
        if isinstance(module, SAGEConv):
            weight_tensor = module.lin_l.weight
            total_params += weight_tensor.numel()
            total_zero_params += (weight_tensor == 0).sum().item()
            if hasattr(module, 'lin_r'):
                weight_tensor = module.lin_r.weight
                total_params += weight_tensor.numel()
                total_zero_params += (weight_tensor == 0).sum().item()
        elif isinstance(module, GCNConv):
            if hasattr(module, 'lin'):  # Check if the internal linear layer exists
                weight_tensor = module.lin.weight
                total_params += weight_tensor.numel()
                total_zero_params += (weight_tensor == 0).sum().item()
        else:
            if hasattr(module, 'weight'):
                weight_tensor = module.weight.data
                total_params += weight_tensor.numel()
                total_zero_params += (weight_tensor == 0).sum().item()
            elif hasattr(module, 'weight_orig'):
                weight_tensor = module.weight_orig.data
                total_params += weight_tensor.numel()
                total_zero_params += (module.weight_mask == 0).sum().item()
    
    sparsity = total_zero_params / total_params
    if verbose==True:
        print(f"Model Sparsity: {sparsity:.2%}")
    return sparsity

def apply_pruning(node_classifier,amount=0.3):
    for name, module in node_classifier.named_modules():
        if isinstance(module, SAGEConv):
            prune.l1_unstructured(module.lin_l, name='weight', amount=amount)
            if hasattr(module, 'lin_r'):
                prune.l1_unstructured(module.lin_r, name='weight', amount=amount)
        elif isinstance(module, GCNConv):
            if hasattr(module, 'lin'):  # Check if the internal linear layer exists
                prune.l1_unstructured(module.lin, name='weight', amount=amount)
            else:
                print(f"{name} does not have an internal 'lin' attribute with 'weight'.")
        elif isinstance(module, (GATConv, GCNConv, GraphConv)):
            prune.l1_unstructured(module, name='weight', amount=amount)
    return node_classifier


# def test_node_classifier(node_classifier, data, subgraph_dict, watermark_loss_kwargs, optimization_kwargs, regression_kwargs, node_classifier_kwargs, train_with_test_set=False, watermark=None, seed=0):
#     node_aug, edge_aug  = collect_augmentations()
#     sacrifice_method    = optimization_kwargs['sacrifice_kwargs']['method']
#     beta_weights        = torch.ones(len(subgraph_dict),data.num_features)     
#     all_subgraph_indices = []
#     for sig in subgraph_dict.keys():
#         nodeIndices = subgraph_dict[sig]['nodeIndices'].tolist()
#         all_subgraph_indices += nodeIndices
#     all_subgraph_indices    = torch.tensor(all_subgraph_indices)
#     train_nodes_to_consider = torch.where(get_train_nodes_to_consider(data, all_subgraph_indices, sacrifice_method, data.x.shape[0], train_with_test_set=train_with_test_set, seed=seed)==True)[0]
#     edge_index, x, y        = augment_data(data, node_aug, edge_aug, train_nodes_to_consider, all_subgraph_indices, seed=seed)
#     log_logits   = node_classifier(x, edge_index, node_classifier_kwargs['dropout'])
#     probas       = log_logits.clone().exp()


#     probas_dict  = separate_forward_passes_per_subgraph(subgraph_dict, node_classifier)
#     acc_trn = accuracy(log_logits[data.train_mask], data.y[data.train_mask])
#     acc_val = accuracy(log_logits[data.val_mask],   data.y[data.val_mask])
#     sign_betas = []
#     match_rates = []
#     for s, sig in enumerate(subgraph_dict.keys()):
#         beta_weights_ = beta_weights[s]
#         _, this_percent_match, _, this_raw_beta = get_watermark_performance_single_subgraph(probas, probas_dict,subgraph_dict, sig,ignore_zeros_from_subgraphs=False, 
#                                                                                                                 debug=False,beta_weights=beta_weights_,similar_subgraph=False,
#                                                                                                                 watermark_loss_kwargs=watermark_loss_kwargs,
#                                                                                                                 regression_kwargs=regression_kwargs,
#                                                                                                                 optimization_kwargs=optimization_kwargs,
#                                                                                                                 this_watermark=watermark)
        

#         sign_betas.append(torch.sign(this_raw_beta))
#         match_rates.append(this_percent_match)
#     ###

#     observed_match_count = count_matches(torch.vstack(sign_betas))
#     mu_natural, sigma_natural = get_natural_match_distribution(data.x.shape[1], len(subgraph_dict))

#     z_score_dict = {0.9:1.645, 0.95:1.96, 0.99:2.576}
#     z_t = z_score_dict[0.99]
#     target_number_matches = np.ceil(min(mu_natural +z_t*sigma_natural,data.x.shape[1]))
#     observed_matches_confidence = get_confidence_observed_matches(observed_match_count, mu_natural, sigma_natural)

#     return match_rates, acc_trn, acc_val, observed_match_count, target_number_matches, observed_matches_confidence
#     ###




# #### prune
# def test_node_classifier_prune(node_classifier, data, subgraph_dict, watermark_loss_kwargs, optimization_kwargs, regression_kwargs, node_classifier_kwargs, seed=0):

#     node_aug, edge_aug  = collect_augmentations()
#     sacrifice_method    = optimization_kwargs['sacrifice_kwargs']['method']
#     beta_weights        = torch.ones(len(subgraph_dict),data.num_features)     
#     all_subgraph_indices = []
#     for sig in subgraph_dict.keys():
#         nodeIndices = subgraph_dict[sig]['nodeIndices'].tolist()
#         all_subgraph_indices += nodeIndices
#     all_subgraph_indices    = torch.tensor(all_subgraph_indices)
#     train_nodes_to_consider = torch.where(get_train_nodes_to_consider(data, all_subgraph_indices, sacrifice_method, data.x.shape[0], train_with_test_set=False, seed=seed)==True)[0]
#     edge_index, x, y        = augment_data(data, node_aug, edge_aug, train_nodes_to_consider, all_subgraph_indices, seed=seed)
#     log_logits   = node_classifier(x, edge_index, node_classifier_kwargs['dropout'])
#     probas       = log_logits.clone().exp()

#     probas_dict  = separate_forward_passes_per_subgraph(subgraph_dict, node_classifier)
#     acc_trn = accuracy(log_logits[data.train_mask], data.y[data.train_mask])
#     acc_val = accuracy(log_logits[data.val_mask],   data.y[data.val_mask])
#     sign_betas = []
#     match_rates = []
#     for s, sig in enumerate(subgraph_dict.keys()):
#         beta_weights_ = beta_weights[s]
#         _, this_percent_match, _, this_raw_beta = get_watermark_performance_single_subgraph(probas, probas_dict,subgraph_dict, sig,ignore_zeros_from_subgraphs=False, 
#                                                                                                                 debug=False,beta_weights=beta_weights_,similar_subgraph=False,
#                                                                                                                 watermark_loss_kwargs=watermark_loss_kwargs,
#                                                                                                                 regression_kwargs=regression_kwargs,
#                                                                                                                 optimization_kwargs=optimization_kwargs,
#                                                                                                                 this_watermark=None)
#         sign_betas.append(torch.sign(this_raw_beta))
#         match_rates.append(this_percent_match)
#     observed_match_count = count_matches(torch.vstack(sign_betas))
#     mu_natural, sigma_natural = get_natural_match_distribution(data.x.shape[1], len(subgraph_dict))
#     z_score_dict = {0.9:1.645, 0.95:1.96, 0.99:2.576}
#     z_t = z_score_dict[0.99]
#     target_number_matches = np.ceil(min(mu_natural +z_t*sigma_natural,data.x.shape[1]))
#     observed_matches_confidence = get_confidence_observed_matches(observed_match_count, mu_natural, sigma_natural)
#     return match_rates, acc_trn, acc_val, target_number_matches, observed_match_count, observed_matches_confidence
# ##


#### fine_tune -- train with test mask
def test_node_classifier(node_classifier, data, subgraph_dict, subgraph_dict_not_watermarked, watermark_loss_kwargs, optimization_kwargs, regression_kwargs, node_classifier_kwargs, task='prune', seed=0):
    assert task=='prune' or task=='fine_tune'
    if task=='prune':
        train_with_test_set=False
        train_mask = data.train_mask
    elif task=='fine_tune':
        train_with_test_set=True
        train_mask = data.test_mask

    sig_0 = list(subgraph_dict.keys())[0]
    watermark = subgraph_dict[sig_0]['watermark']

    node_aug, edge_aug  = collect_augmentations()
    sacrifice_method    = optimization_kwargs['sacrifice_kwargs']['method']
    beta_weights        = torch.ones(len(subgraph_dict),data.num_features)     
    all_subgraph_indices = []
    for sig in subgraph_dict.keys():
        nodeIndices = subgraph_dict[sig]['nodeIndices'].tolist()
        all_subgraph_indices += nodeIndices
    all_subgraph_indices    = torch.tensor(all_subgraph_indices)
    train_nodes_to_consider = torch.where(get_train_nodes_to_consider(data, all_subgraph_indices, sacrifice_method, data.x.shape[0], train_with_test_set=train_with_test_set, seed=seed)==True)[0]
    edge_index, x, y        = augment_data(data, node_aug, edge_aug, train_nodes_to_consider, all_subgraph_indices, seed=seed)
    log_logits   = node_classifier(x, edge_index, node_classifier_kwargs['dropout'])
    probas       = log_logits.clone().exp()

    probas_dict  = separate_forward_passes_per_subgraph(subgraph_dict, node_classifier)
    acc_trn = accuracy(log_logits[train_mask], data.y[train_mask])
    acc_val = accuracy(log_logits[data.val_mask],   data.y[data.val_mask])
    sign_betas = []
    match_rates = []
    for s, sig in enumerate(subgraph_dict.keys()):
        beta_weights_ = beta_weights[s]
        _, this_percent_match, _, this_raw_beta = get_watermark_performance_single_subgraph(probas, probas_dict,subgraph_dict, sig,ignore_zeros_from_subgraphs=False, 
                                                                                                                debug=False,beta_weights=beta_weights_,similar_subgraph=False,
                                                                                                                watermark_loss_kwargs=watermark_loss_kwargs,
                                                                                                                regression_kwargs=regression_kwargs,
                                                                                                                optimization_kwargs=optimization_kwargs,
                                                                                                                this_watermark=None)
        sign_betas.append(torch.sign(this_raw_beta))
        match_rates.append(this_percent_match)
    observed_match_count = count_matches(torch.vstack(sign_betas))
    mu_natural, sigma_natural = get_natural_match_distribution(data.x.shape[1], len(subgraph_dict))
    z_score_dict = {0.9:1.645, 0.95:1.96, 0.99:2.576}
    z_t = z_score_dict[0.99]
    target_number_matches = np.ceil(min(mu_natural +z_t*sigma_natural,data.x.shape[1]))
    observed_matches_confidence = get_confidence_observed_matches(observed_match_count, mu_natural, sigma_natural)


    probas_dict_not_watermarked  = separate_forward_passes_per_subgraph(subgraph_dict_not_watermarked, node_classifier)
    sign_betas_not_watermarked = []
    match_rates_not_watermarked= []
    for s, sig in enumerate(subgraph_dict_not_watermarked.keys()):
        beta_weights_ = beta_weights[s]
        _, this_percent_match_not_watermarked, _, this_raw_beta_not_watermarked = get_watermark_performance_single_subgraph(probas, probas_dict_not_watermarked,subgraph_dict_not_watermarked, sig,ignore_zeros_from_subgraphs=False, 
                                                                                                                debug=False,beta_weights=beta_weights_,similar_subgraph=False,
                                                                                                                watermark_loss_kwargs=watermark_loss_kwargs,
                                                                                                                regression_kwargs=regression_kwargs,
                                                                                                                optimization_kwargs=optimization_kwargs,
                                                                                                                this_watermark=watermark)
        sign_betas_not_watermarked.append(torch.sign(this_raw_beta_not_watermarked))
        match_rates_not_watermarked.append(this_percent_match_not_watermarked)

    observed_match_count_not_watermarked = count_matches(torch.vstack(sign_betas_not_watermarked))
    observed_matches_confidence_not_watermarked = get_confidence_observed_matches(observed_match_count_not_watermarked, mu_natural, sigma_natural)

    return match_rates, acc_trn, acc_val, target_number_matches, observed_match_count, observed_matches_confidence, observed_match_count_not_watermarked, observed_matches_confidence_not_watermarked
