import config
import copy
import gc
import grafog.transforms as T
import matplotlib
import numpy as np
import numpy as np 
import pandas as pd
from   pcgrad.pcgrad import PCGrad # from the following: https://github.com/WeiChengTseng/Pytorch-PCGrad. Renamed to 'pcgrad' and moved to site-packages folder.
import os
import pickle
import random
from   scipy import stats
from   scipy.stats import norm
import torch
import torch.optim as optim
import torch.nn.functional as F
from   torch_geometric.data import Data  
from   torch_geometric.loader import DataLoader
from   torch_geometric.utils import subgraph
from   torch_geometric.transforms import NormalizeFeatures, Compose
from   tqdm.notebook import tqdm

from general_utils import *
from models import *
from regression_utils import *
from subgraph_utils import *
import torch.nn.functional as F
from transform_functions import *
from watermark_utils import *




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
    single_or_multi_graph = dataset_attributes[dataset_name]['single_or_multi_graph']

    if location=='default':
        location = '../data' if dataset_name in ['CORA','CiteSeer','PubMed','computers','photo','PPI','NELL','TWITCH_EN','CS'] else f'../data/{dataset_name}' if dataset_name in ['Flickr','Reddit','Reddit2'] else None

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
    return correct.sum() / len(labels)

def sacrifice_node_indices(train_node_indices,method,subgraph_node_indices=None, verbose=False):
    if method=='subgraph_node_indices':
        assert subgraph_node_indices is not None
        group = subgraph_node_indices
        if verbose==True:
            print(f'Sacrificing subgraph nodes from node classification training')
    elif method=='train_node_indices':
        group=train_node_indices
        if verbose==True:
            print(f'Sacrificing train set nodes from node classification training')
    sacrifice_these = group
    train_node_indices_sans_sacrificed = train_node_indices[~torch.isin(train_node_indices, sacrifice_these)]
    assert torch.all(torch.isin(sacrifice_these,group))
    return train_node_indices_sans_sacrificed 



def extract_results_random_subgraphs(data, dataset_name, sub_size_as_fraction, numSubgraphs, watermark, node_classifier, subgraph_kwargs, regression_kwargs, use_train_mask=False, seed=0):
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
    node_classifier.eval()
    for sig in subgraph_dict.keys():
        data_sub = subgraph_dict[sig]['subgraph']    
        x_sub = data_sub.x
        y_sub =  node_classifier(data.x, data.edge_index).clone().exp()
        omit_indices,_ = get_omit_indices(x_sub, watermark,ignore_zeros_from_subgraphs=False)
        beta = process_beta(solve_regression(x_sub, y_sub, regression_kwargs['lambda']), omit_indices)
        betas_dict[sig].append(beta.clone().detach())
        beta_similarities_dict[sig] = torch.sum(beta*watermark)
    return betas_dict, beta_similarities_dict


def collect_augmentations():
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
    #edge_index, x, y = new_data.edge_index, new_data.x, new_data.y
    return new_data #edge_index, x, y


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
 



def get_watermark_performance_single_subgraph(#probas,
                        probas_dict,
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
        data_sub = subgraph_dict[sig]['subgraph']
    elif similar_subgraph==True:
        data_sub = subgraph_dict[sig]['subgraph_shifted']

    # in case you are testing watermark matches with random/non-watermarked betas (to compare against watermarked performance)
    if this_watermark is None:
        this_watermark = subgraph_dict[sig]['watermark']


    x_sub = data_sub.x
    y_sub = probas_dict[sig]

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

def get_train_nodes_to_consider(data, all_subgraph_indices, sacrifice_method, size_dataset, train_with_test_set=False):
    if train_with_test_set==False:
        train_mask = data.train_mask
    else:
        train_mask = data.test_mask
    train_nodes_not_sacrificed_mask = copy.deepcopy(train_mask)
    if sacrifice_method is not None:
        train_node_indices = torch.arange(size_dataset)[train_mask]
        train_nodes_not_sacrificed = train_node_indices[~torch.isin(train_node_indices, all_subgraph_indices)]
        train_nodes_not_sacrificed_mask = torch.zeros_like(train_mask, dtype=torch.bool)
        train_nodes_not_sacrificed_mask[train_nodes_not_sacrificed] = True
    train_nodes_to_use_mask = train_nodes_not_sacrificed_mask
    return train_nodes_to_use_mask


def get_beta_weights(subgraph_dict, num_features):
    beta_weights = torch.ones(len(subgraph_dict),num_features)        
    return beta_weights


    

def get_watermark_performance(#probas, 
                              probas_dict, subgraph_dict, betas_dict, beta_similarities_dict, is_last_epoch,
                        debug_multiple_subgraphs, beta_weights, penalize_similar_subgraphs=False, shifted_subgraph_loss_coef=0):
    loss_watermark, beta_similarity = torch.tensor(0.0, requires_grad=True), torch.tensor(0.0)
    betas_from_every_subgraph = []
    percent_matches=[]
    sign_betas = []
    for s, sig in enumerate(subgraph_dict.keys()):
        this_loss_watermark, this_percent_match, this_beta_similarity,this_raw_beta = get_watermark_performance_single_subgraph(#probas, 
                                                                                                                                probas_dict, subgraph_dict, sig, 
                                                                                                         ignore_zeros_from_subgraphs=False, 
                                                                                                         debug=debug_multiple_subgraphs,
                                                                                                         beta_weights=beta_weights[s],similar_subgraph=False)
                                                                                                         
        
        if penalize_similar_subgraphs==True:
            similar_subgraph_penalty, _, _, _ = get_watermark_performance_single_subgraph(#probas, 
                                                                                          probas_dict, subgraph_dict, sig, 
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
            self.train_nodes_to_consider_mask = get_train_nodes_to_consider(self.data, None, self.sacrifice_method, self.data.x.shape[0], train_with_test_set=False)
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
            self.train_nodes_to_consider_mask = get_train_nodes_to_consider(self.data, self.all_subgraph_indices, self.sacrifice_method, self.data.x.shape[0], train_with_test_set=False)
            self.train_nodes_to_consider = torch.where(self.train_nodes_to_consider_mask==True)[0]
            self.each_subgraph_feature_importances, self.each_subgraph_watermark_indices, _ = None, None, None
            # self.x_grad = None
        
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
                self.epoch=epoch
                augment_seed=update_seed(augment_seed)
                augmented_data = augment_data(self.data, self.node_aug, self.edge_aug, self.train_nodes_to_consider, self.all_subgraph_indices, seed=augment_seed)

                ### for evaluating on train set -- not just train mask, but (train_mask - subgraph_indices)
                train_node_indices = self.train_nodes_to_consider_mask.nonzero(as_tuple=True)[0]
                self.edge_index_train, _ = subgraph(train_node_indices, augmented_data.edge_index, relabel_nodes=True)
                self.x_train = augmented_data.x[train_node_indices]
                self.y_train = augmented_data.y[train_node_indices]

                ### for evaluating on validation set
                val_node_indices = augmented_data.val_mask.nonzero(as_tuple=True)[0]
                self.edge_index_val, _ = subgraph(val_node_indices, augmented_data.edge_index, relabel_nodes=True)
                self.x_val = augmented_data.x[val_node_indices]
                self.y_val = augmented_data.y[val_node_indices]

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
            save_results(self.dataset_name, self.node_classifier, self.history, self.subgraph_dict, self.each_subgraph_feature_importances, self.each_subgraph_watermark_indices)#, self.probas)
        return self.node_classifier, self.history, self.subgraph_dict, self.each_subgraph_feature_importances, self.each_subgraph_watermark_indices#, self.probas
                    
    def train_clf_only(self, save=True, print_every=1):
        augment_seed=config.seed
        for epoch in tqdm(range(self.epochs)):
            augment_seed=update_seed(augment_seed)
            self.epoch=epoch
            augmented_data = augment_data(self.data, self.node_aug, self.edge_aug, self.train_nodes_to_consider_mask, None,seed=augment_seed)
            self.edge_index, self.x, self.y = augmented_data.edge_index, augmented_data.x, augmented_data.y
            # self.x = self.x.requires_grad_(True) # i think i only neede this for perturbing x
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

    def forward(self, x, edge_index, dropout, mode='train'):
        assert mode in ['train','eval']
        if mode=='train':
            self.node_classifier.train()
            log_logits = self.node_classifier(x, edge_index, dropout)
        elif mode=='eval':
            self.node_classifier.eval()
            log_logits = self.node_classifier(x, edge_index, dropout)
        return log_logits
    
    def separate_forward_passes_per_subgraph_(self, mode='train'):
        return separate_forward_passes_per_subgraph(self.subgraph_dict, self.node_classifier, mode)
    
    ''' old '''
    # def closure_primary(self):
    #     self.optimizer.zero_grad()
    #     log_logits          = self.forward(self.x, self.edge_index, dropout=config.node_classifier_kwargs['dropout'])
    #     self.loss_primary   = F.nll_loss(log_logits[self.train_nodes_to_consider_mask], self.y[self.train_nodes_to_consider_mask])
    #     self.loss_dict, self.unweighted_total, _ = self.get_weighted_losses('primary', self.loss_primary)
    #     self.loss = self.loss_primary = self.loss_primary_weighted = self.unweighted_total
    #     self.compute_accuracy(log_logits, self.y, verbose=False)
    #     self.backward([self.loss], verbose=False, retain_graph=False)
    #     return self.loss

    ''' new '''
    def closure_primary(self):
        self.optimizer.zero_grad()
        # log_logits          = self.forward(self.x, self.edge_index, dropout=config.node_classifier_kwargs['dropout'], mode='train')
        log_logits_train    = self.forward(self.x_train, self.edge_index_train, dropout=config.node_classifier_kwargs['dropout'], mode='train')
        ##
        log_logits_val      = self.forward(self.x_val, self.edge_index_val, dropout=0, mode='eval')
        ##
        # self.loss_primary   = F.nll_loss(log_logits[self.train_nodes_to_consider_mask], self.y[self.train_nodes_to_consider_mask])
        self.loss_primary   = F.nll_loss(log_logits_train, self.y_train)

        self.loss_dict, self.unweighted_total, _ = self.get_weighted_losses('primary', self.loss_primary)
        self.loss = self.loss_primary = self.loss_primary_weighted = self.unweighted_total
        # self.compute_accuracy(log_logits, self.y, verbose=False)
        # self.acc_trn = accuracy(log_logits[self.train_mask], self.y[self.train_mask],verbose=False)
        self.acc_trn = accuracy(log_logits_train, self.y_train,verbose=False)
        self.acc_val = accuracy(log_logits_val, self.y_val,verbose=False)
        self.backward([self.loss], verbose=False, retain_graph=False)
        return self.loss
    
    def closure_watermark(self):
        self.optimizer.zero_grad()
        # log_logits          = self.forward(self.x, self.edge_index, dropout=config.node_classifier_kwargs['dropout'], mode='train')
        log_logits_train          = self.forward(self.x_train, self.edge_index_train, dropout=config.node_classifier_kwargs['dropout'], mode='train')
        ##
        log_logits_val      = self.forward(self.x_val, self.edge_index_val, dropout=0, mode='eval')
        ##
        self.probas_dict = self.separate_forward_passes_per_subgraph_(mode='train')
        self.probas_dict_not_watermarked = separate_forward_passes_per_subgraph(self.subgraph_dict_not_watermarked, self.node_classifier, mode='train')

        if config.watermark_kwargs['watermark_type']=='unimportant' and self.epoch==config.watermark_kwargs['unimportant_selection_kwargs']['clf_only_epochs']:
            self.subgraph_dict = self.apply_watermark_()
        elif config.watermark_kwargs['watermark_type']!='unimportant' and self.epoch==0:
            self.subgraph_dict = self.apply_watermark_()
        # self.loss_primary = F.nll_loss(log_logits[self.train_nodes_to_consider_mask], self.y[self.train_nodes_to_consider_mask])
        self.loss_primary = F.nll_loss(log_logits_train, self.y_train)
        # self.compute_accuracy(log_logits, self.y, verbose=False)
        # self.acc_trn = accuracy(log_logits[self.train_mask], self.y[self.train_mask],verbose=False)
        self.acc_trn = accuracy(log_logits_train, self.y_train,verbose=False)
        self.acc_val = accuracy(log_logits_val, self.y_val,verbose=False)
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
        subgraph_dict, self.each_subgraph_watermark_indices, self.each_subgraph_feature_importances, self.watermarks = apply_watermark(watermark_type, self.num_features, len_watermark, self.subgraph_dict, subgraph_x_concat, #self.probas, 
                                                                                                                                       self.probas_dict, config.watermark_kwargs, seed=config.seed)
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
            self.betas_dict, self.beta_similarities_dict, self.watermark_percent_matches, self.observed_match_count, self.observed_matches_confidence = get_watermark_performance(#self.probas, 
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
                self.observed_match_count_not_watermarked, self.observed_matches_confidence_not_watermarked = get_watermark_performance(#self.probas, 
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
                                        # max_degrees_choices=[20,50,100], 
                                        size_options = [0.001,0.004,0.005,0.01],
                                        # restart_prob_choices = [0,0.1,0.2], 
                                        # nHops_choices=[1,2,3],
                                        size_type='subgraph_size_as_fraction',
                                        seed=0):

    subgraph_kwargs_ =   {'method': 'random',  'subgraph_size_as_fraction': None,  'numSubgraphs': 1,
                          'khop_kwargs':      {'autoChooseSubGs': True,   'nodeIndices':  None,   'numHops': None,   'max_degree': None},
                          'random_kwargs':    {},
                          'rwr_kwargs':       {'restart_prob':None,       'max_steps':1000}}


    use_train_mask=True
    num_ = 100
    avoid_indices = []
    # subgraphs = []
    new_graph_seed = seed

    assert size_type in ['subgraph_size_as_fraction','num_nodes']

    subgraph_dict_by_size ={}

    for i in range(num_):
        new_graph_seed=update_seed(new_graph_seed)
        np.random.seed(new_graph_seed)
        subgraph_kwargs_['method'] = np.random.choice(subgraph_method_choices)
        size = np.random.choice(size_options)
        if size_type=='subgraph_size_as_fraction':
        #if overrule_size_info==False:
            sub_size_as_fraction = subgraph_kwargs_['subgraph_size_as_fraction'] = size
            num_watermarked_nodes = int(sub_size_as_fraction*sum(data.train_mask)*subgraph_kwargs_['numSubgraphs'])
        elif size_type=='num_nodes':
            num_watermarked_nodes = size
            # subgraph_dict_by_size = num_watermarked_nodes

        print(f'Forming subgraph {i+1} of {num_}: {subgraph_kwargs_['method']}')
        # if subgraph_kwargs_['method']=='khop':
            # if dataset_name=='computers':
            #     max_degrees_choices = [40]
        #     subgraph_kwargs_['khop_kwargs']['numHops'] = np.random.choice(nHops_choices)
        #     maxDegree = subgraph_kwargs_['khop_kwargs']['maxDegree'] = np.random.choice(max_degrees_choices)
        #     ranked_nodes = torch.tensor(rank_training_nodes_by_degree(dataset_name, data, max_degree=maxDegree)[:50])
        #     torch.manual_seed(new_graph_seed)
        #     idxs = torch.randperm(len(ranked_nodes))
        #     ranked_nodes = ranked_nodes[idxs]
        #     node_indices_to_watermark = ranked_nodes[:num_watermarked_nodes]
        #     central_node = node_indices_to_watermark[0]
        # elif subgraph_kwargs_['method']=='random_walk_with_restart':
            # if dataset_name=='computers':
            #     max_degrees_choices = [40]
        #     maxDegree = np.random.choice(max_degrees_choices)
        #     subgraph_kwargs_['rwr_kwargs']['restart_prob'] = np.random.choice(restart_prob_choices)
        #     ranked_nodes = torch.tensor(rank_training_nodes_by_degree(dataset_name, data, max_degree=maxDegree)[:50])
        #     torch.manual_seed(new_graph_seed)
        #     idxs = torch.randperm(len(ranked_nodes))
        #     ranked_nodes = ranked_nodes[idxs]
        #     node_indices_to_watermark = ranked_nodes[:num_watermarked_nodes]
        #     print('node_indices_to_watermark:',node_indices_to_watermark)
        #     central_node = node_indices_to_watermark[0]
        # elif subgraph_kwargs_['method']=='random':
        #     central_node=None
        ''' subgraphs constructed from random nodes '''
        central_node=None

        data_sub, _, subgraph_node_indices = generate_subgraph(data, dataset_name, subgraph_kwargs_, central_node, avoid_indices, use_train_mask, show=False,overrule_size_info=True,explicit_size_choice=num_watermarked_nodes, seed=seed)
        # subgraphs.append((data_sub,subgraph_node_indices))
        try:
            avoid_indices += [node_index.item() for node_index in subgraph_node_indices]
        except:
            avoid_indices += [node_index.item() for node_index in subgraph_node_indices]

        # regardless of input size type, dictionary keys will correspond to number of nodes
        subgraph_dict_by_size[num_watermarked_nodes]={'subgraph':data_sub,'nodeIndices':subgraph_node_indices}

    return subgraph_dict_by_size





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


def get_betas_wmk_and_random(trained_node_classifier, subgraph_dict_wmk, random_subgraphs):
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


def separate_forward_passes_per_subgraph(subgraph_dict, node_classifier, mode='train'):
    assert mode in ['train','eval']
    if mode=='train':
        node_classifier.train()
    elif mode=='eval':
        node_classifier.eval()

    if config.optimization_kwargs['separate_forward_passes_per_subgraph']==True:
        probas_dict = {}
        for sig in subgraph_dict.keys():
            subgraph = subgraph_dict[sig]['subgraph']
            log_logits_ = node_classifier(subgraph.x, subgraph.edge_index, dropout=config.node_classifier_kwargs['dropout_subgraphs'])
            probas_dict[sig]= log_logits_.clone().exp()
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


# def find_min_n_uncertain(n_features, mu_natural, sigma_natural, c_LB, c_t, test_effective=False, verbose=False):
#     z_score_dict = {0.9:1.645, 0.95:1.96, 0.99:2.576}
#     z_t = z_score_dict[c_t]
#     t = np.ceil(min(mu_natural +z_t*sigma_natural,n_features))
#     z_LB = z_score_dict[c_LB]
#     LB = max(mu_natural - z_LB*sigma_natural,0)
#     net_matches_needed = max(0,min(t - LB, n_features))
#     p_effective = (n_features-LB)/n_features
#     recommended_watermark_length = int(np.ceil(net_matches_needed/p_effective))

#     def simulate_trial(F, mu_natural, sigma_natural, n_recommended, t):
#         num_initial_matches = int(np.random.normal(mu_natural, sigma_natural))
#         num_initial_matches = np.clip(num_initial_matches, 0, n_features) 
#         tensor = np.zeros(n_features)
#         initial_match_indices = np.random.choice(n_features, num_initial_matches, replace=False)
#         tensor[initial_match_indices] = 1
#         available_indices = np.where(tensor == 0)[0]
#         if len(available_indices)==0:
#             pass
#         else:
#             additional_match_indices = np.random.choice(range(n_features), n_recommended, replace=False)
#             tensor[additional_match_indices] = 1
#         final_num_matches = np.sum(tensor)
#         return final_num_matches >= t

#     if verbose:
#         print('mu_natural:',mu_natural)
#         print('sigma_natural:',sigma_natural)
#         print(f"LB: {LB} (z_LB={z_LB})")
#         print(f"t:  {t} (z_t={z_t})")
#         print(f'Recommended watermark length:', recommended_watermark_length)
#         if test_effective:
#             successes = 0
#             num_trials=1000
#             for _ in range(num_trials):
#                 if simulate_trial(n_features, mu_natural, sigma_natural, recommended_watermark_length, t):
#                     successes += 1
#             success_rate = successes / num_trials
#             print(f"Success rate: {success_rate * 100:.2f}%")
#     return recommended_watermark_length

def find_min_n_uncertain(n_features, mu_natural, sigma_natural, c_LB, c_t, test_effective=False, verbose=False):
    z_t = norm.ppf(c_t)
    t = np.ceil(min(mu_natural +z_t*sigma_natural,n_features))
    z_LB = norm.ppf (c_LB)
    LB = max(mu_natural - z_LB*sigma_natural,0)
    net_matches_needed = max(0,min(t - LB, n_features))
    p_effective = (n_features-LB)/n_features
    recommended_watermark_length = int(np.ceil(net_matches_needed/p_effective))


    def simulate_trial(F, mu_natural, sigma_natural, n_recommended, t):
        num_initial_matches = int(np.random.normal(mu_natural, sigma_natural))
        num_initial_matches = np.clip(num_initial_matches, 0, n_features)
        tensor = np.zeros(n_features)
        initial_match_indices = np.random.choice(n_features, num_initial_matches, replace=False)
        tensor[initial_match_indices] = 1
        available_indices = np.where(tensor == 0)[0]
        if len(available_indices)==0:
            pass
        else:
            additional_match_indices = np.random.choice(range(n_features), n_recommended, replace=False)
            tensor[additional_match_indices] = 1
        final_num_matches = np.sum(tensor)
        return final_num_matches >= t

    if verbose:
        print(f'mu_natural: {mu_natural:.3f}')
        print(f'sigma_natural: {sigma_natural:.3f}')
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




#### fine_tune -- train with test mask
def easy_run_node_classifier(node_classifier, data, subgraph_dict, subgraph_dict_not_watermarked, watermark_loss_kwargs, optimization_kwargs, regression_kwargs, node_classifier_kwargs, train=False, task='prune'):
    ### for evaluating on train set
    if task=='fine_tune':
        assert train==True
        mode='train'
        train_with_test_set=True
    elif task=='prune':
        assert train==False
        mode='eval'
        train_with_test_set=False

    all_subgraph_indices = []
    for sig in subgraph_dict.keys():
        nodeIndices = subgraph_dict[sig]['nodeIndices'].tolist()
        all_subgraph_indices += nodeIndices
    all_subgraph_indices    = torch.tensor(all_subgraph_indices)
    

    sig_0 = list(subgraph_dict.keys())[0]
    watermark = subgraph_dict[sig_0]['watermark']


    ## for evaluating on train set (minus subgraph indices)
    sacrifice_method    = optimization_kwargs['sacrifice_kwargs']['method']
    train_nodes_to_consider_mask = torch.where(get_train_nodes_to_consider(data, all_subgraph_indices, sacrifice_method, data.x.shape[0], train_with_test_set=train_with_test_set)==True)[0]
    train_node_indices = train_nodes_to_consider_mask.nonzero(as_tuple=True)[0]
    if train==True:
        node_aug, edge_aug  = collect_augmentations()
        data        = augment_data(data, node_aug, edge_aug, train_nodes_to_consider_mask, all_subgraph_indices, seed=config.seed)
    edge_index_train, _ = subgraph(train_node_indices, data.edge_index, relabel_nodes=True)
    x_train = data.x[train_node_indices]
    y_train = data.y[train_node_indices]
    if mode=='train':
        node_classifier.train()
        log_logits_train   = node_classifier(x_train, edge_index_train, node_classifier_kwargs['dropout'])
    else:
        node_classifier.eval()
        log_logits_train   = node_classifier(x_train, edge_index_train, 0)
    acc_trn = accuracy(log_logits_train, y_train)

    ### for evaluating on validation set
    val_node_indices = data.val_mask.nonzero(as_tuple=True)[0]
    edge_index_val, _ = subgraph(val_node_indices, data.edge_index, relabel_nodes=True)
    x_val = data.x[val_node_indices]
    y_val = data.y[val_node_indices]
    if mode=='train':
        node_classifier.train()
        log_logits_val = node_classifier(x_val, edge_index_val, node_classifier_kwargs['dropout'])
    else:
        node_classifier.eval()
        log_logits_val = node_classifier(x_val, edge_index_val, 0)
    acc_val = accuracy(log_logits_val, y_val)

    probas_dict  = separate_forward_passes_per_subgraph(subgraph_dict, node_classifier, mode=mode)

    beta_weights = torch.ones(len(subgraph_dict),data.num_features)     
    sign_betas, match_rates = [], []
    for s, sig in enumerate(subgraph_dict.keys()):
        beta_weights_ = beta_weights[s]
        _, this_percent_match, _, this_raw_beta = get_watermark_performance_single_subgraph(probas_dict,subgraph_dict, sig,ignore_zeros_from_subgraphs=False, 
                                                                                                                debug=False,beta_weights=beta_weights_,similar_subgraph=False,
                                                                                                                watermark_loss_kwargs=watermark_loss_kwargs,
                                                                                                                regression_kwargs=regression_kwargs,
                                                                                                                optimization_kwargs=optimization_kwargs,
                                                                                                                this_watermark=None)
        sign_betas.append(torch.sign(this_raw_beta))
        match_rates.append(this_percent_match)

    stacked_sign_betas = torch.vstack(sign_betas)

    total_elements = stacked_sign_betas.numel()
    non_zero_count = torch.count_nonzero(stacked_sign_betas)
    zero_count = total_elements - non_zero_count
    print(f"zeros as % of all WMK sign beta entries: {100*zero_count.item()/total_elements}")

    observed_match_count = count_matches(stacked_sign_betas)
    mu_natural, sigma_natural = get_natural_match_distribution(data.x.shape[1], len(subgraph_dict))
    z_t = norm.ppf(0.99)
    target_number_matches = np.ceil(min(mu_natural +z_t*sigma_natural,data.x.shape[1]))
    observed_matches_confidence = get_confidence_observed_matches(observed_match_count, mu_natural, sigma_natural)


    probas_dict_not_watermarked  = separate_forward_passes_per_subgraph(subgraph_dict_not_watermarked, node_classifier, mode=mode)
    sign_betas_not_watermarked = []
    match_rates_not_watermarked= []
    for s, sig in enumerate(subgraph_dict_not_watermarked.keys()):
        beta_weights_ = beta_weights[s]
        _, this_percent_match_not_watermarked, _, this_raw_beta_not_watermarked = get_watermark_performance_single_subgraph(#probas, 
                                                                                                                            probas_dict_not_watermarked,subgraph_dict_not_watermarked, sig,ignore_zeros_from_subgraphs=False, 
                                                                                                                debug=False,beta_weights=beta_weights_,similar_subgraph=False,
                                                                                                                watermark_loss_kwargs=watermark_loss_kwargs,
                                                                                                                regression_kwargs=regression_kwargs,
                                                                                                                optimization_kwargs=optimization_kwargs,
                                                                                                                this_watermark=watermark)
        sign_betas_not_watermarked.append(torch.sign(this_raw_beta_not_watermarked))
        match_rates_not_watermarked.append(this_percent_match_not_watermarked)

    stacked_sign_betas_not_watermarked = torch.vstack(sign_betas_not_watermarked)

    total_elements = stacked_sign_betas_not_watermarked.numel()
    non_zero_count = torch.count_nonzero(stacked_sign_betas_not_watermarked)
    zero_count = total_elements - non_zero_count
    print(f"zeros as % of all NON-WMK sign beta entries: {100*zero_count.item()/total_elements}")

    observed_match_count_not_watermarked = count_matches(stacked_sign_betas_not_watermarked)
    observed_matches_confidence_not_watermarked = get_confidence_observed_matches(observed_match_count_not_watermarked, mu_natural, sigma_natural)

    return match_rates, acc_trn, acc_val, target_number_matches, observed_match_count, observed_matches_confidence, observed_match_count_not_watermarked, observed_matches_confidence_not_watermarked



def get_node_classifier_and_optimizer_and_subgraph_dict_for_further_processing(model_path, lr):
    subgraph_dict   = pickle.load(open(os.path.join(model_path,'subgraph_dict'),'rb'))
    node_classifier = pickle.load(open(os.path.join(model_path,'node_classifier'),'rb'))
    params_         = list(node_classifier.parameters())
    optimizer       = optim.Adam(params_, lr=lr)
    return node_classifier, optimizer, subgraph_dict
