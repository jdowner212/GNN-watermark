import config
import copy
import gc
import grafog.transforms as T
import matplotlib
import numpy as np
import numpy as np 
import pandas as pd
from   pcgrad.pcgrad import PCGrad # from the following: https://github.com/WeiChengTseng/Pytorch-PCGrad. Renamed to 'pcgrad' and moved to site-packages folder.
# instructions:
''' (from conda environment)
git clone https://github.com/WeiChengTseng/Pytorch-PCGrad.git
mv Pytorch-PCGrad pcgrad
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
mv pcgrad "$SITE_PACKAGES"
'''

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

from config import *
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

def prep_data(dataset_name='CS', 
              location='default', 
              batch_size='default',
              transform_list = 'default', #= NormalizeFeatures())
              train_val_test_split=[0.6,0.2,0.2],
              seed=0,
              load=True,
              save=False):
    train_ratio, val_ratio, test_ratio = train_val_test_split

    print('load:',load)

    class_ = dataset_attributes[dataset_name]['class']
    single_or_multi_graph = dataset_attributes[dataset_name]['single_or_multi_graph']


    if location=='default':
        location = '../data' if dataset_name in ['CORA','CiteSeer','PubMed','computers','photo','PPI','NELL','TWITCH_EN','CS'] else f'../data/{dataset_name}' if dataset_name in ['Flickr','Reddit','Reddit2'] else None
    if batch_size=='default':
        batch_size = 'All'
    if transform_list=='default':
        transform_list = []
        if dataset_name in ['CS','PubMed']:
            transform_list = [CreateMaskTransform(train_ratio, val_ratio, test_ratio, seed)]
        if dataset_name in ['computers', 'photo']:
            transform_list = [CreateMaskTransform(train_ratio, val_ratio, test_ratio, seed)]
    transform = Compose(transform_list)

    if single_or_multi_graph=='single':
        saved_location = f'../data/{dataset_name}/load_this_dataset_trn_{train_val_test_split[0]:.2f}_val_{train_val_test_split[1]:2f}_test_{train_val_test_split[2]:2f}.pkl'
        if load==True:
            try:
                dataset = pickle.load(open(saved_location,'rb'))
                print("train_mask:", torch.sum(dataset[0].train_mask).item())
                print("test_mask:",  torch.sum(dataset[0].test_mask).item())
                print("val_mask:",   torch.sum(dataset[0].val_mask).item())
                return dataset
            except:
                print(f'No saved dataset exists at path:\n{saved_location}')
                print("Existing paths:")
                for f in os.listdir(f'../data/{dataset_name}'):
                    print(f'-- {f}')
                print('\nCreating dataset from scratch.')
                load=False
        if load==False:
        # if single_or_multi_graph=='single':
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
            if save==True:
                print('save at:',saved_location)
                with open(saved_location,'wb') as f:
                    pickle.dump(dataset, f)
        return dataset
    
    elif single_or_multi_graph=='multi':
        saved_train_dataset_location = f'../data/{dataset_name}/load_this_train_dataset_split_amount_{train_val_test_split[0]:.2f}.pkl'
        saved_val_dataset_location = f'../data/{dataset_name}/load_this_val_dataset_split_amount_{train_val_test_split[1]:2f}.pkl'
        saved_test_dataset_location = f'../data/{dataset_name}/load_this_test_dataset_split_amount_{train_val_test_split[2]:2f}.pkl'
        saved_train_loader_location = f'../data/{dataset_name}/load_this_train_loader_split_amount_{train_val_test_split[0]:.2f}.pkl'
        saved_val_loader_location = f'../data/{dataset_name}/load_this_val_loader_split_amount_{train_val_test_split[1]:2f}.pkl'
        saved_test_loader_location = f'../data/{dataset_name}/load_this_test_loader_split_amount_{train_val_test_split[2]:2f}.pkl'
        if load==True:
            try:
                train_dataset = pickle.load(open(saved_train_dataset_location,'rb'))
                val_dataset = pickle.load(open(saved_val_dataset_location,'rb'))
                test_dataset = pickle.load(open(saved_test_dataset_location,'rb'))
                train_loader = pickle.load(open(saved_train_loader_location,'rb'))
                val_loader = pickle.load(open(saved_val_loader_location,'rb'))
                test_loader = pickle.load(open(saved_test_loader_location,'rb'))
            except:
                print('No saved data exists.')
                print("Existing paths:")
                for f in os.listdir(f'../data/{dataset_name}'):
                    print(f'-- {f}')
                print('\nCreating data from scratch.')
                load=False

        if load==False:
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

            if save==True:
                with open(saved_train_dataset_location,'wb') as f:
                    pickle.dump(train_dataset, f)
                with open(saved_val_dataset_location,'wb') as f:
                    pickle.dump(val_dataset, f)
                with open(saved_test_dataset_location,'wb') as f:
                    pickle.dump(test_dataset, f)
                with open(saved_train_loader_location,'wb') as f:
                    pickle.dump(train_loader, f)
                with open(saved_val_loader_location,'wb') as f:
                    pickle.dump(val_loader, f)
                with open(saved_test_loader_location,'wb') as f:
                    pickle.dump(test_loader, f)
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
    new_data = data
    #new_data = copy.deepcopy(data)
    #new_data.x = new_data.x.detach().clone()

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
            del subgraph_data
            del all_subgraph_indices_keep
        del trn_minus_subgraph_nodes_keep
        del train_minus_subgraph_data

    elif config.augment_kwargs['separate_trainset_from_subgraphs'] == False or config.optimization_kwargs['clf_only']==True:
        new_data = apply_augmentations(new_data, [node_aug, edge_aug])
    return new_data 




    # history['train_accs'].append(acc_trn)
    # history['val_accs'].append(acc_val)
    # history['watermark_percent_matches'].append(watermark_percent_matches)
    # history['match_counts_with_zeros'].append(match_count_with_zeros)
    # history['match_counts_without_zeros'].append(match_count_without_zeros)
    # history['match_count_confidence_with_zeros'].append(confidence_with_zeros)
    # history['match_count_confidence_without_zeros'].append(confidence_without_zeros)

def setup_history(clf_only=False, subgraph_signatures=None):
    if clf_only==False:
        assert subgraph_signatures is not None
    history = {
        'losses': [], 
        'losses_primary': [], 'losses_watermark': [], 'regs':[], 
        'losses_primary_weighted': [], 'losses_watermark_weighted': [], 'regs_weighted':[], 
        'betas': [], 'beta_similarities': [], 'train_accs': [], 'val_accs': [], 'test_accs': [],'watermark_percent_matches': [], 
        'match_counts_with_zeros': [], 'match_counts_without_zeros':[],
        'match_count_confidence_with_zeros': [], 'match_count_confidence_without_zeros': []
    }
    betas_dict = {sig: [] for sig in subgraph_signatures} if clf_only==False else {}
    beta_similarities_dict = {sig: None for sig in subgraph_signatures} if clf_only==False else {}
    return history, betas_dict, beta_similarities_dict

def setup_subgraph_dict(data, dataset_name, not_wmk=False, seed=0):
    subgraph_kwargs = config.subgraph_kwargs
    subgraph_dict, all_subgraph_indices = collect_subgraphs_within_single_graph_for_watermarking(data, dataset_name, use_train_mask=True, subgraph_kwargs=subgraph_kwargs, not_watermarked=not_wmk, seed=seed)
    subgraph_signatures = list(subgraph_dict.keys())
    return subgraph_dict, subgraph_signatures, all_subgraph_indices



def print_epoch_status(epoch, loss_primary, acc_trn, acc_val, acc_test, condition_met=False, loss_watermark=None, \
                       match_count_with_zeros=None, match_count_without_zeros=None, 
                       confidence_with_zeros=None, confidence_without_zeros=None,
                       match_count_not_wmk_with_zeros=None, match_count_not_wmk_without_zeros=None,\
                       confidence_not_wmk_with_zeros=None, confidence_not_wmk_without_zeros=None, 
                       clf_only=True):
    if clf_only==True:
        epoch_printout = f'Epoch: {epoch:3d}, L_clf = {loss_primary:.3f}, acc (trn/val/test)= {acc_trn:.3f}/{acc_val:.3f}/{acc_test:.3f}'
    elif clf_only==False:
        if condition_met:
           epoch_printout = f'Epoch: {epoch:3d}, L (clf/wmk) = {loss_primary:.3f}/{loss_watermark:.3f}, acc (trn/val/test)= {acc_trn:.3f}/{acc_val:.3f}/{acc_test:.3f}, #_match_WMK w/wout 0s = {match_count_with_zeros}/{match_count_without_zeros}, conf w/wout 0s = {confidence_with_zeros:.3f}/{confidence_without_zeros:.3f}, #_match_NOT_WMK w/wout 0s = {match_count_not_wmk_with_zeros}/{match_count_not_wmk_without_zeros}, conf w/wout 0s= {confidence_not_wmk_with_zeros:.3f}/{confidence_not_wmk_without_zeros:.3f}'
        else:
          epoch_printout = f'Epoch: {epoch:3d}, L_clf = {loss_primary:.3f}, L_wmk = n/a, B*W = n/a, trn acc = {acc_trn:.3f}, val acc = {acc_val:.3f}, test_acc = {acc_test:.3f}'
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

def get_train_nodes_to_consider(data, all_subgraph_indices, sacrifice_method, size_dataset, train_with_test_set=False, clf_only=False):
    if train_with_test_set==False:
        train_mask = data.train_mask
    else:
        train_mask = data.test_mask
    train_nodes_not_sacrificed_mask = copy.deepcopy(train_mask)
    if clf_only==True:
        pass
    else:
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
    match_count_with_zeros = count_matches(sign_betas, ignore_zeros=False)
    match_count_without_zeros = count_matches(sign_betas, ignore_zeros=True)

    n_features = sign_betas[0].shape[0]
    mu_natural, sigma_natural = get_natural_match_distribution(n_features, len(subgraph_dict))
    confidence_with_zeros = get_confidence_matches(match_count_with_zeros, mu_natural, sigma_natural)
    confidence_without_zeros = get_confidence_matches(match_count_without_zeros, mu_natural, sigma_natural)

    ###

    betas_from_every_subgraph = torch.vstack(betas_from_every_subgraph)
    loss_watermark  = loss_watermark/len(subgraph_dict)
    beta_similarity = beta_similarity/len(subgraph_dict)
    return loss_watermark, beta_similarity, betas_from_every_subgraph, betas_dict, beta_similarities_dict, percent_matches, \
        match_count_with_zeros, match_count_without_zeros, \
            confidence_with_zeros,  confidence_without_zeros



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
    def __init__(self, data, dataset_name, target_number_matches=None):

        self.data = data
        print('data:::',data)
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
        self.target_number_matches=target_number_matches
        validate_kwargs()



        if config.optimization_kwargs['clf_only']==True:
            self.history, _, _                = setup_history(clf_only=True)
            self.train_nodes_to_consider_mask = get_train_nodes_to_consider(self.data, None, self.sacrifice_method, self.data.x.shape[0], train_with_test_set=False, clf_only=True)
            self.train_nodes_to_consider = torch.where(self.train_nodes_to_consider_mask==True)[0]
            if config.optimization_kwargs['use_pcgrad']==True:
                print('Defaulting to regular Adam optimizer since only one learning task (node classification).')
                config.optimization_kwargs['use_pcgrad']=False
                self.use_pcgrad=False

        else:
            # subgraphs being watermarked
            self.subgraph_dict, self.subgraph_signatures, self.all_subgraph_indices = setup_subgraph_dict(data, dataset_name, not_wmk=False, seed=config.seed)
            self.history, self.betas_dict, self.beta_similarities_dict              = setup_history(subgraph_signatures=self.subgraph_signatures)
            self.beta_weights                                                       = get_beta_weights(self.subgraph_dict, self.num_features)
            # random subgraphs to test against
            self.subgraph_dict_not_wmk, self.subgraph_signatures_not_wmk, self.all_subgraph_indices_not_wmk = setup_subgraph_dict(data, dataset_name, not_wmk=True, seed=config.random_seed)
            self.history_not_wmk, self.betas_dict_not_wmk, self.beta_similarities_dict_not_wmk              = setup_history(subgraph_signatures=self.subgraph_signatures_not_wmk)
            self.beta_weights_not_wmk                                                                                       = get_beta_weights(self.subgraph_dict_not_wmk, self.num_features)

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
        self.best_train_acc, self.best_val_acc, self.best_match_count = 0, 0, 0
        self.debug_multiple_subgraphs = debug_multiple_subgraphs
        if config.optimization_kwargs['clf_only']==True:
            return self.train_clf_only(save=save, print_every=print_every)
        else:
            self.build_wmk_coef_sched(num_changes=3) # builds even if coefWmk is constant
            mu_natural, sigma_natural = get_natural_match_distribution(self.data.x.shape[1], len(self.subgraph_dict))
            print(f'\n\nNatural match distribution across {len(self.subgraph_dict)} tensors of length {self.data.x.shape[1]}: mu={mu_natural:.3f}, sigma={sigma_natural:.3f}\n')
            print('Target # matches:',self.target_number_matches)
            augment_seed = config.seed



            self.train_node_indices = self.train_nodes_to_consider_mask.nonzero(as_tuple=True)[0]

            ### for evaluating on train set -- not just train mask, but (train_mask - subgraph_indices)
            unaugmented_x = self.data.x.clone()
            unaugmented_edge_index = self.data.edge_index.clone()
            unaugmented_y= self.data.y.clone()
            self.edge_index_train_unaugmented, _ = subgraph(self.train_node_indices, self.data.edge_index, relabel_nodes=True)
            self.edge_index_train_unaugmented = self.edge_index_train_unaugmented.clone()
            self.x_train_unaugmented = self.data.x[self.train_node_indices].clone()
            self.y_train_unaugmented = self.data.y[self.train_node_indices].clone()
            ### for evaluating on test set
            test_node_indices = self.data.test_mask.nonzero(as_tuple=True)[0]
            self.edge_index_test, _ = subgraph(test_node_indices, self.data.edge_index, relabel_nodes=True)
            self.edge_index_test = self.edge_index_test.clone()
            self.x_test = self.data.x[test_node_indices].clone()
            self.y_test = self.data.y[test_node_indices].clone()
            
            ### for evaluating on validation set
            val_node_indices = self.data.val_mask.nonzero(as_tuple=True)[0]
            self.edge_index_val, _ = subgraph(val_node_indices, self.data.edge_index, relabel_nodes=True)
            self.edge_index_val = self.edge_index_val.clone()
            self.x_val = self.data.x[val_node_indices].clone()
            self.y_val = self.data.y[val_node_indices].clone()

            for epoch in tqdm(range(self.epochs)):
                self.epoch=epoch
                augment_seed=update_seed(augment_seed)

                augmented_data = augment_data(self.data, self.node_aug, self.edge_aug, self.train_nodes_to_consider, self.all_subgraph_indices, seed=augment_seed)
                self.edge_index_train, _ = subgraph(self.train_node_indices, augmented_data.edge_index, relabel_nodes=True)
                self.x_train = augmented_data.x[self.train_node_indices]
                self.y_train = augmented_data.y[self.train_node_indices]

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


                self.history = update_history_one_epoch(self.history, self.loss, self.loss_dict, self.acc_trn, self.acc_val, self.acc_test, self.watermark_percent_matches, 
                                                        self.match_count_wmk_with_zeros, self.confidence_wmk_with_zeros,
                                                        self.match_count_wmk_without_zeros, self.confidence_wmk_without_zeros)
                


                if self.epoch%print_every==0:
                    print_epoch_status(self.epoch, self.loss_primary_weighted, self.acc_trn, self.acc_val, self.acc_test, wmk_optimization_condition_met, self.loss_watermark_weighted, \
                                       self.match_count_wmk_with_zeros, self.match_count_wmk_without_zeros,
                                       self.confidence_wmk_with_zeros, self.confidence_wmk_without_zeros,
                                       self.match_count_not_wmk_with_zeros, self.match_count_not_wmk_without_zeros,
                                       self.confidence_not_wmk_with_zeros, self.confidence_not_wmk_without_zeros,
                                       False)

                gc.collect()
                torch.cuda.empty_cache() 


                self.data.x = unaugmented_x.clone()
                self.data.edge_index = unaugmented_edge_index.clone()
                self.data.y = unaugmented_y.clone()

                self.history['betas']=self.betas_dict
                self.history['beta_similarities'] = self.beta_similarities_dict


                if save==True:
                #     train_acc_improved, val_acc_improved, match_count_improved = False, False, False
                #     if self.acc_trn > self.best_train_acc:
                #         train_acc_improved=True
                #         self.best_train_acc = self.acc_trn
                #     if self.acc_val > self.best_val_acc:
                #         val_acc_improved=True
                #         self.best_val_acc = self.acc_val
                #     if self.match_count_wmk_with_zeros > self.best_match_count:
                #         match_count_improved=True
                #         self.best_match_count = self.match_count_wmk_with_zeros
                #     if train_acc_improved+val_acc_improved+match_count_improved>=2: #if two of the three improved
                #         print('-- saving')
                    with open(os.path.join(get_results_folder_name(self.dataset_name),'Trainer'), 'wb') as f:
                        print('results dir:',get_results_folder_name(self.dataset_name))
                        pickle.dump(self, f)
                    save_results(self.dataset_name, self.node_classifier, self.history, self.subgraph_dict, self.each_subgraph_feature_importances, self.each_subgraph_watermark_indices)


        self.history['betas']=self.betas_dict
        self.history['beta_similarities'] = self.beta_similarities_dict
        self.history = replace_history_Nones(self.history)
        if save==True:

            save_results(self.dataset_name, self.node_classifier, self.history, self.subgraph_dict, self.each_subgraph_feature_importances, self.each_subgraph_watermark_indices, verbose=False)

        return self.node_classifier, self.history, self.subgraph_dict, self.each_subgraph_feature_importances, self.each_subgraph_watermark_indices
                    
    def train_clf_only(self, save=True, print_every=1):
        augment_seed=config.seed
        for epoch in tqdm(range(self.epochs)):
            self.epoch=epoch
            augment_seed=update_seed(augment_seed)
            augmented_data = augment_data(self.data, self.node_aug, self.edge_aug, self.train_nodes_to_consider_mask, None,seed=augment_seed)

            ### for evaluating on train set -- not just train mask, but (train_mask - subgraph_indices)
            train_node_indices = self.train_nodes_to_consider_mask.nonzero(as_tuple=True)[0]
            self.edge_index_train, _ = subgraph(train_node_indices, augmented_data.edge_index, relabel_nodes=True)
            self.x_train = augmented_data.x[train_node_indices]
            self.y_train = augmented_data.y[train_node_indices]

            ### for evaluating on validation set
            val_node_indices = self.val_mask.nonzero(as_tuple=True)[0]
            self.edge_index_val, _ = subgraph(val_node_indices, augmented_data.edge_index, relabel_nodes=True)
            self.x_val = augmented_data.x[val_node_indices]
            self.y_val = augmented_data.y[val_node_indices]



            closure = self.closure_primary
            if config.optimization_kwargs['use_sam']==True:
                self.optimizer.step(closure)
            else:
                closure()
                self.optimizer.step()



            self.history = update_history_one_epoch(self.history, self.loss, self.loss_dict, self.acc_trn, self.acc_val, self.acc_test, None, None, None, None, None)
            if self.epoch%print_every==0:
                print_epoch_status(self.epoch, self.loss_primary, self.acc_trn, self.acc_val, 
                                   True, None, None, None, None, None, None, None, None, None, True)

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

    def closure_primary(self):
        self.optimizer.zero_grad()
        # optimize
        log_logits_train    = self.forward(self.x_train, self.edge_index_train, dropout=config.node_classifier_kwargs['dropout'], mode='train')
        self.loss_primary   = F.nll_loss(log_logits_train, self.y_train)
        self.loss_dict, self.unweighted_total, _ = self.get_weighted_losses('primary', self.loss_primary)
        self.loss = self.loss_primary = self.loss_primary_weighted = self.unweighted_total

        # eval
        log_logits_train_eval = self.forward(self.x_train_unaugmented, self.edge_index_train_unaugmented, dropout=0, mode='eval')
        log_logits_val        = self.forward(self.x_val, self.edge_index_val, dropout=0, mode='eval')
        log_logits_test        = self.forward(self.x_test, self.edge_index_test, dropout=0, mode='eval')
        self.acc_trn  = accuracy(log_logits_train_eval, self.y_train_unaugmented,verbose=False)
        self.acc_val  = accuracy(log_logits_val, self.y_val,verbose=False)
        self.acc_test = accuracy(log_logits_test, self.y_test,verbose=False)
 
        self.backward([self.loss], verbose=False, retain_graph=False)
        return self.loss
    
    def closure_watermark(self):
        self.optimizer.zero_grad()
        # optimize
        self.node_classifier.train()
        log_logits_train          = self.forward(self.x_train, self.edge_index_train, dropout=config.node_classifier_kwargs['dropout'], mode='train')
        self.probas_dict = self.separate_forward_passes_per_subgraph_(mode='train')
        self.probas_dict_not_wmk = separate_forward_passes_per_subgraph(self.subgraph_dict_not_wmk, self.node_classifier, mode='eval')
        if config.watermark_kwargs['watermark_type']=='unimportant' and self.epoch==config.watermark_kwargs['unimportant_selection_kwargs']['clf_only_epochs']:
            self.subgraph_dict = self.apply_watermark_()
        elif config.watermark_kwargs['watermark_type']!='unimportant' and self.epoch==0:
            self.subgraph_dict = self.apply_watermark_()
        self.loss_primary = F.nll_loss(log_logits_train, self.y_train)

        # eval
        log_logits_train_eval   = self.forward(self.x_train_unaugmented, self.edge_index_train_unaugmented, dropout=0, mode='eval')
        log_logits_val      = self.forward(self.x_val, self.edge_index_val, dropout=0, mode='eval')
        log_logits_test      = self.forward(self.x_test, self.edge_index_test, dropout=0, mode='eval')
        self.acc_trn = accuracy(log_logits_train_eval, self.y_train_unaugmented,verbose=False)
        self.acc_val = accuracy(log_logits_val, self.y_val,verbose=False)
        self.acc_test = accuracy(log_logits_test, self.y_test,verbose=False)

        is_last_epoch = self.epoch==self.epochs-1
        self.get_watermark_performance_(is_last_epoch)
        self.reg = get_reg_term(self.betas_from_every_subgraph)
        self.loss_dict, self.unweighted_total, self.weighted_total = self.get_weighted_losses('combined', self.loss_primary, self.loss_watermark, self.reg)
        self.loss = self.weighted_total
        self.unweighted_losses = [self.loss_dict[k] for k in ['loss_primary','loss_watermark','reg']]
        self.weighted_losses   = [self.loss_dict[k] for k in ['loss_primary_weighted','loss_watermark_weighted','reg_weighted']]
        self.loss_primary_weighted, self.loss_watermark_weighted = self.weighted_losses[:2]
        self.weighted_losses_backward = self.weighted_losses[:2] if self.weighted_losses[2] is None else self.weighted_losses
        self.backward(self.weighted_losses_backward, verbose=False, retain_graph=False)


    def apply_watermark_(self):
        watermark_type = config.watermark_kwargs['watermark_type']
        len_watermark = int(config.watermark_kwargs['percent_of_features_to_watermark']*self.num_features/100)
        subgraph_x_concat = torch.concat([self.subgraph_dict[k]['subgraph'].x for k in self.subgraph_dict.keys()])
        subgraph_dict, self.each_subgraph_watermark_indices, self.each_subgraph_feature_importances, watermarks = apply_watermark(watermark_type, self.num_features, len_watermark, self.subgraph_dict, subgraph_x_concat, #self.probas, 
                                                                                                                                       self.probas_dict, config.watermark_kwargs, seed=config.seed)
        del subgraph_x_concat
        torch.cuda.empty_cache()

        for i, subgraph_sig in enumerate(self.subgraph_dict_not_wmk.keys()):
            self.subgraph_dict_not_wmk[subgraph_sig]['watermark']=watermarks[i]
        del watermarks
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
        self.loss_watermark, self.beta_similarity, self.betas_from_every_subgraph, self.betas_dict, self.beta_similarities_dict, \
            self.watermark_percent_matches, self.match_count_wmk_with_zeros, self.match_count_wmk_without_zeros, \
                self.confidence_wmk_with_zeros,  self.confidence_wmk_without_zeros = get_watermark_performance(self.probas_dict, 
                                                                                                                self.subgraph_dict, 
                                                                                                                self.betas_dict, 
                                                                                                                self.beta_similarities_dict, 
                                                                                                                is_last_epoch,
                                                                                                                self.debug_multiple_subgraphs, 
                                                                                                                self.beta_weights,
                                                                                                                penalize_similar_subgraphs=config.optimization_kwargs['penalize_similar_subgraphs'],
                                                                                                                shifted_subgraph_loss_coef=config.optimization_kwargs['shifted_subgraph_loss_coef'])


        self.loss_watermark_not_wmk, self.beta_similarity_not_wmk, self.betas_from_every_subgraph_not_wmk, \
            self.betas_dict_not_wmk, self.beta_similarities_dict_not_wmk, self.watermark_percent_matches_not_wmk, \
                self.match_count_not_wmk_with_zeros, self.match_count_not_wmk_without_zeros, \
                self.confidence_not_wmk_with_zeros,  self.confidence_not_wmk_without_zeros = get_watermark_performance(self.probas_dict_not_wmk, 
                                                                                                                        self.subgraph_dict_not_wmk, 
                                                                                                                        self.betas_dict_not_wmk, 
                                                                                                                        self.beta_similarities_dict_not_wmk, 
                                                                                                                        is_last_epoch,
                                                                                                                        self.debug_multiple_subgraphs, 
                                                                                                                        self.beta_weights_not_wmk,
                                                                                                                        penalize_similar_subgraphs=config.optimization_kwargs['penalize_similar_subgraphs'],
                                                                                                                        shifted_subgraph_loss_coef=config.optimization_kwargs['shifted_subgraph_loss_coef'])








def update_history_one_epoch(history, loss, loss_dict, acc_trn, acc_val, test_acc, watermark_percent_matches, 
                             match_count_with_zeros, confidence_with_zeros,
                             match_count_without_zeros, confidence_without_zeros):
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
    history['test_accs'].append(test_acc)
    history['watermark_percent_matches'].append(watermark_percent_matches)
    history['match_counts_with_zeros'].append(match_count_with_zeros)
    history['match_counts_without_zeros'].append(match_count_without_zeros)
    history['match_count_confidence_with_zeros'].append(confidence_with_zeros)
    history['match_count_confidence_without_zeros'].append(confidence_without_zeros)

    return history



def gather_random_subgraphs_not_wmk(data, dataset_name, 
                                            subgraph_method='random',
                                        size = 0.001,
                                        size_type='subgraph_size_as_fraction',
                                        seed=0,
                                        num_subgraphs=100):

    subgraph_kwargs_ =   {'method': 'random',  'subgraph_size_as_fraction': None,  'numSubgraphs': 1,
                          'khop_kwargs':      {'autoChooseSubGs': True,   'nodeIndices':  None,   'numHops': None,   'max_degree': None},
                          'random_kwargs':    {},
                          'rwr_kwargs':       {'restart_prob':None,       'max_steps':1000}}


    use_train_mask=True
    avoid_indices_set = set()
    new_graph_seed = seed

    assert size_type in ['subgraph_size_as_fraction','num_nodes']

    subgraphs = []
    subgraph_kwargs_['method'] = subgraph_method
    if size_type=='subgraph_size_as_fraction':
        sub_size_as_fraction = subgraph_kwargs_['subgraph_size_as_fraction'] = size
        num_watermarked_nodes = int(sub_size_as_fraction*sum(data.train_mask))
    elif size_type=='num_nodes':
        num_watermarked_nodes = size


    for i in range(num_subgraphs):
        new_graph_seed=update_seed(new_graph_seed)
        np.random.seed(new_graph_seed)

        print(f'Forming subgraph {i+1} of {num_subgraphs}: {subgraph_kwargs_["method"]}')
        ''' subgraphs constructed from random nodes '''
        central_node=None

        data_sub, _, subgraph_node_indices = generate_subgraph(data, dataset_name, subgraph_kwargs_, central_node, list(avoid_indices_set), use_train_mask, show=False,overrule_size_info=True,explicit_size_choice=num_watermarked_nodes, seed=seed)
        try:
            avoid_indices_set.update(node_index.item() for node_index in subgraph_node_indices)
        except:
            avoid_indices_set.update(node_index.item() for node_index in subgraph_node_indices)

        subgraphs.append({'subgraph':data_sub,'nodeIndices':subgraph_node_indices})
        
        del data_sub, subgraph_node_indices
        gc.collect()

    return subgraphs


        # 'losses': [], 
        # 'losses_primary': [], 'losses_watermark': [], 'regs':[], 
        # 'losses_primary_weighted': [], 'losses_watermark_weighted': [], 'regs_weighted':[], 
        # 'betas': [], 'beta_similarities': [], 'train_accs': [], 'val_accs': [], 'watermark_percent_matches': [], 
        # 'match_counts_with_zeros': [], 'match_counts_without_zeros':[],
        # 'match_count_confidence_with_zeros': [], 'match_count_confidence_without_zeros': []


def get_performance_trends(history, subgraph_dict):
    primary_loss_curve = history['losses_primary'] 
    primary_acc_curve = history['train_accs']
    train_acc = np.round(history['train_accs'][-1],3)
    val_acc = np.round(history['val_accs'][-1],3)
    test_acc = np.round(history['test_accs'][-1],3)

    match_counts_with_zeros = history['match_counts_with_zeros'][-1]
    match_counts_without_zeros = history['match_counts_without_zeros'][-1]
    match_count_confidence_with_zeros = np.round(history['match_count_confidence_with_zeros'][-1],3)
    match_count_confidence_without_zeros = np.round(history['match_count_confidence_without_zeros'][-1],3)


    if config.optimization_kwargs['clf_only']==True:
        return primary_loss_curve, None, None, None, None, None, None, primary_acc_curve, None, train_acc, val_acc, test_acc
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
    
    return primary_loss_curve, watermark_loss_curve, final_betas, watermarks, percent_matches, percent_match_mean, percent_match_std, primary_acc_curve, watermark_acc_curve, train_acc, val_acc, test_acc, match_counts_with_zeros, match_counts_without_zeros,match_count_confidence_with_zeros,match_count_confidence_without_zeros


               
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
    f.close()
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
    if is_notebook()==False:
        matplotlib.use('Agg')  # Use non-interactive backend only if not in a notebook    
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
    match_counts_with_zeros = []
    match_counts_without_zeros = []
    for i in range(n):
        if verbose:
            print(f'{i}/{len(n_tuplets)}',end='\r')
        bs = torch.vstack([betas_random__[j] for j in n_tuplets[i]])
        match_counts_with_zeros.append(count_matches(bs, ignore_zeros=False))
        match_counts_without_zeros.append(count_matches(bs, ignore_zeros=True))

    sample_mean_matches_with_zeros = np.mean(match_counts_with_zeros)
    sample_std_matches_with_zeros = np.std(match_counts_with_zeros, ddof=1)
    sample_mean_matches_without_zeros = np.mean(match_counts_without_zeros)
    sample_std_matches_without_zeros = np.std(match_counts_without_zeros, ddof=1)
    return match_counts_with_zeros, match_counts_without_zeros, sample_mean_matches_with_zeros, sample_mean_matches_without_zeros, sample_std_matches_with_zeros, sample_std_matches_without_zeros

def get_necessary_number_of_matches(sample_mean_matches, sample_std_matches, desired_p_value, n, verbose=False):
    desired_p_value = 0.05
    critical_z = stats.norm.ppf(1 - desired_p_value)
    matches_required = sample_mean_matches + (critical_z * sample_std_matches)
    matches_required = int(np.ceil(matches_required))
    if verbose:
        print(f"To obtain p-value={desired_p_value}, need {matches_required} matches needed across {n} sign(beta) tensors")
    return matches_required

def compute_likelihood_of_matches(betas_wmk__, sample_mean_matches, sample_std_matches, verbose=False):
    ''' watermarked-subgraphs '''
    bs = torch.vstack(betas_wmk__)
    test_value_with_zeros = count_matches(bs, ignore_matches=False)                
    z_score_with_zeros = (test_value_with_zeros - sample_mean_matches)/sample_std_matches
    p_value_with_zeros = 1 - stats.norm.cdf(z_score_with_zeros)
    test_value_without_zeros = count_matches(bs, ignore_matches=True)                 
    z_score_without_zeros = (test_value_without_zeros - sample_mean_matches)/sample_std_matches
    p_value_without_zeros = 1 - stats.norm.cdf(z_score_without_zeros)
    if verbose:
        print(f'\nPopulation Mean, Standard Error: {np.round(sample_mean_matches,3)}, {np.round(sample_std_matches,3)}')
        print(f'# Matches among the {len(bs)} watermarked betas: {test_value_with_zeros} with zeros, {test_value_without_zeros} without zeros\n')
        print(f'(z_score, p_value): with zeros = ({np.round(z_score_with_zeros,3)}, {np.round(p_value_with_zeros,5)}), without zeros = ({np.round(z_score_without_zeros,3)}, {np.round(p_value_without_zeros,5)})')
    return test_value_with_zeros, z_score_with_zeros, p_value_with_zeros, test_value_without_zeros, z_score_without_zeros, p_value_without_zeros

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

def get_confidence_matches(matches, mu_natural, sigma_natural):
    z_score = (matches - mu_natural) / sigma_natural
    one_tail_confidence_level = norm.cdf(z_score)
    return one_tail_confidence_level


def find_min_n_uncertain(n_features, mu_natural, sigma_natural, c_LB, c_t, test_effective=False, verbose=False):
    z_t = norm.ppf(c_t)
    t = np.ceil(min(mu_natural +z_t*sigma_natural,n_features))
    z_LB = norm.ppf (c_LB)
    LB = max(mu_natural - z_LB*sigma_natural,0)
    net_matches_needed = max(0,min(t - LB, n_features))
    # p_effective = (n_features-LB)/n_features
    p_effective = (n_features-t)/n_features
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




def easy_run_node_classifier(Trainer_object, node_classifier, data, mu_natural, sigma_natural, subgraph_dict, subgraph_dict_not_wmk, watermark_loss_kwargs, optimization_kwargs, 
                             regression_kwargs, node_classifier_kwargs, task='prune', target_confidence=0.99):


    all_subgraph_indices = []
    for sig in subgraph_dict.keys():
        nodeIndices = subgraph_dict[sig]['nodeIndices'].tolist()
        all_subgraph_indices += nodeIndices
    all_subgraph_indices    = torch.tensor(all_subgraph_indices)
    

    sig_0 = list(subgraph_dict.keys())[0]
    watermark = subgraph_dict[sig_0]['watermark']


    # node_classifier = Trainer_object.node_classifier
    node_classifier.eval()
    log_logits_train = node_classifier(Trainer_object.x_train_unaugmented,Trainer_object.edge_index_train_unaugmented, 0)
    acc_train = accuracy(log_logits_train, Trainer_object.y_train_unaugmented)

    # edge_index_test, _ = subgraph(trainer.test_mask, data.edge_index, relabel_nodes=True)
    # x_test, y_test = data.x[data.test_mask], data.y[data.test_mask]
    node_classifier.eval()
    log_logits_test = node_classifier(Trainer_object.x_test, Trainer_object.edge_index_test, 0)
    acc_test = accuracy(log_logits_test, Trainer_object.y_test)

    # edge_index_val, _ = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)
    # x_val, y_val = data.x[data.val_mask], data.y[data.val_mask]
    node_classifier.eval()
    log_logits_val = node_classifier(Trainer_object.x_val, Trainer_object.edge_index_val, 0)
    acc_val = accuracy(log_logits_val, Trainer_object.y_val)


    probas_dict  = separate_forward_passes_per_subgraph(subgraph_dict, node_classifier, mode='eval')
    beta_weights = torch.ones(len(subgraph_dict),data.num_features)     
    sign_betas, match_rates = [], []
    for s, sig in enumerate(subgraph_dict.keys()):
        beta_weights_ = beta_weights[s]
        _, this_percent_match, _, this_raw_beta = get_watermark_performance_single_subgraph(probas_dict,subgraph_dict, sig,ignore_zeros_from_subgraphs=False, debug=False,
                                                                                            beta_weights=beta_weights_,similar_subgraph=False, watermark_loss_kwargs=watermark_loss_kwargs,
                                                                                            regression_kwargs=regression_kwargs, optimization_kwargs=optimization_kwargs, this_watermark=None)
        sign_betas.append(torch.sign(this_raw_beta))
        match_rates.append(this_percent_match)

    stacked_sign_betas = torch.vstack(sign_betas)
    match_count_wmk_with_zeros = count_matches(stacked_sign_betas, ignore_zeros=False)
    match_count_wmk_without_zeros = count_matches(stacked_sign_betas, ignore_zeros=True)


    z_t = norm.ppf(target_confidence)
    target_number_matches = np.ceil(min(mu_natural +z_t*sigma_natural,data.x.shape[1]))
    print(f'Target # matches for {(100*target_confidence):3f}% confidence:',target_number_matches)
    confidence_wmk_with_zeros = get_confidence_matches(match_count_wmk_with_zeros, mu_natural, sigma_natural)
    confidence_wmk_without_zeros = get_confidence_matches(match_count_wmk_without_zeros, mu_natural, sigma_natural)


    probas_dict_not_wmk  = separate_forward_passes_per_subgraph(subgraph_dict_not_wmk, node_classifier, mode='eval')
    sign_betas_not_wmk = []
    match_rates_not_wmk= []
    for s, sig in enumerate(subgraph_dict_not_wmk.keys()):
        beta_weights_ = beta_weights[s]
        _, this_percent_match_not_wmk, _, this_raw_beta_not_wmk = get_watermark_performance_single_subgraph(probas_dict_not_wmk,subgraph_dict_not_wmk, sig,ignore_zeros_from_subgraphs=False, 
                                                                                                                debug=False,beta_weights=beta_weights_,similar_subgraph=False,
                                                                                                                watermark_loss_kwargs=watermark_loss_kwargs,
                                                                                                                regression_kwargs=regression_kwargs,
                                                                                                                optimization_kwargs=optimization_kwargs,
                                                                                                                this_watermark=watermark)
        sign_betas_not_wmk.append(torch.sign(this_raw_beta_not_wmk))
        match_rates_not_wmk.append(this_percent_match_not_wmk)

    stacked_sign_betas_not_wmk = torch.vstack(sign_betas_not_wmk)



    match_count_not_wmk_with_zeros = count_matches(stacked_sign_betas_not_wmk, ignore_zeros=False)
    match_count_not_wmk_without_zeros = count_matches(stacked_sign_betas_not_wmk, ignore_zeros=True)
    confidence_not_wmk_with_zeros = get_confidence_matches(match_count_not_wmk_with_zeros, mu_natural, sigma_natural)
    confidence_not_wmk_without_zeros = get_confidence_matches(match_count_not_wmk_without_zeros, mu_natural, sigma_natural)


    return match_rates, acc_train, acc_test, acc_val, target_number_matches, \
        [match_count_wmk_with_zeros, match_count_wmk_without_zeros, confidence_wmk_with_zeros, confidence_wmk_without_zeros], \
        [match_count_not_wmk_with_zeros, match_count_not_wmk_without_zeros, confidence_not_wmk_with_zeros, confidence_not_wmk_without_zeros]



def get_node_classifier_and_optimizer_and_subgraph_dict_for_further_processing(model_path, lr):
    subgraph_dict   = pickle.load(open(os.path.join(model_path,'subgraph_dict'),'rb'))
    node_classifier = pickle.load(open(os.path.join(model_path,'node_classifier'),'rb'))
    params_         = list(node_classifier.parameters())
    optimizer       = optim.Adam(params_, lr=lr)
    return node_classifier, optimizer, subgraph_dict
