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

from graphlime import GraphLIME

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
              save=False,
              verbose=True):
    train_ratio, val_ratio, test_ratio = train_val_test_split


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
                if verbose==True:
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
    

def prep_data_graph_clf(dataset_name='CS', 
              location='default', 
              batch_size='default',
              transform_list = 'default', #= NormalizeFeatures())
              train_val_test_split=[0.6,0.2,0.2],
              seed=0,
              load=True,
              save=False,
              verbose=True):
    train_ratio, val_ratio, test_ratio = train_val_test_split


    # class_ = dataset_attributes[dataset_name]['class']
    # single_or_multi_graph = dataset_attributes[dataset_name]['single_or_multi_graph']
# pyg_dataset = torch_geometric.datasets.TUDataset(root=f'/tmp/{dataset}', name=dataset,cleaned=cleaned,use_edge_attr=use_edge_attr)


    # if location=='default':
    #     location = '../data' if dataset_name in ['CORA','CiteSeer','PubMed','computers','photo','PPI','NELL','TWITCH_EN','CS'] else f'../data/{dataset_name}' if dataset_name in ['Flickr','Reddit','Reddit2'] else None
    # if batch_size=='default':
    #     batch_size = 'All'
    # if transform_list=='default':
    #     transform_list = []
    #     if dataset_name in ['CS','PubMed']:
    #         transform_list = [CreateMaskTransform(train_ratio, val_ratio, test_ratio, seed)]
    #     if dataset_name in ['computers', 'photo']:
    #         transform_list = [CreateMaskTransform(train_ratio, val_ratio, test_ratio, seed)]
    # transform = Compose(transform_list)


    saved_location = f'../data/{dataset_name}/load_this_dataset_trn_{train_val_test_split[0]:.2f}_val_{train_val_test_split[1]:2f}_test_{train_val_test_split[2]:2f}.pkl'

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
        # dataset = torch_geometric.datasets.TUDataset(root=f'/tmp/{dataset}', name=dataset,cleaned=cleaned,use_edge_attr=use_edge_attr)

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
    # print('pred:',pred)
    # print('labels:',labels)
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
        del data_sub, x_sub, y_sub, omit_indices, beta
    del subgraph_dict
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



def augment_data(data, clf_only, node_aug, edge_aug, train_nodes_to_consider, all_subgraph_indices, sampling_used=False, seed=0):
    p = config.augment_kwargs['p']
    # new_data = data

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

    if config.augment_kwargs['separate_trainset_from_subgraphs'] == True and config.using_our_method==True:
        if sampling_used==True:
            original_node_indices = data.node_idx
            original_to_new_node_mapping = {original_idx.item():new_idx for (new_idx,original_idx) in zip(range(len(original_node_indices)), original_node_indices)}
            train_nodes_to_consider = torch.tensor([original_to_new_node_mapping[original_idx] for original_idx in train_nodes_to_consider])
            all_subgraph_indices    = torch.tensor([original_to_new_node_mapping[original_idx] for original_idx in all_subgraph_indices])

        trn_minus_subgraph_nodes = torch.tensor(list(set(train_nodes_to_consider.tolist())-set(all_subgraph_indices)))
        trn_minus_subgraph_nodes_keep = select_random_indices(trn_minus_subgraph_nodes, p)
        train_minus_subgraph_data = get_subgraph_from_node_indices(data, trn_minus_subgraph_nodes_keep)
        train_minus_subgraph_data = apply_augmentations(train_minus_subgraph_data, [node_aug, edge_aug])
        update_data(data, trn_minus_subgraph_nodes_keep, train_minus_subgraph_data)
        if config.augment_kwargs['ignore_subgraphs']==False:
            all_subgraph_indices_keep = select_random_indices(all_subgraph_indices,p)
            subgraph_data = get_subgraph_from_node_indices(data, all_subgraph_indices_keep)
            subgraph_data = apply_augmentations(subgraph_data, [node_aug, edge_aug])
            update_data(data, all_subgraph_indices_keep, subgraph_data)
            del subgraph_data
            del all_subgraph_indices_keep
        del trn_minus_subgraph_nodes_keep
        del train_minus_subgraph_data

    elif config.augment_kwargs['separate_trainset_from_subgraphs'] == False or config.using_our_method==False:
        data = apply_augmentations(data, [node_aug, edge_aug])
    return data 




    # history['train_accs'].append(acc_trn)
    # history['val_accs'].append(acc_val)
    # history['watermark_percent_matches'].append(watermark_percent_matches)
    # history['match_counts_with_zeros'].append(match_count_with_zeros)
    # history['match_counts_without_zeros'].append(match_count_without_zeros)
    # history['match_count_confidence_with_zeros'].append(confidence_with_zeros)
    # history['match_count_confidence_without_zeros'].append(confidence_without_zeros)

def setup_history(clf_only=False, watermark_random_backdoor=False, watermark_graphlime_backdoor=False, subgraph_signatures=None):
    if clf_only==False:
        assert subgraph_signatures is not None
    history = {
        'losses': [], 
        'losses_primary': [], 'losses_watermark': [], #'regs':[], 
        'losses_primary_weighted': [], 'losses_watermark_weighted': [], #'regs_weighted':[], 
        'betas': [], 'beta_similarities': [], 'train_accs': [], 'val_accs': [], 'test_accs': [],'watermark_percent_matches': [], 
        'match_counts_with_zeros': [], 'match_counts_without_zeros':[],
        'match_count_confidence_with_zeros': [], 'match_count_confidence_without_zeros': []
    }
    if watermark_random_backdoor==True:
        history['trigger_accs']=[]
    if watermark_graphlime_backdoor==True:
        history['graphlime_backdoor_accs']=[]

    betas_dict = {sig: [] for sig in subgraph_signatures} if clf_only==False else {}
    beta_similarities_dict = {sig: None for sig in subgraph_signatures} if clf_only==False else {}
    return history, betas_dict, beta_similarities_dict

def setup_subgraph_dict(data, dataset_name, not_wmk=False, seed=0):
    subgraph_kwargs = config.subgraph_kwargs
    subgraph_dict, all_subgraph_indices = collect_subgraphs_within_single_graph_for_watermarking(data, dataset_name, use_train_mask=True, subgraph_kwargs=subgraph_kwargs, not_watermarked=not_wmk, seed=seed, verbose=True)
    subgraph_signatures = list(subgraph_dict.keys())
    return subgraph_dict, subgraph_signatures, all_subgraph_indices



def print_epoch_status(epoch, loss_primary, acc_trn, acc_val, acc_test, condition_met=False, loss_watermark=None, \
                       match_count_with_zeros=None, match_count_without_zeros=None, 
                       confidence_with_zeros=None, confidence_without_zeros=None,
                       match_count_not_wmk_with_zeros=None, match_count_not_wmk_without_zeros=None,\
                       confidence_not_wmk_with_zeros=None, confidence_not_wmk_without_zeros=None, 
                       clf_only=True, acc_trn_KD=None, acc_val_KD=None, acc_test_KD=None, distillation_loss=None,acc_trigger=None,acc_graphlime_backdoor=None,
                       additional_content=''):
    


    additional_content = ', ' + additional_content if len(additional_content)>0 else additional_content
    
    if config.is_kd_attack==True:
        epoch_printout = f'Epoch: {epoch:3d}, L_clf/L_KD = {loss_primary:.3f}/{distillation_loss:.3f}, acc_clf (trn/test)= {acc_trn:.3f}/{acc_test:.3f} acc_KD (trn/test)= {acc_trn_KD:.3f}/{acc_test_KD:.3f}' + additional_content    
    elif clf_only==True:
        epoch_printout = f'Epoch: {epoch:3d}, L_clf = {loss_primary:.3f}, acc (trn/val/test)= {acc_trn:.3f}/{acc_val:.3f}/{acc_test:.3f}' + additional_content
    elif config.watermark_random_backdoor==True:
        epoch_printout = f'Epoch: {epoch:3d}, L_clf = {loss_primary:.3f}, acc (trn/val/test)= {acc_trn:.3f}/{acc_val:.3f}/{acc_test:.3f}, trigger acc = {acc_trigger:.3f}' + additional_content
    elif config.watermark_graphlime_backdoor==True:
        epoch_printout = f'Epoch: {epoch:3d}, L_clf = {loss_primary:.3f}, acc (trn/val/test)= {acc_trn:.3f}/{acc_val:.3f}/{acc_test:.3f}, backdoor nodes acc = {acc_graphlime_backdoor:.3f}' + additional_content    
    else:
        if condition_met:
           epoch_printout = f'Epoch: {epoch:3d}, L (clf/wmk) = {loss_primary:.3f}/{loss_watermark:.3f}, acc (trn/val/test)= {acc_trn:.3f}/{acc_val:.3f}/{acc_test:.3f}, #_match_WMK w/wout 0s = {match_count_with_zeros}/{match_count_without_zeros}, conf w/wout 0s = {confidence_with_zeros:.3f}/{confidence_without_zeros:.3f}, #_match_NOT_WMK w/wout 0s = {match_count_not_wmk_with_zeros}/{match_count_not_wmk_without_zeros}, conf w/wout 0s= {confidence_not_wmk_with_zeros:.3f}/{confidence_not_wmk_without_zeros:.3f}' + additional_content
        else:
          epoch_printout = f'Epoch: {epoch:3d}, L_clf = {loss_primary:.3f}, L_wmk = n/a, B*W = n/a, trn acc = {acc_trn:.3f}, val acc = {acc_val:.3f}, test_acc = {acc_test:.3f}' + additional_content
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
                    #    optimization_kwargs=None,
                       this_watermark=None):
    
    
    if watermark_loss_kwargs==None:
        watermark_loss_kwargs = config.watermark_loss_kwargs
    if regression_kwargs==None:
        regression_kwargs = config.regression_kwargs
    # if optimization_kwargs==None:
        # optimization_kwargs = config.optimization_kwargs

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

    del omit_indices, not_omit_indices, beta, B_x_W, this_sign_beta, this_matches, watermark_non_zero
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



# def get_reg_term(betas_from_every_subgraph):
#     regularization_type = config.optimization_kwargs['regularization_type']
#     print('regularization_type:',regularization_type)
#     lambda_l2 = config.optimization_kwargs['lambda_l2']
#     if regularization_type==None:
#         return None
#     else:
#         if regularization_type=='L2':
#             reg = sum(torch.norm(betas_from_every_subgraph[i]) for i in range(len(betas_from_every_subgraph)))
#             reg = reg*lambda_l2
#         elif regularization_type=='beta_var':
#             inter_tensor_variance = torch.std(betas_from_every_subgraph, dim=0, unbiased=False)
#             reg = torch.sum(inter_tensor_variance)
#         return reg


def get_subgraph_from_node_indices(data, node_indices):
    sub_edge_index, _ = subgraph(node_indices, data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes)
    sub_data = Data(
        x=data.x[node_indices] if data.x is not None else None,
        edge_index=sub_edge_index,
        y=data.y[node_indices] if data.y is not None else None,
        train_mask=data.train_mask[node_indices] if data.train_mask is not None else None,
        test_mask=data.test_mask[node_indices] if data.test_mask is not None else None,
        val_mask=data.val_mask[node_indices] if data.val_mask is not None else None)
    del sub_edge_index
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


    

def get_watermark_performance(probas_dict, subgraph_dict, betas_dict, beta_similarities_dict, is_last_epoch,
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
    
        del this_raw_beta, this_beta_similarity


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
    loss_dict = {'loss_primary':torch.tensor(0.0),   'loss_watermark': torch.tensor(0.0),  # 'reg':torch.tensor(0.0),
                 'loss_primary_weighted':torch.tensor(0.0),  'loss_watermark_weighted':torch.tensor(0.0),    # 'reg_weighted':torch.tensor(0.0)
                 }
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


def continue_interrupted_training(dataset_name, arch, num_subgraphs, seed, nLayers=None, hDim=None, dropout=None, train_ratio=0.6, previous_starting_epoch=0):
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(parent_dir,'data',dataset_name)
    assert os.path.exists(data_dir)

    val_ratio = test_ratio = (1-train_ratio)/2
    dataset = prep_data(dataset_name=dataset_name, location='default', batch_size='default', transform_list='default',  train_val_test_split=[train_ratio,val_ratio,test_ratio], seed=seed, load=True)
    data = dataset[0]

    get_presets(dataset,dataset_name)
    del dataset

    config.seed = seed
    config.node_classifier_kwargs['arch']=arch
    if nLayers is not None:
        config.node_classifier_kwargs['nLayers']=nLayers
    if hDim is not None:
        config.node_classifier_kwargs['hDim']=hDim
    if dropout is not None:
        config.node_classifier_kwargs['dropout']=dropout
    config.subgraph_kwargs['numSubgraphs']=num_subgraphs

    n_features = data.x.shape[1]
    c = config.subgraph_kwargs['numSubgraphs']
    mu_natural, sigma_natural = get_natural_match_distribution(n_features, c)
    print('mu_natural:',mu_natural)
    print('sigma_natural:',sigma_natural)
    target_confidence = 0.999999
    c_LB=target_confidence
    c_t=target_confidence
    recommended_watermark_length = find_min_n_uncertain(n_features, mu_natural, sigma_natural, c_LB, c_t, test_effective=True, verbose=True)
    recommended_percent = 100*recommended_watermark_length/n_features
    print(f'recommended_watermark_length for confidence={c_t}: {recommended_watermark_length}')
    print(f'recommended_percent: {recommended_percent:.3f}')
    config.watermark_kwargs['percent_of_features_to_watermark']=recommended_percent

    results_dir = get_results_folder_name(dataset_name)
    if previous_starting_epoch>0:
        Trainer_object = pickle.load(open(os.path.join(results_dir,f'Trainer_continuation_from_{previous_starting_epoch}'),'rb'))
        history = pickle.load(open(os.path.join(results_dir,f'history_continuation_from_{previous_starting_epoch}'),'rb'))
    else:
        Trainer_object = pickle.load(open(os.path.join(results_dir,f'Trainer'),'rb'))
        history = pickle.load(open(os.path.join(results_dir,f'history'),'rb'))
    total_epochs = config.optimization_kwargs['epochs']
    epochs_so_far = len(history['train_accs'])
    remaining_epochs = total_epochs-epochs_so_far

    config.node_classifier_kwargs['epochs']=Trainer_object.epochs=remaining_epochs
    Trainer_object.train(debug_multiple_subgraphs=False, save=True, print_every=1, continuation=True, starting_epoch=epochs_so_far)



# num_nodes_trigger = int(0.05*len(self.data.x))
# prob_edge = 0.1
# proportion_ones = 0.3
# num_classes = dataset_attributes[dataset_name]['num_classes']
# feature_dim = data.x.shape[1]


def create_random_trigger_graph(num_nodes_trigger, prob_edge, proportion_ones, num_classes, feature_dim, seed):
    np.random.seed(seed)
    trigger_A = np.random.rand(num_nodes_trigger, num_nodes_trigger)<prob_edge
    np.fill_diagonal(trigger_A,0)
    trigger_A = torch.tensor(trigger_A)
    # trigger_edge_index = torch.tensor(np.array(trigger_A),dtype=torch.int)

    # Get edge indices (row and column indices of non-zero elements)
    trigger_edge_index = trigger_A.nonzero(as_tuple=True)

    # Stack the indices into a 2xM tensor
    trigger_edge_index = torch.stack(trigger_edge_index, dim=0)

    print('trigger_edge_index:',trigger_edge_index)
    trigger_X = np.zeros((num_nodes_trigger,feature_dim))
    num_ones = int(feature_dim*proportion_ones)
    for i in range(num_nodes_trigger):
        ones_indices=np.random.choice(feature_dim, num_ones, replace=False)
        trigger_X[i,ones_indices]=1
    trigger_X  =torch.tensor(trigger_X,dtype=torch.float)
    np.random.seed(seed)
    trigger_y=torch.tensor(np.random.randint(0,num_classes, size=num_nodes_trigger),dtype=torch.long)
    trigger_graph = Data(x=trigger_X,edge_index=trigger_edge_index,y=trigger_y)
    return trigger_graph


class Trainer():
    def __init__(self, data, dataset_name, target_number_matches=None):
        self.data = data
        self.dataset_name = dataset_name
        self.num_features = data.x.shape[1]
        self.train_mask = data.train_mask
        self.val_mask = data.val_mask
        self.test_mask = data.test_mask
        self.optimization_kwargs = config.optimization_kwargs
        self.use_pcgrad        = self.optimization_kwargs['use_pcgrad']
        self.lr                = self.optimization_kwargs['lr']
        self.epochs            = self.optimization_kwargs['epochs']
        self.node_aug, self.edge_aug = collect_augmentations()
        self.sacrifice_method  = self.optimization_kwargs['sacrifice_kwargs']['method']
        self.loss_dict = setup_loss_dict()
        self.coefWmk = self.optimization_kwargs['coefWmk_kwargs']['coefWmk']
        self.node_classifier = Net(**config.node_classifier_kwargs)
        self.node_classifier.train()
        self.instantiate_optimizer()
        self.target_number_matches=target_number_matches
        self.best_train_acc, self.best_val_acc, self.best_match_count = 0, 0, 0
        validate_kwargs()


        self.loss = torch.tensor(0.0)
        self.loss_primary = torch.tensor(0.0)
        self.loss_watermark = torch.tensor(0.0)
        self.loss_primary_weighted = torch.tensor(0.0)
        self.loss_watermark_weighted = torch.tensor(0.0)
        self.beta_similarity = torch.tensor(0.0)
        self.loss_watermark_weighted = None
        self.match_count_wmk_with_zeros = None
        self.match_count_wmk_without_zeros = None
        self.confidence_wmk_with_zeros = None
        self.confidence_wmk_without_zeros = None
        self.match_count_not_wmk_with_zeros = None
        self.match_count_not_wmk_without_zeros = None
        self.confidence_not_wmk_with_zeros = None
        self.confidence_not_wmk_without_zeros = None
        self.watermark_percent_matches=None
        self.betas_dict=None
        self.beta_similarities_dict=None
        self.subgraph_dict=None
        self.each_subgraph_feature_importances=None
        self.each_subgraph_watermark_indices=None
        self.all_subgraph_indices = None
        self.sacrifice_method=None

        ## watermark_random_backdoor
        self.acc_trigger=None
        self.acc_graphlime_backdoor=None

        print('***')
        print('config.watermark_graphlime_backdoor:',config.watermark_graphlime_backdoor)
        print('config.using_our_method:',config.using_our_method)
        print('***')

        if config.using_our_method==False:
            self.history, _, _                = setup_history(clf_only=True, watermark_random_backdoor=config.watermark_random_backdoor, watermark_graphlime_backdoor=config.watermark_graphlime_backdoor)
            self.train_nodes_to_consider_mask = get_train_nodes_to_consider(self.data, self.all_subgraph_indices, self.sacrifice_method, self.data.x.shape[0], train_with_test_set=False)
            self.train_node_indices = self.train_nodes_to_consider_mask.nonzero(as_tuple=True)[0]
            self.train_nodes_to_consider = torch.where(self.train_nodes_to_consider_mask==True)[0]
            print('train_nodes_to_consider:',self.train_nodes_to_consider)
        # if self.optimization_kwargs['clf_only'] == True or config.watermark_random_backdoor==True:

            if config.watermark_random_backdoor==True:
                num_classes = dataset_attributes[dataset_name]['num_classes']
                prob_edge = config.watermark_random_backdoor_prob_edge
                proportion_ones = config.watermark_random_backdoor_proportion_ones
                trigger_proportion = config.watermark_random_backdoor_trigger_size_proportion
                num_nodes_trigger = int(trigger_proportion*len(self.data.x))
                self.trigger_graph = create_random_trigger_graph(num_nodes_trigger, prob_edge, proportion_ones, num_classes, data.x.shape[1], config.seed)

            elif config.watermark_graphlime_backdoor==True:
                self.data, self.graphlime_backdoor_info = backdoor_GraphLIME(dataset_name, data, self.train_nodes_to_consider, config.watermark_graphlime_backdoor_target_label,
                                                                config.watermark_graphlime_backdoor_poison_rate, config.watermark_graphlime_backdoor_size, config.seed)
                self.graphlime_backdoored_node_indices = list(self.graphlime_backdoor_info['ranked_feature_dict'].keys())

                

        elif config.using_our_method==True:
            self.subgraph_dict, self.subgraph_signatures, self.all_subgraph_indices = setup_subgraph_dict(data, dataset_name, not_wmk=False, seed=config.seed)
            self.history, self.betas_dict, self.beta_similarities_dict              = setup_history(subgraph_signatures=self.subgraph_signatures)
            self.beta_weights                                                       = get_beta_weights(self.subgraph_dict, self.num_features)
            self.subgraph_dict_not_wmk, self.subgraph_signatures_not_wmk, self.all_subgraph_indices_not_wmk = setup_subgraph_dict(data, dataset_name, not_wmk=True, seed=config.random_seed)
            self.history_not_wmk, self.betas_dict_not_wmk, self.beta_similarities_dict_not_wmk              = setup_history(subgraph_signatures=self.subgraph_signatures_not_wmk)
            self.beta_weights_not_wmk                                                                       = get_beta_weights(self.subgraph_dict_not_wmk, self.num_features)
            self.train_nodes_to_consider_mask = get_train_nodes_to_consider(self.data, self.all_subgraph_indices, self.sacrifice_method, self.data.x.shape[0], train_with_test_set=False)
            self.train_nodes_to_consider = torch.where(self.train_nodes_to_consider_mask==True)[0]
            self.each_subgraph_feature_importances, self.each_subgraph_watermark_indices, _ = None, None, None
            self.build_wmk_coef_sched(num_changes=3) # builds even if coefWmk is constant
            self.train_node_indices = self.train_nodes_to_consider_mask.nonzero(as_tuple=True)[0]

        ### for evaluating on train set -- not just train mask, but (train_mask - subgraph_indices)
        if config.preserve_edges_between_subsets==False:
            self.edge_index_train_unaugmented, _ = subgraph(self.train_node_indices, self.data.edge_index, relabel_nodes=True)
            self.edge_index_train_unaugmented = self.edge_index_train_unaugmented.clone()
        self.x_train_unaugmented = self.data.x[self.train_node_indices].clone()
        self.y_train_unaugmented = self.data.y[self.train_node_indices].clone()
        ### for evaluating on test set
        self.test_node_indices = self.data.test_mask.nonzero(as_tuple=True)[0]
        if config.preserve_edges_between_subsets==False:
            self.edge_index_test, _ = subgraph(self.test_node_indices, self.data.edge_index, relabel_nodes=True)
            self.edge_index_test = self.edge_index_test.clone()
        self.x_test = self.data.x[self.test_node_indices].clone()
        self.y_test = self.data.y[self.test_node_indices].clone()
        ### for evaluating on validation set
        self.val_node_indices = self.data.val_mask.nonzero(as_tuple=True)[0]
        if config.preserve_edges_between_subsets==False:
            self.edge_index_val, _ = subgraph(self.val_node_indices, self.data.edge_index, relabel_nodes=True)
            self.edge_index_val = self.edge_index_val.clone()
        self.x_val = self.data.x[self.val_node_indices].clone()
        self.y_val = self.data.y[self.val_node_indices].clone()
        print('len(train_node_indices):',len(self.train_node_indices))
        print('len(test_node_indices):',len(self.test_node_indices))
        print('len(val_node_indices):',len(self.val_node_indices))
        if config.watermark_graphlime_backdoor==True:
            if config.preserve_edges_between_subsets==False:
            # self.backdoored_node_indices
                self.edge_index_graphlime_backdoor, _  = subgraph(self.graphlime_backdoored_node_indices, self.data.edge_index, relabel_nodes=True)
                self.edge_index_graphlime_backdoor = self.edge_index_graphlime_backdoor.clone()
            self.x_graphlime_backdoor = self.data.x[self.graphlime_backdoored_node_indices].clone()
            self.y_graphlime_backdoor = self.data.y[self.graphlime_backdoored_node_indices].clone()
        return

    def save_process(self, continuation=False,starting_epoch=None, file_ext=None):
        if file_ext is None or file_ext == '':
            file_ext=''
        else:
            file_ext = "_"+file_ext
        if config.preserve_edges_between_subsets==True:
            file_ext = '_preserve_edges' + file_ext
        if continuation==True:
            Trainer_str = f'Trainer{file_ext}_continuation_from_{starting_epoch}'
        else:
            Trainer_str = f'Trainer{file_ext}'
        Trainer_path = os.path.join(get_results_folder_name(self.dataset_name),Trainer_str)
        with open(Trainer_path, 'wb') as f:
            pickle.dump(self, f)
        save_results(self.dataset_name, self.node_classifier, self.history, self.subgraph_dict, 
                        self.each_subgraph_feature_importances, self.each_subgraph_watermark_indices, 
                        verbose=False,continuation=continuation, starting_epoch=starting_epoch)

    def instantiate_optimizer(self):
        if self.optimization_kwargs['use_sam']==True:
            optimizer = SAM(self.node_classifier.parameters(), optim.SGD, lr=self.lr, momentum=self.optimization_kwargs['sam_momentum'],rho = self.optimization_kwargs['sam_rho'])
        else:
            optimizer = optim.Adam(self.node_classifier.parameters(), lr=self.lr)
        if self.use_pcgrad==True:
            optimizer = PCGrad(optimizer)
        self.optimizer = optimizer

    def build_wmk_coef_sched(self, num_changes=3):
        if config.watermark_kwargs['watermark_type']=='unimportant':
            wmk_start_epoch = config.watermark_kwargs['unimportant_selection_kwargs']['clf_only_epochs']
        else:
            wmk_start_epoch = 0
        if self.optimization_kwargs['coefWmk_kwargs']['schedule_coef_wmk']==True:
            min_coef = self.optimization_kwargs['coefWmk_kwargs']['min_coefWmk_scheduled']
            max_coef = self.coefWmk
            reach_max_by = self.optimization_kwargs['coefWmk_kwargs']['reach_max_coef_wmk_by_epoch']
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



    def train(self, debug_multiple_subgraphs=True, save=True, print_every=10, continuation=False, starting_epoch=0):
        clf_only = self.optimization_kwargs['clf_only']
        self.debug_multiple_subgraphs = debug_multiple_subgraphs

        if config.using_our_method==True:
            mu_natural, sigma_natural = get_natural_match_distribution(self.data.x.shape[1], len(self.subgraph_dict))
            print(f'\n\nNatural match distribution across {len(self.subgraph_dict)} tensors of length {self.data.x.shape[1]}: mu={mu_natural:.3f}, sigma={sigma_natural:.3f}\n')
            print('Target # matches:',self.target_number_matches)
        elif config.using_our_method==False:
            self.all_subgraph_indices=None
            self.train_nodes_to_consider = self.train_node_indices = self.train_nodes_to_consider_mask.nonzero(as_tuple=True)[0]
        augment_seed = config.seed
        self.unaugmented_x = self.data.x.clone()
        self.unaugmented_edge_index = self.data.edge_index.clone()
        self.unaugmented_y= self.data.y.clone()

        for epoch in tqdm(range(self.epochs)):
            # print('augment_seed:',augment_seed)
            epoch += starting_epoch
            self.epoch=epoch
            augment_seed=update_seed(augment_seed)

            self.augmented_data = augment_data(self.data, clf_only,self.node_aug, self.edge_aug, self.train_nodes_to_consider, self.all_subgraph_indices, seed=augment_seed)
            # print('x:',augmented_data.x)
            self.edge_index_train, _ = subgraph(self.train_node_indices, self.augmented_data.edge_index, relabel_nodes=True)
            self.x_train = self.augmented_data.x[self.train_node_indices]
            self.y_train = self.augmented_data.y[self.train_node_indices]
            # del augmented_data

            wmk_optimization_condition_met_op1 = config.watermark_kwargs['watermark_type']=='basic' or config.watermark_kwargs['watermark_type']=='most_represented'
            wmk_optimization_condition_met_op2 = config.watermark_kwargs['watermark_type']=='unimportant' and self.epoch>=config.watermark_kwargs['unimportant_selection_kwargs']['clf_only_epochs']
            wmk_optimization_condition_met = wmk_optimization_condition_met_op1 or wmk_optimization_condition_met_op2

            if config.using_our_method==False:
                wmk_optimization_condition_met=False

            if not wmk_optimization_condition_met:
                if config.using_our_method==True:
                    self.watermark_percent_matches = [0]*(len(self.subgraph_signatures))
                if clf_only==True:
                    closure = self.closure_primary
                elif config.watermark_random_backdoor==True:
                    closure = self.closure_random_backdoor_watermark
                elif config.watermark_graphlime_backdoor==True:
                    closure = self.closure_graphlime_backdoor_watermark
            elif wmk_optimization_condition_met:
                self.coefWmk = self.wmk_coef_schedule_dict[epoch]
                closure = self.closure_watermark

            if self.optimization_kwargs['use_sam']==True:
                self.optimizer.step(closure)
            else:
                closure()
                total_norm=0
                for p in self.node_classifier.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.optimizer.step()


            self.history = update_history_one_epoch(self.history, self.loss, self.loss_dict, self.acc_trn, self.acc_val, self.acc_test, 
                                                    self.watermark_percent_matches, self.match_count_wmk_with_zeros, self.confidence_wmk_with_zeros,
                                                    self.match_count_wmk_without_zeros, self.confidence_wmk_without_zeros, self.acc_trigger, self.acc_graphlime_backdoor)
            


            if self.epoch%print_every==0:
                
                print_epoch_status(self.epoch, self.loss_primary_weighted, self.acc_trn, self.acc_val, self.acc_test, wmk_optimization_condition_met, 
                                    self.loss_watermark_weighted, self.match_count_wmk_with_zeros, self.match_count_wmk_without_zeros,
                                    self.confidence_wmk_with_zeros, self.confidence_wmk_without_zeros,
                                    self.match_count_not_wmk_with_zeros, self.match_count_not_wmk_without_zeros,
                                    self.confidence_not_wmk_with_zeros, self.confidence_not_wmk_without_zeros,
                                    clf_only=clf_only, acc_trigger=self.acc_trigger, acc_graphlime_backdoor=self.acc_graphlime_backdoor)

            gc.collect()
            torch.cuda.empty_cache() 


            self.data.x = self.unaugmented_x.clone()
            self.data.edge_index = self.unaugmented_edge_index.clone()
            self.data.y = self.unaugmented_y.clone()

            self.history['betas']=self.betas_dict
            self.history['beta_similarities'] = self.beta_similarities_dict


            if save==True:
                if config.watermark_random_backdoor==True:
                    file_ext = '_watermark_random_backdoor'
                elif config.watermark_graphlime_backdoor==True:
                    file_ext = '_watermark_graphlime_backdoor'
                else:
                    file_ext = ''
                self.save_process(continuation,starting_epoch, file_ext=file_ext)


        self.history['betas']=self.betas_dict
        self.history['beta_similarities'] = self.beta_similarities_dict
        self.history = replace_history_Nones(self.history)
        if save==True:
            if config.watermark_random_backdoor==True:
                file_ext = '_watermark_random_backdoor'
            elif config.watermark_graphlime_backdoor==True:
                file_ext = '_watermark_graphlime_backdoor'
            else:
                file_ext = ''
            self.save_process(continuation,starting_epoch,file_ext=file_ext)
            # if continuation==True:
            #     Trainer_str = f'Trainer_continuation_from_{starting_epoch}'
            # else:
            #     Trainer_str = 'Trainer'
            # with open(os.path.join(get_results_folder_name(self.dataset_name),Trainer_str), 'wb') as f:
            #     pickle.dump(self, f)
            # save_results(self.dataset_name, self.node_classifier, self.history, self.subgraph_dict, 
            #              self.each_subgraph_feature_importances, self.each_subgraph_watermark_indices, 
            #              verbose=False,continuation=continuation, starting_epoch=starting_epoch)

        gc.collect()
        if config.using_our_method==True:
            return self.node_classifier, self.history, self.subgraph_dict, self.each_subgraph_feature_importances, self.each_subgraph_watermark_indices
        elif config.using_our_method==False:
            return self.node_classifier, self.history
        
    # def train_clf_only(self, save=True, print_every=1):
    #     train_node_indices = self.train_nodes_to_consider_mask.nonzero(as_tuple=True)[0]
    #     augment_seed=config.seed
    #     # self.edge_index_train_unaugmented = 
    #     self.unaugmented_x = self.data.x.clone()
    #     self.unaugmented_edge_index = self.data.edge_index.clone()
    #     self.unaugmented_y= self.data.y.clone()
    #     self.edge_index_train_unaugmented , _ = subgraph(train_node_indices, self.data.edge_index, relabel_nodes=True)
    #     self.x_train_unaugmented = self.data.x.clone()[train_node_indices]
    #     self.y_train_unaugmented = self.data.y.clone()[train_node_indices]

    #     ### for evaluating on validation set
    #     val_node_indices = self.val_mask.nonzero(as_tuple=True)[0]
    #     self.edge_index_val, _ = subgraph(val_node_indices, self.data.edge_index, relabel_nodes=True)
    #     self.x_val = self.data.x[val_node_indices]
    #     self.y_val = self.data.y[val_node_indices]

    #     ### for evaluating on test set
    #     test_node_indices = self.test_mask.nonzero(as_tuple=True)[0]
    #     self.edge_index_test, _ = subgraph(test_node_indices, self.data.edge_index, relabel_nodes=True)
    #     self.x_test = self.data.x[test_node_indices]
    #     self.y_test = self.data.y[test_node_indices]


    #     for epoch in tqdm(range(self.epochs)):
    #         # print('augment_seed:',augment_seed)
    #         self.epoch=epoch
    #         augment_seed=update_seed(augment_seed)
    #         augmented_data = augment_data(self.data, True, self.node_aug, self.edge_aug, self.train_nodes_to_consider_mask, None,seed=augment_seed)
    #         # print('x:',augmented_data.x)

    #         ### for evaluating on train set -- not just train mask, but (train_mask - subgraph_indices)
    #         self.edge_index_train, _ = subgraph(train_node_indices, augmented_data.edge_index, relabel_nodes=True)
    #         self.x_train = augmented_data.x[train_node_indices]
    #         self.y_train = augmented_data.y[train_node_indices]
    #         del augmented_data

    #         closure = self.closure_primary
    #         if self.optimization_kwargs['use_sam']==True:
    #             self.optimizer.step(closure)
    #         else:
    #             closure()
    #             self.optimizer.step()

    #         self.history = update_history_one_epoch(self.history, self.loss, self.loss_dict, self.acc_trn, self.acc_val, self.acc_test, None, None, None, None, None)
    #         if self.epoch%print_every==0:
    #             print_epoch_status(self.epoch, self.loss_primary, self.acc_trn, self.acc_val, self.acc_test,
    #                                True,None, None, None,None, None, None, None, None, None, True)

    #         gc.collect()

    #     self.history['betas'], self.history['beta_similarities'] = {},{} ## include for consistency with watermarking outputs
    #     if save==True:
    #         print('****')
    #         save_results(self.dataset_name, self.node_classifier, self.history)
    #     return self.node_classifier, self.history


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
        if config.preserve_edges_between_subsets==True:
            log_logits_train    = self.forward(self.augmented_data.x, self.augmented_data.edge_index, dropout=config.node_classifier_kwargs['dropout'], mode='train')
            log_logits_train    = log_logits_train[self.train_nodes_to_consider]
        elif config.preserve_edges_between_subsets==False:
            log_logits_train    = self.forward(self.x_train, self.edge_index_train, dropout=config.node_classifier_kwargs['dropout'], mode='train')
        # print('log_logits_train:',log_logits_train)
        self.loss_primary   = F.nll_loss(log_logits_train, self.y_train)
        self.loss_dict, self.unweighted_total, _ = self.get_weighted_losses('primary', self.loss_primary)
        self.loss = self.loss_primary = self.loss_primary_weighted = self.unweighted_total
        del log_logits_train

        # eval
        if config.preserve_edges_between_subsets==True:
            log_logits_eval = self.forward(self.unaugmented_x, self.unaugmented_edge_index, dropout=config.node_classifier_kwargs['dropout'], mode='eval')
            log_logits_train_eval = log_logits_eval[self.train_node_indices]
            log_logits_val = log_logits_eval[self.val_node_indices]
            log_logits_test = log_logits_eval[self.test_node_indices]
        elif config.preserve_edges_between_subsets==False:
            log_logits_train_eval = self.forward(self.x_train_unaugmented, self.edge_index_train_unaugmented, dropout=0, mode='eval')
            log_logits_val        = self.forward(self.x_val, self.edge_index_val, dropout=0, mode='eval')
            log_logits_test        = self.forward(self.x_test, self.edge_index_test, dropout=0, mode='eval')
        self.acc_trn  = accuracy(log_logits_train_eval, self.y_train_unaugmented,verbose=False)
        self.acc_val  = accuracy(log_logits_val, self.y_val,verbose=False)
        self.acc_test = accuracy(log_logits_test, self.y_test,verbose=False)
        del log_logits_train_eval, log_logits_val, log_logits_test
 
        self.backward([self.loss], verbose=False, retain_graph=False)
        # print('loss:',self.loss)
        return self.loss
    
    def closure_watermark(self):
        self.optimizer.zero_grad()
        self.node_classifier.train()

        if config.preserve_edges_between_subsets==True:
            log_logits    = self.forward(self.augmented_data.x, self.augmented_data.edge_index, dropout=config.node_classifier_kwargs['dropout'], mode='train')
            log_logits_train    = log_logits[self.train_nodes_to_consider]
        elif config.preserve_edges_between_subsets==False:
            log_logits_train          = self.forward(self.x_train, self.edge_index_train, dropout=config.node_classifier_kwargs['dropout'], mode='train')
        
        self.loss_primary = F.nll_loss(log_logits_train, self.y_train)
        if self.optimization_kwargs['clf_only']==False:
            self.probas_dict = self.separate_forward_passes_per_subgraph_(mode='train')
            self.probas_dict_not_wmk = separate_forward_passes_per_subgraph(self.subgraph_dict_not_wmk, self.node_classifier, mode='eval')
            if config.watermark_kwargs['watermark_type']=='unimportant' and self.epoch==config.watermark_kwargs['unimportant_selection_kwargs']['clf_only_epochs']:
                # self.subgraph_dict = 
                self.apply_watermark_()
            elif config.watermark_kwargs['watermark_type']!='unimportant' and self.epoch==0:
                #self.subgraph_dict = 
                self.apply_watermark_()
        del log_logits_train

        # eval
        if config.preserve_edges_between_subsets==True:
            log_logits   = self.forward(self.unaugmented_x, self.unaugmented_edge_index, dropout=0, mode='eval')
            log_logits_train_eval = log_logits[self.train_nodes_to_consider]
            log_logits_val = log_logits[self.val_node_indices]
            log_logits_test = log_logits[self.test_node_indices]
        elif config.preserve_edges_between_subsets==False:
            log_logits_train_eval   = self.forward(self.x_train_unaugmented, self.edge_index_train_unaugmented, dropout=0, mode='eval')
            log_logits_val      = self.forward(self.x_val, self.edge_index_val, dropout=0, mode='eval')
            log_logits_test      = self.forward(self.x_test, self.edge_index_test, dropout=0, mode='eval')

        self.acc_trn = accuracy(log_logits_train_eval, self.y_train_unaugmented,verbose=False)
        self.acc_val = accuracy(log_logits_val, self.y_val,verbose=False)
        self.acc_test = accuracy(log_logits_test, self.y_test,verbose=False)
        del log_logits_train_eval, log_logits_val, log_logits_test

        is_last_epoch = self.epoch==self.epochs-1
        if self.optimization_kwargs['clf_only']==False:
            self.get_watermark_performance_(is_last_epoch)
            # self.reg = get_reg_term(self.betas_from_every_subgraph)
            self.loss_dict, self.unweighted_total, self.weighted_total = self.get_weighted_losses('combined', self.loss_primary, self.loss_watermark)#self.reg)
            self.loss = self.weighted_total
            self.unweighted_losses = [self.loss_dict[k] for k in ['loss_primary','loss_watermark']]#,'reg']]
            self.weighted_losses   = [self.loss_dict[k] for k in ['loss_primary_weighted','loss_watermark_weighted']]#,'reg_weighted']]
            self.loss_primary_weighted, self.loss_watermark_weighted = self.weighted_losses#[:2]
            self.weighted_losses_backward = self.weighted_losses#[:2] if self.weighted_losses[2] is None else self.weighted_losses

        elif self.optimization_kwargs['clf_only']==True:
            self.loss_dict, self.unweighted_total, self.weighted_total = self.get_weighted_losses('primary', self.loss_primary, None, None)
            self.loss_dict, self.unweighted_total, _ = self.get_weighted_losses('primary', self.loss_primary)
            self.loss = self.loss_primary = self.loss_primary_weighted = self.unweighted_total
        self.backward(self.weighted_losses_backward, verbose=False, retain_graph=False)

    def closure_random_backdoor_watermark(self):
        self.optimizer.zero_grad()
        # optimize
        if config.preserve_edges_between_subsets==True:
            log_logits_train    = self.forward(self.augmented_data.x, self.augmented_data.edge_index, dropout=config.node_classifier_kwargs['dropout'], mode='train')
            log_logits_train    = log_logits_train[self.train_nodes_to_consider]
        elif config.preserve_edges_between_subsets==False:
            log_logits_train    = self.forward(self.x_train, self.edge_index_train, dropout=config.node_classifier_kwargs['dropout'], mode='train')
        log_logits_trigger = self.forward(self.trigger_graph.x, self.trigger_graph.edge_index, dropout=0, mode='train')

        self.loss_primary = self.loss_primary_weighted  = F.nll_loss(log_logits_train, self.y_train)
        self.loss_dict, self.unweighted_total, _ = self.get_weighted_losses('primary', self.loss_primary)
        self.loss_trigger   = F.nll_loss(log_logits_trigger, self.trigger_graph.y)
        alpha = config.watermark_random_backdoor_trigger_alpha
        self.loss = self.loss_primary + alpha*self.loss_trigger
        del log_logits_train, log_logits_trigger

        # eval
        if config.preserve_edges_between_subsets==True:
            log_logits_eval = self.forward(self.unaugmented_x, self.unaugmented_edge_index, dropout=config.node_classifier_kwargs['dropout'], mode='eval')
            log_logits_train_eval = log_logits_eval[self.train_node_indices]
            log_logits_val = log_logits_eval[self.val_node_indices]
            log_logits_test = log_logits_eval[self.test_node_indices]
        elif config.preserve_edges_between_subsets==False:
            log_logits_train_eval = self.forward(self.x_train_unaugmented, self.edge_index_train_unaugmented, dropout=0, mode='eval')
            log_logits_val        = self.forward(self.x_val, self.edge_index_val, dropout=0, mode='eval')
            log_logits_test        = self.forward(self.x_test, self.edge_index_test, dropout=0, mode='eval')
        log_logits_trigger_eval = self.forward(self.trigger_graph.x, self.trigger_graph.edge_index, dropout=0, mode='eval')

        self.acc_trigger  = accuracy(log_logits_trigger_eval, self.trigger_graph.y,verbose=False)
        self.acc_trn  = accuracy(log_logits_train_eval, self.y_train_unaugmented,verbose=False)
        self.acc_val  = accuracy(log_logits_val, self.y_val,verbose=False)
        self.acc_test = accuracy(log_logits_test, self.y_test,verbose=False)
        del log_logits_train_eval, log_logits_val, log_logits_test
 
        self.backward([self.loss], verbose=False, retain_graph=False)
        return self.loss
    
    def closure_graphlime_backdoor_watermark(self):
        self.optimizer.zero_grad()
        # optimize
        if config.preserve_edges_between_subsets==True:
            log_logits_train    = self.forward(self.augmented_data.x, self.augmented_data.edge_index, dropout=config.node_classifier_kwargs['dropout'], mode='train')
            log_logits_train    = log_logits_train[self.train_nodes_to_consider]
        elif config.preserve_edges_between_subsets==False:
            log_logits_train    = self.forward(self.x_train, self.edge_index_train, dropout=config.node_classifier_kwargs['dropout'], mode='train')
        # print('log_logits_train:',log_logits_train)
        self.loss_primary   = F.nll_loss(log_logits_train, self.y_train)
        self.loss_dict, self.unweighted_total, _ = self.get_weighted_losses('primary', self.loss_primary)
        self.loss = self.loss_primary = self.loss_primary_weighted = self.unweighted_total
        del log_logits_train

        # eval
        if config.preserve_edges_between_subsets==True:
            log_logits_eval = self.forward(self.unaugmented_x, self.unaugmented_edge_index, dropout=config.node_classifier_kwargs['dropout'], mode='eval')
            log_logits_train_eval = log_logits_eval[self.train_node_indices]
            log_logits_val = log_logits_eval[self.val_node_indices]
            log_logits_test = log_logits_eval[self.test_node_indices]
            log_logits_backdoor = log_logits_eval[self.graphlime_backdoored_node_indices]
        # if config.watermark_graphlime_backdoor==True and config.preserve_edges_between_subsets==False:
        #     # self.backdoored_node_indices
        #     self.edge_index_graphlime_backdoor, _  = subgraph(self.graphlime_backdoored_node_indices, self.data.edge_index, relabel_nodes=True)
        #     self.edge_index_graphlime_backdoor = self.edge_index_graphlime_backdoor.clone()
        # self.x_graphlime_backdoor = self.data.x[self.graphlime_backdoored_node_indices].clone()
        # self.y_graphlime_backdoor = self.data.y[self.graphlime_backdoored_node_indices].clone()

        elif config.preserve_edges_between_subsets==False:
            log_logits_train_eval = self.forward(self.x_train_unaugmented, self.edge_index_train_unaugmented, dropout=0, mode='eval')
            log_logits_val        = self.forward(self.x_val, self.edge_index_val, dropout=0, mode='eval')
            log_logits_test        = self.forward(self.x_test, self.edge_index_test, dropout=0, mode='eval')
            log_logits_backdoor        = self.forward(self.x_graphlime_backdoor, self.edge_index_graphlime_backdoor, dropout=0, mode='eval')
        self.acc_trn  = accuracy(log_logits_train_eval, self.y_train_unaugmented,verbose=False)
        self.acc_val  = accuracy(log_logits_val, self.y_val,verbose=False)
        self.acc_test = accuracy(log_logits_test, self.y_test,verbose=False)
        self.acc_graphlime_backdoor = accuracy(log_logits_backdoor, self.y_graphlime_backdoor,verbose=False)
        del log_logits_train_eval, log_logits_val, log_logits_test, log_logits_backdoor
 
        self.backward([self.loss], verbose=False, retain_graph=False)
        # print('loss:',self.loss)
        return self.loss

    def apply_watermark_(self):
        watermark_type = config.watermark_kwargs['watermark_type']
        len_watermark = int(config.watermark_kwargs['percent_of_features_to_watermark']*self.num_features/100)
        subgraph_x_concat = torch.concat([self.subgraph_dict[k]['subgraph'].x for k in self.subgraph_dict.keys()])
        self.subgraph_dict, self.each_subgraph_watermark_indices, self.each_subgraph_feature_importances, watermarks = apply_watermark(watermark_type, self.num_features, len_watermark, self.subgraph_dict, subgraph_x_concat, #self.probas, 
                                                                                                                                       self.probas_dict, config.watermark_kwargs, seed=config.seed)
        del subgraph_x_concat
        torch.cuda.empty_cache()

        for i, subgraph_sig in enumerate(self.subgraph_dict_not_wmk.keys()):
            self.subgraph_dict_not_wmk[subgraph_sig]['watermark']=watermarks[i]
        del watermarks
        # return subgraph_dict

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


    def get_weighted_losses(self, type_='primary', loss_primary=None, loss_watermark=None):##, reg=None):
        self.loss_dict['loss_primary']=loss_primary
        self.loss_dict['loss_watermark']=loss_watermark
        # self.loss_dict['reg']=reg
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
            # self.loss_dict['reg_weighted'] = reg 
        unweighted_total = torch_add_not_None([self.loss_dict[k] for k in ['loss_primary','loss_watermark']])#,'reg']])
        weighted_total = torch_add_not_None([self.loss_dict[k] for k in ['loss_primary_weighted','loss_watermark_weighted']])#,'reg_weighted']])
        return self.loss_dict, unweighted_total, weighted_total

    def get_watermark_performance_(self, is_last_epoch):
        if self.optimization_kwargs['penalize_similar_subgraphs']==True:
            for sig in self.subgraph_signatures:
                subgraph_node_indices = self.subgraph_dict[sig]['nodeIndices']
                shifted_subgraph, shifted_subgraph_node_indices = self.shift_subgraph_(self.optimization_kwargs['p_swap'], subgraph_node_indices)
                self.subgraph_dict[sig]['subgraph_shifted']=shifted_subgraph
                self.subgraph_dict[sig]['nodeIndices_shifted']=shifted_subgraph_node_indices
                del subgraph_node_indices, shifted_subgraph, shifted_subgraph_node_indices
        self.loss_watermark, self.beta_similarity, self.betas_from_every_subgraph, self.betas_dict, self.beta_similarities_dict, \
            self.watermark_percent_matches, self.match_count_wmk_with_zeros, self.match_count_wmk_without_zeros, \
                self.confidence_wmk_with_zeros,  self.confidence_wmk_without_zeros = get_watermark_performance(self.probas_dict, 
                                                                                                                self.subgraph_dict, 
                                                                                                                self.betas_dict, 
                                                                                                                self.beta_similarities_dict, 
                                                                                                                is_last_epoch,
                                                                                                                self.debug_multiple_subgraphs, 
                                                                                                                self.beta_weights,
                                                                                                                penalize_similar_subgraphs=self.optimization_kwargs['penalize_similar_subgraphs'],
                                                                                                                shifted_subgraph_loss_coef=self.optimization_kwargs['shifted_subgraph_loss_coef'])


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
                                                                                                                        penalize_similar_subgraphs=self.optimization_kwargs['penalize_similar_subgraphs'],
                                                                                                                        shifted_subgraph_loss_coef=self.optimization_kwargs['shifted_subgraph_loss_coef'])








def update_history_one_epoch(history, loss, loss_dict, acc_trn, acc_val, test_acc, watermark_percent_matches, 
                             match_count_with_zeros, confidence_with_zeros,
                             match_count_without_zeros, confidence_without_zeros, trigger_acc=None, acc_graphlime_backdoor=None):
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
        history['losses_primary_weighted'].append(loss_dict['loss_primary_weighted'].clone().detach())
    except:
        history['losses_primary_weighted'].append(loss_dict['loss_primary_weighted'])
    try:
        history['losses_watermark_weighted'].append(loss_dict['loss_watermark_weighted'].clone().detach())
    except:
        history['losses_watermark_weighted'].append(loss_dict['loss_watermark_weighted'])
    history['train_accs'].append(acc_trn)
    history['val_accs'].append(acc_val)
    history['test_accs'].append(test_acc)
    history['watermark_percent_matches'].append(watermark_percent_matches)
    history['match_counts_with_zeros'].append(match_count_with_zeros)
    history['match_counts_without_zeros'].append(match_count_without_zeros)
    history['match_count_confidence_with_zeros'].append(confidence_with_zeros)
    history['match_count_confidence_without_zeros'].append(confidence_without_zeros)
    if trigger_acc is not None:
        history['trigger_accs'].append(trigger_acc)
    if acc_graphlime_backdoor is not None:
        history['graphlime_backdoor_accs'].append(acc_graphlime_backdoor)
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


def get_performance_trends(history, subgraph_dict, optimization_kwargs):
    primary_loss_curve = history['losses_primary'] 
    primary_acc_curve = history['train_accs']
    train_acc = np.round(history['train_accs'][-1],3)
    val_acc = np.round(history['val_accs'][-1],3)
    test_acc = np.round(history['test_accs'][-1],3)

    if optimization_kwargs['clf_only']==True:
        return primary_loss_curve, None, None, None, None, None, None, primary_acc_curve, None, train_acc, val_acc, test_acc
    if config.watermark_random_backdoor==True:
        trigger_acc = np.round(history['trigger_accs'][-1],3)
        return primary_loss_curve, None, None, None, None, None, None, primary_acc_curve, None, train_acc, val_acc, test_acc, trigger_acc
    if config.watermark_graphlime_backdoor==True:
        graphlime_backdoor_acc = np.round(history['graphlime_backdoor_accs'][-1],3)
        return primary_loss_curve, None, None, None, None, None, None, primary_acc_curve, None, train_acc, val_acc, test_acc, graphlime_backdoor_acc

    else:
        match_counts_with_zeros = history['match_counts_with_zeros'][-1]
        match_counts_without_zeros = history['match_counts_without_zeros'][-1]
        match_count_confidence_with_zeros = np.round(history['match_count_confidence_with_zeros'][-1],3)
        match_count_confidence_without_zeros = np.round(history['match_count_confidence_without_zeros'][-1],3)
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

            del this_watermark, this_nonzero_indices, watermark_non_zero, this_final_beta, this_sign_beta, this_matches

        percent_match_mean, percent_match_std = np.round(np.mean(percent_matches),1), np.round(np.std(percent_matches),3)
        watermark_acc_curve =  history['watermark_percent_matches']
    
    return primary_loss_curve, watermark_loss_curve, final_betas, watermarks, percent_matches, \
        percent_match_mean, percent_match_std, primary_acc_curve, watermark_acc_curve, train_acc, \
            val_acc, test_acc, match_counts_with_zeros, match_counts_without_zeros,match_count_confidence_with_zeros,\
                match_count_confidence_without_zeros


               
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

        del sub, x_sub, edge_index_sub, log_logits, y_sub, this_raw_beta, this_sign_beta


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

        del x_sub, edge_index, log_logits, y_sub, beta

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
            del subgraph, log_logits_
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
    test_value_with_zeros = count_matches(bs, ignore_zerose=False)                
    z_score_with_zeros = (test_value_with_zeros - sample_mean_matches)/sample_std_matches
    p_value_with_zeros = 1 - stats.norm.cdf(z_score_with_zeros)
    test_value_without_zeros = count_matches(bs, ignore_zeros=True)                 
    z_score_without_zeros = (test_value_without_zeros - sample_mean_matches)/sample_std_matches
    p_value_without_zeros = 1 - stats.norm.cdf(z_score_without_zeros)
    if verbose:
        print(f'\nPopulation Mean, Standard Error: {np.round(sample_mean_matches,3)}, {np.round(sample_std_matches,3)}')
        print(f'# Matches among the {len(bs)} watermarked betas: {test_value_with_zeros} with zeros, {test_value_without_zeros} without zeros\n')
        print(f'(z_score, p_value): with zeros = ({np.round(z_score_with_zeros,3)}, {np.round(p_value_with_zeros,5)}), without zeros = ({np.round(z_score_without_zeros,3)}, {np.round(p_value_without_zeros,5)})')
    del bs
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
                             regression_kwargs, target_confidence=0.99, also_show_un_watermarked_counts=True):


    all_subgraph_indices = []
    for sig in subgraph_dict.keys():
        all_subgraph_indices += subgraph_dict[sig]['nodeIndices'].tolist()
    all_subgraph_indices = torch.tensor(all_subgraph_indices)
    

    sig_0 = list(subgraph_dict.keys())[0]
    # watermark = subgraph_dict[sig_0]['watermark']


    # node_classifier = Trainer_object.node_classifier
    node_classifier.eval()
    if config.preserve_edges_between_subsets==True:
        log_logits_eval = node_classifier(Trainer_object.unaugmented_x, Trainer_object.unaugmented_x, dropout=config.node_classifier_kwargs['dropout'])
        log_logits_train = log_logits_eval[Trainer_object.train_node_indices]
        log_logits_val = log_logits_eval[Trainer_object.val_node_indices]
        log_logits_test = log_logits_eval[Trainer_object.test_node_indices]
    elif config.preserve_edges_between_subsets==False:
        log_logits_train = node_classifier(Trainer_object.x_train_unaugmented, Trainer_object.edge_index_train_unaugmented, dropout=0)
        log_logits_val        = node_classifier(Trainer_object.x_val, Trainer_object.edge_index_val, dropout=0)
        log_logits_test        = node_classifier(Trainer_object.x_test, Trainer_object.edge_index_test, dropout=0)

    # log_logits_train = node_classifier(Trainer_object.x_train_unaugmented,Trainer_object.edge_index_train_unaugmented, 0)
    acc_train = accuracy(log_logits_train, Trainer_object.y_train_unaugmented)
    del log_logits_train

    # edge_index_test, _ = subgraph(trainer.test_mask, data.edge_index, relabel_nodes=True)
    # x_test, y_test = data.x[data.test_mask], data.y[data.test_mask]
    node_classifier.eval()
    log_logits_test = node_classifier(Trainer_object.x_test, Trainer_object.edge_index_test, 0)
    acc_test = accuracy(log_logits_test, Trainer_object.y_test)
    del log_logits_test

    # edge_index_val, _ = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)
    # x_val, y_val = data.x[data.val_mask], data.y[data.val_mask]
    node_classifier.eval()

    log_logits_val = node_classifier(Trainer_object.x_val, Trainer_object.edge_index_val, 0)
    acc_val = accuracy(log_logits_val, Trainer_object.y_val)
    del log_logits_val


    probas_dict  = separate_forward_passes_per_subgraph(subgraph_dict, node_classifier, mode='eval')
    beta_weights = torch.ones(len(subgraph_dict),data.num_features)     
    sign_betas, match_rates = [], []
    for s, sig in enumerate(subgraph_dict.keys()):
        beta_weights_ = beta_weights[s]
        _, this_percent_match, _, this_raw_beta = get_watermark_performance_single_subgraph(probas_dict,subgraph_dict, sig,ignore_zeros_from_subgraphs=False, debug=False,
                                                                                            beta_weights=beta_weights_,similar_subgraph=False, watermark_loss_kwargs=watermark_loss_kwargs,
                                                                                            regression_kwargs=regression_kwargs, #optimization_kwargs=optimization_kwargs, 
                                                                                            this_watermark=None)
        sign_betas.append(torch.sign(this_raw_beta))
        match_rates.append(this_percent_match)
    del probas_dict, this_raw_beta, this_percent_match

    stacked_sign_betas = torch.vstack(sign_betas)
    match_count_wmk_with_zeros = count_matches(stacked_sign_betas, ignore_zeros=False)
    match_count_wmk_without_zeros = count_matches(stacked_sign_betas, ignore_zeros=True)
    del stacked_sign_betas


    z_t = norm.ppf(target_confidence)
    target_number_matches = np.ceil(min(mu_natural +z_t*sigma_natural,data.x.shape[1]))
    # print(f'Target # matches for {(100*target_confidence):3f}% confidence:',target_number_matches)
    confidence_wmk_with_zeros = get_confidence_matches(match_count_wmk_with_zeros, mu_natural, sigma_natural)
    confidence_wmk_without_zeros = get_confidence_matches(match_count_wmk_without_zeros, mu_natural, sigma_natural)


    if also_show_un_watermarked_counts==True:
        probas_dict_not_wmk  = separate_forward_passes_per_subgraph(subgraph_dict_not_wmk, node_classifier, mode='eval')
        sign_betas_not_wmk = []
        match_rates_not_wmk= []
        for s, sig in enumerate(subgraph_dict_not_wmk.keys()):
            beta_weights_ = beta_weights[s]
            _, this_percent_match_not_wmk, _, this_raw_beta_not_wmk = get_watermark_performance_single_subgraph(probas_dict_not_wmk,subgraph_dict_not_wmk, sig,ignore_zeros_from_subgraphs=False, 
                                                                                                                    debug=False,beta_weights=beta_weights_,similar_subgraph=False,
                                                                                                                    watermark_loss_kwargs=watermark_loss_kwargs,
                                                                                                                    regression_kwargs=regression_kwargs,
                                                                                                                    #optimization_kwargs=optimization_kwargs,
                                                                                                                    this_watermark=subgraph_dict[sig_0]['watermark'])
            sign_betas_not_wmk.append(torch.sign(this_raw_beta_not_wmk))
            match_rates_not_wmk.append(this_percent_match_not_wmk)
        del probas_dict_not_wmk, this_raw_beta_not_wmk, this_percent_match_not_wmk

        stacked_sign_betas_not_wmk = torch.vstack(sign_betas_not_wmk)
        match_count_not_wmk_with_zeros = count_matches(stacked_sign_betas_not_wmk, ignore_zeros=False)
        match_count_not_wmk_without_zeros = count_matches(stacked_sign_betas_not_wmk, ignore_zeros=True)
        confidence_not_wmk_with_zeros = get_confidence_matches(match_count_not_wmk_with_zeros, mu_natural, sigma_natural)
        confidence_not_wmk_without_zeros = get_confidence_matches(match_count_not_wmk_without_zeros, mu_natural, sigma_natural)
        del stacked_sign_betas_not_wmk
    
    else:
        match_count_not_wmk_with_zeros, match_count_not_wmk_without_zeros, confidence_not_wmk_with_zeros, confidence_not_wmk_without_zeros = None, None, None, None

    return match_rates, acc_train, acc_test, acc_val, target_number_matches, \
        [match_count_wmk_with_zeros, match_count_wmk_without_zeros, confidence_wmk_with_zeros, confidence_wmk_without_zeros], \
        [match_count_not_wmk_with_zeros, match_count_not_wmk_without_zeros, confidence_not_wmk_with_zeros, confidence_not_wmk_without_zeros]



def get_node_classifier_and_optimizer_and_subgraph_dict_for_further_processing(model_path, lr):
    subgraph_dict   = pickle.load(open(os.path.join(model_path,'subgraph_dict'),'rb'))
    node_classifier = pickle.load(open(os.path.join(model_path,'node_classifier'),'rb'))
    params_         = list(node_classifier.parameters())
    optimizer       = optim.Adam(params_, lr=lr)
    del params_
    return node_classifier, optimizer, subgraph_dict






####


#####

class Trainer_KD():
    def __init__(self, student_model, teacher_model, data, dataset_name, subgraph_dict, subgraph_dict_not_wmk, alpha, temperature, train_node_indices, test_node_indices, val_node_indices):
        # self.temperature = 5#model_kwargs.get('temperature', 5.0)  # Temperature for KD
        # self.alpha = 0.5#model_kwargs.get('alpha', 0.5)  # Weight for distillation loss
        self.temperature = temperature
        self.alpha = alpha
        self.data = data
        self.dataset_name = dataset_name
        self.num_features = data.x.shape[1]
        self.train_mask = data.train_mask
        self.val_mask = data.val_mask
        self.test_mask = data.test_mask
        self.optimization_kwargs = config.KD_student_optimization_kwargs
        self.use_pcgrad        = self.optimization_kwargs['use_pcgrad']
        self.lr                = self.optimization_kwargs['lr']
        self.epochs            = self.optimization_kwargs['epochs']
        # self.node_aug, self.edge_aug = collect_augmentations()
        self.loss_dict = setup_loss_dict()
        self.node_classifier=student_model
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.instantiate_optimizer()
        self.best_train_acc, self.best_val_acc = 0, 0
        validate_kwargs()

        self.loss = torch.tensor(0.0)
        # self.loss_primary = torch.tensor(0.0)
        # self.loss_primary_weighted = torch.tensor(0.0)
        self.subgraph_dict = subgraph_dict
        self.subgraph_dict_not_wmk = subgraph_dict_not_wmk
        self.num_features = data.x.shape[1]
        self.each_subgraph_feature_importances=None
        self.each_subgraph_watermark_indices=None
        self.subgraph_signatures = list(subgraph_dict.keys())
        all_subgraph_indices = []
        for sig in subgraph_dict.keys():
            nodeIndices = subgraph_dict[sig]['nodeIndices'].tolist()
            all_subgraph_indices.append(nodeIndices)
        self.all_subgraph_indices = torch.tensor(all_subgraph_indices)
        self.subgraph_signatures_not_wmk = list(subgraph_dict_not_wmk.keys())
        self.history, self.betas_dict, self.beta_similarities_dict              = setup_history(subgraph_signatures=self.subgraph_signatures)
        self.beta_weights                                                       = get_beta_weights(self.subgraph_dict, self.num_features)
        self.history_not_wmk, self.betas_dict_not_wmk, self.beta_similarities_dict_not_wmk              = setup_history(subgraph_signatures=self.subgraph_signatures_not_wmk)
        self.beta_weights_not_wmk                                                                       = get_beta_weights(self.subgraph_dict_not_wmk, self.num_features)
        self.train_node_indices = train_node_indices
        if config.kd_subgraphs_only==True:
            config.kd_train_on_subgraphs=True
        if config.kd_train_on_subgraphs==True:
            self.separate_forward_passes_per_subgraph_=True
            if config.kd_subgraphs_only==True:
                self.train_nodes_to_consider = self.all_subgraph_indices.reshape(-1)
                self.train_node_indices =self.train_nodes_to_consider
            else:
                self.train_nodes_to_consider_mask = get_train_nodes_to_consider(self.data, self.all_subgraph_indices, 'subgraph_node_indices', self.data.x.shape[0], train_with_test_set=False)
                self.train_node_indices = self.train_nodes_to_consider_mask.nonzero(as_tuple=True)[0]
                self.train_nodes_to_consider = torch.where(self.train_nodes_to_consider_mask==True)[0]
        else:
            self.separate_forward_passes_per_subgraph_=False
            self.train_nodes_to_consider = train_node_indices
        print('self.train_nodes_to_consider:',self.train_nodes_to_consider)
        

        # self.train_nodes_to_consider_mask = get_train_nodes_to_consider(self.data, self.all_subgraph_indices, self.optimization_kwargs['sacrifice_kwargs']['method'], self.data.x.shape[0], False, False)
        # self.train_node_indices = self.train_nodes_to_consider_mask.nonzero(as_tuple=True)[0]
        ### for evaluating on train set -- not just train mask, but (train_mask - subgraph_indices)
        
        print('self.train_node_indices:',sum(self.train_node_indices),self.train_node_indices)

        if config.preserve_edges_between_subsets==False:
            self.edge_index_train, _ = subgraph(self.train_node_indices, self.data.edge_index, relabel_nodes=True)
            self.edge_index_train = self.edge_index_train.clone()
        self.x_train = self.data.x[self.train_node_indices].clone()
        self.y_train = self.data.y[self.train_node_indices].clone()
        ### for evaluating on test set
        # test_node_indices = self.data.test_mask.nonzero(as_tuple=True)[0]
        self.test_node_indices = test_node_indices
        if config.preserve_edges_between_subsets==False:
            self.edge_index_test, _ = subgraph(self.test_node_indices, self.data.edge_index, relabel_nodes=True)
            self.edge_index_test = self.edge_index_test.clone()
        self.x_test = self.data.x[self.test_node_indices].clone()
        self.y_test = self.data.y[self.test_node_indices].clone()
        
        ### for evaluating on validation set
        # val_node_indices = self.data.val_mask.nonzero(as_tuple=True)[0]
        self.val_node_indices = val_node_indices
        if config.preserve_edges_between_subsets==False:
            self.edge_index_val, _ = subgraph(val_node_indices, self.data.edge_index, relabel_nodes=True)
            self.edge_index_val = self.edge_index_val.clone()
        self.x_val = self.data.x[self.val_node_indices].clone()
        self.y_val = self.data.y[self.val_node_indices].clone()
        return

    def save_process(self, continuation=False,starting_epoch=None, file_ext=None, verbose=False):
        if file_ext is None or file_ext == '':
            file_ext=''
        else:
            file_ext = "_"+file_ext
        if config.preserve_edges_between_subsets==True:
            file_ext = '_preserve_edges' + file_ext
        if continuation==True:
            Trainer_str = f'Trainer{file_ext}_continuation_from_{starting_epoch}'
        else:
            Trainer_str = f'Trainer{file_ext}'
        with open(os.path.join(get_results_folder_name(self.dataset_name),Trainer_str), 'wb') as f:
            pickle.dump(self, f)
        save_results(self.dataset_name, self.node_classifier, self.history, self.subgraph_dict, 
                        self.each_subgraph_feature_importances, self.each_subgraph_watermark_indices, 
                        verbose=verbose,continuation=continuation, starting_epoch=starting_epoch)

    def instantiate_optimizer(self):
        if self.optimization_kwargs['use_sam']==True:
            optimizer = SAM(self.node_classifier.parameters(), optim.SGD, lr=self.lr, momentum=self.optimization_kwargs['sam_momentum'],rho = self.optimization_kwargs['sam_rho'])
        else:
            optimizer = optim.Adam(self.node_classifier.parameters(), lr=self.lr)
        if self.use_pcgrad==True:
            optimizer = PCGrad(optimizer)
        self.optimizer = optimizer


    def train_KD(self, save=True, print_every=10, continuation=False, starting_epoch=0):
        for epoch in tqdm(range(self.epochs)):
            extra_print=''
            epoch += starting_epoch
            self.epoch=epoch
            closure = self.closure_KD
            if config.KD_student_optimization_kwargs['use_sam']==True:
                self.optimizer.step(closure)
            else:
                closure()
                self.optimizer.step()
            self.history = update_history_one_epoch(self.history, self.loss, self.loss_dict, self.acc_trn, self.acc_val, self.acc_test, None, None, None,None, None)
            if epoch%1==0:
                watermarked_subgraph_results, unwatermarked_subgraph_results = self.test_watermark()
                match_count_without_zeros_wmk, _, confidence_without_zeros_wmk = watermarked_subgraph_results[7:10]
                match_count_without_zeros_un_wmk, _, confidence_without_zeros_un_wmk = unwatermarked_subgraph_results[7:10]
                extra_print = f'wmk/unwmk counts (without zeros): {match_count_without_zeros_wmk}/{match_count_without_zeros_un_wmk}, wmk/unwmk confidence: {confidence_without_zeros_wmk:.3f}/{confidence_without_zeros_un_wmk:.3f}'

            if self.epoch%print_every==0:
                print_epoch_status(self.epoch, self.classification_loss, self.acc_trn, self.acc_val, self.acc_test, False, None, None, None,None, None,None, None,None, None,True,self.acc_trn_KD, self.acc_val_KD, self.acc_test_KD, self.distillation_loss_, additional_content=extra_print)
            gc.collect()
            torch.cuda.empty_cache() 
            if save==True:
                self.save_process(continuation,starting_epoch,'KD',verbose=epoch==self.epochs-1)
        self.history = replace_history_Nones(self.history)
        if save==True:
            self.save_process(continuation,starting_epoch,'KD')            
        gc.collect()
        return self.node_classifier, self.history
    
    def distillation_loss(self, student_logits, teacher_logits):
        student_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        return F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)

    def closure_KD(self):
        self.optimizer.zero_grad()
        self.teacher_model.eval()

        self.distillation_loss_=torch.tensor(0,dtype=torch.float)
        self.classification_loss=torch.tensor(0,dtype=torch.float)
        ##

        if config.preserve_edges_between_subsets==True:
            if config.kd_subgraphs_only==False:
                student_logits = self.forward(self.data.x, self.data.edge_index, dropout=config.KD_student_node_classifier_kwargs['dropout'], mode='train')
                student_logits_train = student_logits[self.train_node_indices]
            student_logits_eval  = self.forward(self.data.x, self.data.edge_index, dropout=config.KD_student_node_classifier_kwargs['dropout'], mode='eval')
            student_logits_train_eval = student_logits_eval[self.train_node_indices]
            student_logits_val        = student_logits_eval[self.val_node_indices]
            student_logits_test       = student_logits_eval[self.test_node_indices]

            self.teacher_model.eval()
            teacher_logits = self.teacher_model(self.data.x, self.data.edge_index)
            teacher_logits_train = teacher_logits[self.train_node_indices]
            teacher_logits_val   = teacher_logits[self.val_node_indices]
            teacher_logits_test  = teacher_logits[self.test_node_indices]


        elif config.preserve_edges_between_subsets==False:
            if config.kd_subgraphs_only==False:
                student_logits_train = self.forward(self.x_train, self.edge_index_train, dropout=config.KD_student_node_classifier_kwargs['dropout'], mode='train')
            self.teacher_model.eval()
            teacher_logits_train = self.teacher_model(self.x_train, self.edge_index_train)
            teacher_logits_val   = self.teacher_model(self.x_val, self.edge_index_val)
            teacher_logits_test  = self.teacher_model(self.x_test, self.edge_index_test)
            student_logits_train_eval = self.forward(self.x_train, self.edge_index_train, dropout=0, mode='eval')
            student_logits_val        = self.forward(self.x_val, self.edge_index_val, dropout=0, mode='eval')
            student_logits_test       = self.forward(self.x_test, self.edge_index_test, dropout=0, mode='eval')


        if config.kd_subgraphs_only==False:
            self.distillation_loss_   = self.distillation_loss(student_logits_train, teacher_logits_train)
            self.classification_loss = F.nll_loss(student_logits_train, self.y_train)
        
        if config.kd_train_on_subgraphs==True:
            probas_dict_student = {}
            probas_dict_teacher = {}
            for sig in self.subgraph_dict.keys():
                subgraph = self.subgraph_dict[sig]['subgraph']
                student_logits = self.forward(subgraph.x, subgraph.edge_index, dropout=config.node_classifier_kwargs['dropout_subgraphs'])
                probas_dict_student[sig]= student_logits.clone().exp()
                self.teacher_model.eval()
                teacher_logits = self.teacher_model(subgraph.x, subgraph.edge_index, dropout=config.node_classifier_kwargs['dropout_subgraphs'])
                probas_dict_teacher[sig]= teacher_logits.clone().exp()
                self.distillation_loss_ += self.distillation_loss(student_logits, teacher_logits)
                self.classification_loss += F.nll_loss(student_logits, subgraph.y)
                del subgraph, student_logits, teacher_logits
            self.loss = self.alpha * self.distillation_loss_ + (1 - self.alpha) * self.classification_loss

        self.loss = self.alpha * self.distillation_loss_ + (1 - self.alpha) * self.classification_loss

        self.acc_trn_KD  = accuracy(student_logits_train_eval, teacher_logits_train.argmax(dim=1),verbose=False)
        self.acc_val_KD  = accuracy(student_logits_val, teacher_logits_val.argmax(dim=1),verbose=False)
        self.acc_test_KD  = accuracy(student_logits_test, teacher_logits_test.argmax(dim=1),verbose=False)
        self.acc_trn  = accuracy(student_logits_train_eval, self.y_train,verbose=False)
        self.acc_val  = accuracy(student_logits_val, self.y_val,verbose=False)
        self.acc_test = accuracy(student_logits_test, self.y_test,verbose=False)
        del teacher_logits_train, teacher_logits_val, teacher_logits_test
        del student_logits_train_eval, student_logits_val, student_logits_test
        try:
            del tudent_logits_train
        except:
            pass
        self.backward([self.loss], verbose=False, retain_graph=False)
        return self.loss

    def forward(self, x, edge_index, dropout, mode='train'):
        assert mode in ['train','eval']
        if mode=='train':
            self.node_classifier.train()
            log_logits = self.node_classifier(x, edge_index, dropout)
        elif mode=='eval':
            self.node_classifier.eval()
            log_logits = self.node_classifier(x, edge_index, dropout)
        return log_logits
    
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


    def get_weighted_losses(self, type_='primary', loss_primary=None, loss_watermark=None):
        self.loss_dict['loss_primary']=loss_primary
        self.loss_dict['loss_watermark']=loss_watermark
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
        unweighted_total = torch_add_not_None([self.loss_dict[k] for k in ['loss_primary','loss_watermark']])
        weighted_total = torch_add_not_None([self.loss_dict[k] for k in ['loss_primary_weighted','loss_watermark_weighted']])
        return self.loss_dict, unweighted_total, weighted_total



    def apply_watermark_(self):
        watermark_type = config.watermark_kwargs['watermark_type']
        len_watermark = int(config.watermark_kwargs['percent_of_features_to_watermark']*self.num_features/100)
        subgraph_x_concat = torch.concat([self.subgraph_dict[k]['subgraph'].x for k in self.subgraph_dict.keys()])
        self.subgraph_dict, self.each_subgraph_watermark_indices, self.each_subgraph_feature_importances, watermarks = apply_watermark(watermark_type, self.num_features, len_watermark, self.subgraph_dict, subgraph_x_concat,
                                                                                                                                       self.probas_dict, config.watermark_kwargs, seed=config.seed)
        del subgraph_x_concat
        torch.cuda.empty_cache()
        for i, subgraph_sig in enumerate(self.subgraph_dict_not_wmk.keys()):
            self.subgraph_dict_not_wmk[subgraph_sig]['watermark']=watermarks[i]
        del watermarks
        # return subgraph_dict

    def test_watermark(self):    
        is_last_epoch = self.epoch==self.epochs-1
        beta_weights = get_beta_weights(self.subgraph_dict, self.num_features)
        self.probas_dict = separate_forward_passes_per_subgraph(self.subgraph_dict, self.node_classifier, 'eval')
        self.probas_dict_not_wmk = separate_forward_passes_per_subgraph(self.subgraph_dict_not_wmk, self.node_classifier, mode='eval')
        self.apply_watermark_()
        watermarked_subgraph_results = get_watermark_performance(self.probas_dict, 
                                                                 self.subgraph_dict, 
                                                                 {k:[] for k in self.subgraph_dict.keys()}, 
                                                                 {k:None for k in self.subgraph_dict.keys()}, 
                                                                 is_last_epoch,
                                                                 False, beta_weights, penalize_similar_subgraphs=False, shifted_subgraph_loss_coef=0)


        beta_weights = get_beta_weights(self.subgraph_dict_not_wmk, self.num_features)
        unwatermarked_subgraph_results = get_watermark_performance(self.probas_dict_not_wmk, 
                                                                   self.subgraph_dict_not_wmk, 
                                                                   {k:[] for k in self.subgraph_dict_not_wmk.keys()}, 
                                                                   {k:None for k in self.subgraph_dict_not_wmk.keys()}, 
                                                                   is_last_epoch,
                                                                   False, beta_weights, penalize_similar_subgraphs=False, shifted_subgraph_loss_coef=0)
        return watermarked_subgraph_results, unwatermarked_subgraph_results


def get_ranked_features_GraphLIME(dataset_name, data, node_indices):
    config.optimization_kwargs['clf_only']=True
    results_folder = get_results_folder_name(dataset_name)
    clf_only_node_classifier_path = os.path.join(results_folder, 'node_classifier')
    clf_only_node_classifier = pickle.load(open(clf_only_node_classifier_path,'rb'))
    # ranked from least important to most important
    feature_importance_dict = {node_idx.item():None for node_idx in node_indices}
    for i, node_idx in enumerate(node_indices):
        print(f'assessing node {i}/{len(node_indices)}')#,end='\r')
        explainer = GraphLIME(clf_only_node_classifier, hop=1, rho=0.1)
        coefs = explainer.explain_node(node_idx.item(), data.x, data.edge_index)
        abs_coefficients = np.abs(coefs)
        least_representative_indices = np.argsort(abs_coefficients)
        feature_importance_dict[node_idx.item()]=least_representative_indices
    config.optimization_kwargs['clf_only']=False
    return feature_importance_dict

def backdoor_GraphLIME(dataset_name, dataset, training_indices, attack_target_label,poison_rate=0.1, watermark_size=0.2, seed=0):
    model_folder_config_name_seed_version_name = get_results_folder_name(dataset_name)
    backdoor_items_path = os.path.join(model_folder_config_name_seed_version_name,'graphlime_backdoor_items')
    num_nodes_to_poison = int(poison_rate*len(dataset.x))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random_indices = torch.randperm(len(training_indices))[:num_nodes_to_poison]
    poisoned_node_indices = training_indices[random_indices]
    num_node_features = dataset.x.shape[1]
    num_backdoor_features = int(watermark_size*num_node_features)
    watermark = torch.randint(0, 2, (num_backdoor_features,),dtype=torch.float)

    assert torch.max(poisoned_node_indices)<=torch.max(training_indices)
    if poison_rate==1:
        assert len(poisoned_node_indices) == len(dataset.x)
    print('backdoor_items_path:',backdoor_items_path)
    if os.path.exists(backdoor_items_path)==False or config.watermark_random_backdoor_re_explain==True:
        if config.watermark_random_backdoor_re_explain==False:
            print('GraphLIME explanations haven\'t been generated yet. Doing that now...')
        elif config.watermark_random_backdoor_re_explain==True:
            print('Generating GraphLIME explanations...')
        ranked_feature_dict = get_ranked_features_GraphLIME(dataset_name, dataset, poisoned_node_indices)
        backdoor_info = {'watermark':watermark, 'ranked_feature_dict':ranked_feature_dict}
        for idx in poisoned_node_indices:
            idx = idx.item()
            feature_indices_ranked = ranked_feature_dict[idx]
            backdoor_feature_indices = feature_indices_ranked[:num_backdoor_features]
            dataset.x[idx][backdoor_feature_indices]=watermark
            dataset.y[idx]=attack_target_label
        pickle.dump([dataset, backdoor_info],open(backdoor_items_path,'wb'))
        
    else:
        [dataset, backdoor_info] = pickle.load(open(backdoor_items_path,'rb'))
        ranked_feature_dict = backdoor_info['ranked_feature_dict']
        assert set(poisoned_node_indices.tolist())==set(backdoor_info['ranked_feature_dict'].keys())
        assert torch.all(watermark==backdoor_info['watermark'])
    return dataset, backdoor_info



def refine_K_and_P(trigger_size, graph_type='ER', K=0, P=0):
    # Chooses automatically if K==0 or P==0, otherwise validates user choice
    '''
    For an Erdos-Renyi (ER) graph generation, P is the probability of adding an edge between any two nodes. If 0, will automatically reset to 1.
    For Small-World (SW) graph generation, P is the probability of rewiring each edge. If 0, will automaticall reset to 1.
    For Small-World (SW) graph generation, K is the number of neighbors in initial ring lattice. If 0, will automatically compute default value as a function of trigger size.
    For Preferential-Attachment (PA) graph generation, K is the number of edges to attach from a new node to existing nodes. If 0, will automatically compute default value as a function of trigger size.
    '''
    def validate_K(graph_type, K, trigger_size):
        if graph_type=='ER':
            return True # K doesn't apply for ER graphs
        elif graph_type == 'PA' and K > trigger_size:
            print('Invalid K: for PA graphs, K must be less than or equal to trigger size.'); return False
        elif graph_type == 'SW' and K<2:
            print('Invalid K: for SW graphs, K must be greater than 2.'); return False
        else:
            return True
    if graph_type=='SW' or graph_type=='PA':
        assert K is not None
        if K==0:
            K=trigger_size=1
            assert validate_K(graph_type, K, trigger_size)
    elif graph_type=='SW' or graph_type=='ER':
        assert P is not None
        if P==0:
            P=1
    return K,P



def inject_backdoor_trigger_subgraph(dgl_graph, trigger_graph=None):
    ''' Inserts trigger at random nodes in graph. '''
    num_trigger_nodes = len(trigger_graph.nodes())
    if num_trigger_nodes > len(dgl_graph.nodes()):
        raise ValueError("Number of nodes to select is greater than the number of nodes in the graph.")
    rand_select_nodes=[]
    remaining_nodes = list(dgl_graph.nodes())
    for _ in range(num_trigger_nodes):
        if rand_select_nodes:
            no_edge_nodes = [n for n in remaining_nodes if all(not dgl_graph.has_edges_between(n, m) and not dgl_graph.has_edges_between(m, n) for m in rand_select_nodes)]
            if no_edge_nodes:
                new_node = random.choice(no_edge_nodes)
            else:
                new_node = random.choice(remaining_nodes)
        else:
            new_node = random.choice(remaining_nodes)
        rand_select_nodes.append(new_node.item())
        remaining_nodes.remove(new_node.item())

    node_mapping = {trigger_node: main_node for trigger_node, main_node in zip(trigger_graph.nodes(), rand_select_nodes)}
    edges_start = [[t0.item(), t1.item()] for [t0,t1] in zip(*dgl_graph.edges())]
    edges_final = copy.deepcopy(edges_start)
    ''' Remove any existing connections between selected trigger nodes. '''
    for n0 in rand_select_nodes:
        for n1 in rand_select_nodes:
            if [n0, n1] in edges_final:
                edge_id = dgl_graph.edge_ids(torch.tensor([n0]), torch.tensor([n1]))
                dgl_graph.remove_edges(edge_id)
                edges_final.remove([n0, n1])

    ''' Add edges specified by trigger graph. '''
    trigger_edges = []
    for e in trigger_graph.edges():
        edge = [node_mapping[e[0]], node_mapping[e[1]]]
        (n0, n1) = (min(edge), max(edge))
        trigger_edges.append([n0,n1])
        edges_final.append([n0, n1])
        dgl_graph.add_edges(torch.tensor([n0]), torch.tensor([n1]))
    edges_final = [[t0.item(), t1.item()] for [t0,t1] in zip(*dgl_graph.edges())]
    return dgl_graph, trigger_edges

def inject_backdoor_trigger_features(dgl_graph, trigger_graph=None):
    ''' Inserts trigger at random nodes in graph. '''
    num_trigger_nodes = len(trigger_graph.nodes())
    if num_trigger_nodes > len(dgl_graph.nodes()):
        raise ValueError("Number of nodes to select is greater than the number of nodes in the graph.")
    rand_select_nodes=[]
    remaining_nodes = list(dgl_graph.nodes())
    for _ in range(num_trigger_nodes):
        if rand_select_nodes:
            no_edge_nodes = [n for n in remaining_nodes if all(not dgl_graph.has_edges_between(n, m) and not dgl_graph.has_edges_between(m, n) for m in rand_select_nodes)]
            if no_edge_nodes:
                new_node = random.choice(no_edge_nodes)
            else:
                new_node = random.choice(remaining_nodes)
        else:
            new_node = random.choice(remaining_nodes)
        rand_select_nodes.append(new_node.item())
        remaining_nodes.remove(new_node.item())

    node_mapping = {trigger_node: main_node for trigger_node, main_node in zip(trigger_graph.nodes(), rand_select_nodes)}
    edges_start = [[t0.item(), t1.item()] for [t0,t1] in zip(*dgl_graph.edges())]
    edges_final = copy.deepcopy(edges_start)
    ''' Remove any existing connections between selected trigger nodes. '''
    for n0 in rand_select_nodes:
        for n1 in rand_select_nodes:
            if [n0, n1] in edges_final:
                edge_id = dgl_graph.edge_ids(torch.tensor([n0]), torch.tensor([n1]))
                dgl_graph.remove_edges(edge_id)
                edges_final.remove([n0, n1])

    ''' Add edges specified by trigger graph. '''
    trigger_edges = []
    for e in trigger_graph.edges():
        edge = [node_mapping[e[0]], node_mapping[e[1]]]
        (n0, n1) = (min(edge), max(edge))
        trigger_edges.append([n0,n1])
        edges_final.append([n0, n1])
        dgl_graph.add_edges(torch.tensor([n0]), torch.tensor([n1]))
    edges_final = [[t0.item(), t1.item()] for [t0,t1] in zip(*dgl_graph.edges())]
    return dgl_graph, trigger_edges


def create_dataset_from_files(root, dataset_name, sample_size=None):
    '''For graph classification task (MUTAG)'''
    # Load raw files
    edges = pd.read_csv(os.path.join(root, f"{dataset_name}_A.txt"), header=None, sep=",")
    graph_indicator = pd.read_csv(os.path.join(root, f"{dataset_name}_graph_indicator.txt"), header=None)
    graph_labels = pd.read_csv(os.path.join(root, f"{dataset_name}_graph_labels.txt"), header=None)
    node_labels = pd.read_csv(os.path.join(root, f"{dataset_name}_node_labels.txt"), header=None)

    data_list = []
    N = graph_labels.shape[0]  # Number of graphs in the dataset
    indices_to_use = range(1, N + 1)  # Graph indices start from 1

    if sample_size is not None:
        print(f"Taking random sample of size {sample_size}")
        indices_to_use = random.sample(range(1, N + 1), sample_size)

    # Process each graph
    for c, i in enumerate(indices_to_use):
        print(f"Processing graph {c + 1}/{len(indices_to_use)}", end="\r")
        
        # Find nodes belonging to the current graph
        node_indices = graph_indicator[graph_indicator[0] == i].index

        # Map global node indices to local indices
        node_idx_map = {idx: j for j, idx in enumerate(node_indices)}
        graph_edges = edges[edges[0].isin(node_indices + 1) & edges[1].isin(node_indices + 1)]
        # graph_edges = graph_edges.applymap(lambda x: node_idx_map[x - 1])
        graph_edges = graph_edges.apply(lambda col: col.map(lambda x: node_idx_map[x - 1]))
        edge_index = torch.tensor(graph_edges.values, dtype=torch.long).t().contiguous()

        # Node features from node labels
        x = torch.tensor(node_labels.iloc[node_indices].values, dtype=torch.float)

        # Graph label (target)
        y = torch.tensor(graph_labels.iloc[i - 1].values, dtype=torch.long)

        # Create a Data object for the current graph
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)

    return data_list