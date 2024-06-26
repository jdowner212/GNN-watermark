import copy
import grafog.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import numpy as np 
import os
import pickle
import random
from   sklearn.model_selection import train_test_split
from   tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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
        transform_list = [NormalizeFeatures()]
        if dataset_name in ['CORA','CiteSeer','PubMed','Flickr']:
            transform_list.append(ChooseLargestMaskForTrain())
        if dataset_name in ['computers', 'photo']:
            transform_list.append(CreateMaskTransform(*train_val_test_split))
        if dataset_name in ['Reddit','Reddit2']:
            transform_list.append(KHopsFractionDatasetTransform(1,3))
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
        print('correct:',correct,'output:',len(output),'labels:',len(labels))
    correct = copy.deepcopy(correct.sum()) # i don't think the copy.deepcopy is necessary
    return correct / len(labels)

def sacrifice_node_node_indices(train_node_indices,method,p_sacrifice,subgraph_node_indices=None):
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


def augment_data(data, node_aug, edge_aug):
    edge_index, x, y = data.edge_index, data.x, data.y
    new_data = copy.deepcopy(data)
    if node_aug is not None:
        new_data = node_aug(new_data)
        edge_index, x, y = new_data.edge_index, new_data.x, new_data.y
    if edge_aug is not None:
        new_data = edge_aug(new_data)
        edge_index, x, y = new_data.edge_index, new_data.x, new_data.y
    x.requires_grad_(True)
    return edge_index, x, y


def initialize_training():
    lr = config.optimization_kwargs['lr']
    node_classifier_kwargs = config.node_classifier_kwargs
    node_classifier = Net(**node_classifier_kwargs)
    optimizer = optim.Adam(node_classifier.parameters(), lr=lr)
    node_classifier.train()
    return node_classifier, optimizer

def setup_history(subgraph_signatures):
    history = {
        'losses': [], 'losses_primary': [], 'losses_watermark': [], 'betas': [],
        'beta_similarities': [], 'train_accs': [], 'val_accs': []
    }
    betas_dict = {sig: [] for sig in subgraph_signatures}
    beta_similarities_dict = {sig: None for sig in subgraph_signatures}
    return history, betas_dict, beta_similarities_dict

def setup_subgraph_dict(data, dataset_name):#, subgraph_kwargs, watermark_kwargs):
    subgraph_kwargs = config.subgraph_kwargs
    watermark_kwargs = config.watermark_kwargs
    subgraph_dict, all_subgraph_indices = collect_subgraphs_within_single_graph_for_watermarking(data, dataset_name, use_train_mask=True, subgraph_kwargs=subgraph_kwargs)
    subgraph_signatures = list(subgraph_dict.keys())
    if watermark_kwargs['watermark_type']=='basic':
        subgraph_dict = apply_basic_watermark(data, subgraph_dict, watermark_kwargs)
    return subgraph_dict, subgraph_signatures, all_subgraph_indices

def print_epoch_status(epoch, loss_primary, loss_watermark, beta_similarity, acc_trn, acc_val, condition_met,beta_std_dev_penalty):
    if condition_met:
        epoch_printout = f'Epoch: {epoch:3d}, loss_primary = {loss_primary:.3f}, loss_watermark = {loss_watermark:.3f}, B*W = {beta_similarity:.5f}, train acc = {acc_trn:.3f}, val acc = {acc_val:.3f}, beta_std_dev: {beta_std_dev_penalty:.3f}'
    else:
        epoch_printout = f'Epoch: {epoch:3d}, loss_primary = {loss_primary:.3f}, loss_watermark = n/a, B*W = n/a, train acc = {acc_trn:.3f}, val acc = {acc_val:.3f}'
    print(epoch_printout)
 

def optimize_watermark(loss_watermark, beta_similarity, x_sub, y_sub, this_watermark, betas_dict, beta_similarities_dict, sig, 
                       epoch_condition, ignore_zeros_from_subgraphs=False, 
                       debug=False,
                       watermark_loss_kwargs={},
                       regression_kwargs={},
                       balanced_beta_weights=None):
    

    ''' epoch condtion: epoch==epoch-1'''
    omit_indices,not_omit_indices = get_omit_indices(x_sub, this_watermark,ignore_zeros_from_subgraphs=ignore_zeros_from_subgraphs) #indices where watermark is 0
    
    raw_beta            = solve_regression(x_sub, y_sub, regression_kwargs['lambda'])
    beta                = process_beta(raw_beta, watermark_loss_kwargs['alpha'], omit_indices, watermark_loss_kwargs['scale_beta_method'])

    B_x_W = (beta*this_watermark).clone()
    B_x_W = B_x_W[not_omit_indices]
    balanced_beta_weights = balanced_beta_weights[not_omit_indices]

    this_loss_watermark = torch.mean(torch.clamp(watermark_loss_kwargs['epsilon']-B_x_W, min=0)*balanced_beta_weights)
    this_beta_similarity = torch.mean(B_x_W)
    loss_watermark  += this_loss_watermark
    beta_similarity += this_beta_similarity
    if debug:
        print(f"Subgraph: Loss Watermark: {this_loss_watermark.item()}, Beta Similarity: {this_beta_similarity.item()}")
    
    
    if epoch_condition:
        beta_similarities_dict[sig] = this_beta_similarity
    betas_dict[sig].append(raw_beta.clone().detach())
    return loss_watermark, beta_similarity, betas_dict, beta_similarities_dict

def get_reg_term(betas_from_every_subgraph, regularization_type, lambda_l2):
    if regularization_type==None:
        return 0
    else:
        if regularization_type=='L2':
            reg = sum(torch.norm(betas_from_every_subgraph[i]) for i in range(len(betas_from_every_subgraph)))
            reg *= lambda_l2
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



def train(data, dataset_name, 
          debug_multiple_subgraphs=True, 
          save=True,
          print_every=10
          ):
    
    optimization_kwargs = config.optimization_kwargs
    #node_classifier_kwargs = config.node_classifier_kwargs
    watermark_kwargs = config.watermark_kwargs
    #subgraph_kwargs = config.subgraph_kwargs
    #augment_kwargs = config.augment_kwargs
    watermark_loss_kwargs = config.watermark_loss_kwargs
    regression_kwargs = config.regression_kwargs


    assert watermark_loss_kwargs['scale_beta_method'] in [None,'tan','tanh','clip']

    validate_kwargs()#optimization_kwargs, node_classifier_kwargs, subgraph_kwargs, augment_kwargs, watermark_kwargs, watermark_loss_kwargs, regression_kwargs)
    
    node_classifier, optimizer                  = initialize_training()#node_classifier_kwargs, optimization_kwargs['lr'])
    subgraph_dict, subgraph_signatures, all_subgraph_indices = setup_subgraph_dict(data, dataset_name)#, subgraph_kwargs, watermark_kwargs)
    node_aug, edge_aug                          = collect_augmentations()#augment_kwargs, node_classifier_kwargs['outDim'])
    history, betas_dict, beta_similarities_dict = setup_history(subgraph_signatures)

    if watermark_loss_kwargs['balance_beta_weights'] == True:
        balanced_weights = get_balanced_beta_weights([subgraph_dict[sig]['subgraph'] for sig in subgraph_signatures])
    elif watermark_loss_kwargs['balance_beta_weights'] == False:
        balanced_weights = torch.ones(len(subgraph_dict),data.x.shape[1])

    epochs = optimization_kwargs['epochs']
    sacrifice_method = optimization_kwargs['sacrifice_kwargs']['method']
    if sacrifice_method is not None:
        train_node_indices = torch.arange(data.x.shape[0])[data.train_mask]
        p_sacrifice = optimization_kwargs['sacrifice_kwargs']['percentage']
        train_nodes_to_consider = sacrifice_node_node_indices(train_node_indices,sacrifice_method,p_sacrifice,all_subgraph_indices)
    #     in [None,'subgraph_node_indices','train_node_indices']
    #     if optimization_kwargs['sacrifice_kwargs']['method']=='subgraph_node_indices':

    # if optimization_kwargs['sacrifice_subgraph_nodes']==True:
    #     p_sacrifice = optimization_kwargs['p_sacrifice_subgraph']
    #     train_node_indices = torch.arange(data.x.shape[0])[data.train_mask]
    #     train_node_indices_sans_sacrificed = sacrifice_subgraph_node_indices(train_node_indices, all_subgraph_indices, p_sacrifice)
    #     train_nodes_to_consider = train_node_indices_sans_sacrificed
    # if optimization_kwargs['sacrifice_node_indices_general']==True:
    #     p_sacrifice = optimization_kwargs['p_sacrifice_general']
    #     train_node_indices = torch.arange(data.x.shape[0])[data.train_mask]
    #     train_node_indices_sans_sacrificed = sacrifice_training_node_indices(train_node_indices, all_subgraph_indices, p_sacrifice)
    #     train_nodes_to_consider = train_node_indices_sans_sacrificed
    else:
        train_nodes_to_consider = data.train_mask

    all_feature_importances, all_watermark_indices, coef_wmk, probas = None, None, optimization_kwargs['coefWmk'], None
    # beta = torch.zeros(data.x.shape[1],dtype=torch.float)
    for epoch in tqdm(range(epochs)):

        optimizer.zero_grad()

        edge_index, x, y    = augment_data(data, node_aug, edge_aug)
        log_logits          = node_classifier(x, edge_index)
        loss_primary        = F.nll_loss(log_logits[train_nodes_to_consider], y[train_nodes_to_consider])
        loss_watermark, beta_similarity = torch.tensor(0,dtype=float), torch.tensor(0,dtype=float)
        beta_std_dev_penalty=torch.tensor(0,dtype=float)

        wmk_optimization_condtion_met = (watermark_kwargs['watermark_type']=='basic') or (watermark_kwargs['watermark_type']=='fancy' and epoch>=watermark_kwargs['fancy_selection_kwargs']['clf_only_epochs']) 
        if not wmk_optimization_condtion_met:
            ''' if designing based on coefficients but the time hasn't come to optimize watermark, hold off '''
            loss = loss_primary
            loss.backward()

        elif wmk_optimization_condtion_met:
            ''' otherwise if not doing a fancy watermark or if the epoch has been reached, optimize watermark '''
            
            probas = log_logits.clone().exp()

            if watermark_kwargs['watermark_type']=='fancy' and epoch==watermark_kwargs['fancy_selection_kwargs']['clf_only_epochs']:
                ''' Define watermark at subset of feature indices. If `False`, then watermark was previously-defined. '''
                subgraph_dict, all_watermark_indices, all_feature_importances = apply_fancy_watermark(data, subgraph_dict, probas)#, watermark_kwargs, regression_kwargs)



            betas_from_every_subgraph = []
            for s, sig in enumerate(subgraph_signatures):
                this_watermark, data_sub, subgraph_node_indices = [subgraph_dict[sig][k] for k in ['watermark','subgraph','nodeIndices']]

                x_sub = data_sub.x
                y_sub = probas[subgraph_node_indices]
                x_sub, y_sub = data_sub.x, probas[subgraph_node_indices]


                loss_watermark, beta_similarity, betas_dict, beta_similarities_dict = optimize_watermark(loss_watermark, beta_similarity, x_sub, y_sub, this_watermark, betas_dict, 
                                                                                                         beta_similarities_dict, sig, 
                                                                                                         epoch==epochs-1, 
                                                                                                         ignore_zeros_from_subgraphs=False, 
                                                                                                         debug=debug_multiple_subgraphs,
                                                                                                         watermark_loss_kwargs=watermark_loss_kwargs,
                                                                                                         regression_kwargs=regression_kwargs,
                                                                                                         balanced_beta_weights=balanced_weights[s])
                betas_from_every_subgraph.append(betas_dict[sig][-1])
            betas_from_every_subgraph = torch.vstack(betas_from_every_subgraph)

            loss_watermark /= len(subgraph_dict)
            beta_similarity /= len(subgraph_dict)


            reg = get_reg_term(betas_from_every_subgraph, optimization_kwargs['regularization_type'], optimization_kwargs['lambda_l2'])
            loss=loss_primary + coef_wmk*(loss_watermark+beta_std_dev_penalty)  +  reg
            loss.backward()

        optimizer.step()

        acc_trn = accuracy(log_logits[train_nodes_to_consider], y[train_nodes_to_consider])
        acc_val = accuracy(log_logits[train_nodes_to_consider],   y[train_nodes_to_consider])
        history = update_history_one_epoch(history, loss, loss_primary, loss_watermark, acc_trn, acc_val)
        if epoch%print_every==0:
            print_epoch_status(epoch, loss_primary, loss_watermark, beta_similarity, acc_trn, acc_val, wmk_optimization_condtion_met,beta_std_dev_penalty)
    
    history['betas']=betas_dict
    history['beta_similarities'] = beta_similarities_dict

    if save==True:
        save_results(node_classifier, history, subgraph_dict, all_feature_importances, all_watermark_indices, probas, dataset_name)#, node_classifier_kwargs, watermark_kwargs, subgraph_kwargs, optimization_kwargs, augment_kwargs, watermark_loss_kwargs, regression_kwargs)
    return node_classifier, history, subgraph_dict, all_feature_importances, all_watermark_indices, probas



def update_history_one_epoch(history, loss, loss_primary, loss_watermark, acc_trn, acc_val):
    history['losses'].append(loss.clone().detach())
    history['losses_primary'].append(loss_primary.clone().detach())
    history['losses_watermark'].append(loss_watermark.clone().detach())
    history['train_accs'].append(acc_trn)
    history['val_accs'].append(acc_val)
    return history


def gather_random_subgraphs_for_testing(data, dataset_name, 
                                        max_degrees_choices=[20,50,100], 
                                        frac_choices = [0.001,0.004,0.005,0.01], 
                                        restart_prob_choices = [0,0.1,0.2], 
                                        nHops_choices=[1,2,3], 
                                        limit_khop_num_nodes=False):
    
    fixed = False
    assert fixed, 'Need to address "num_watermarked_nodes" -- make sure it means what i want it to'
    # create blank subgraph_dict to fill
    subgraph_kwargs =   {'method':          'random',  
                         'fraction':        None,
                         'numSubgraphs':    1,
                                'khop_kwargs':      {'autoChooseSubGs': True,   'nodeIndices':  None,   'numHops': None,   'max_degree': None},
                                'random_kwargs':    {},
                                'rwr_kwargs':       {'restart_prob':None,       'max_steps':1000}}
    
    if dataset_name=='computers':
        max_degrees_choices = [20]

    use_train_mask=True
    num_ = 50
    avoid_indices = []
    subgraphs = []
    for i in range(num_):
        subgraph_kwargs['method'] = np.random.choice(['khop','random','random_walk_with_restart'])
        fraction = subgraph_kwargs['fraction'] = np.random.choice(frac_choices)
        print(f'Forming subgraph {i+1} of {num_}: {subgraph_kwargs['method']}',end='\r')
        if subgraph_kwargs['method']=='khop':
            subgraph_kwargs['khop_kwargs']['numHops'] = np.random.choice(nHops_choices)
            maxDegree = subgraph_kwargs['khop_kwargs']['maxDegree'] = np.random.choice(max_degrees_choices)
            random.seed(2575)
            num_watermarked_nodes = int(fraction*sum(data.train_mask))
            ranked_nodes = torch.tensor(rank_training_nodes_by_degree(dataset_name, data, max_degree=maxDegree)[:100])
            idxs = torch.randperm(len(ranked_nodes))
            ranked_nodes = ranked_nodes[idxs]
            node_indices_to_watermark = ranked_nodes[:num_watermarked_nodes]
            central_node = node_indices_to_watermark[0]
        elif subgraph_kwargs['method']=='random_walk_with_restart':
            subgraph_kwargs['rwr_kwargs']['restart_prob'] = np.random.choice(restart_prob_choices)
            ranked_nodes = torch.tensor(rank_training_nodes_by_degree(dataset_name, data, max_degree=maxDegree)[:100])
            idxs = torch.randperm(len(ranked_nodes))
            ranked_nodes = ranked_nodes[idxs]
            node_indices_to_watermark = ranked_nodes[:num_watermarked_nodes]
            central_node = node_indices_to_watermark[0]
        elif subgraph_kwargs['method']=='random':
            central_node=None

        data_sub, _, subgraph_node_indices = generate_subgraph(data, dataset_name, subgraph_kwargs, central_node, avoid_indices, use_train_mask, show=False)
        subgraphs.append((data_sub,subgraph_node_indices))
        try:
            avoid_indices += [node_index.item() for node_index in subgraph_node_indices]
        except:
            avoid_indices += [node_index.item() for node_index in subgraph_node_indices]

    return subgraphs


