import copy
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os
import numpy as np 
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
from subgraph_utils import *
from regression_utils import *
from transform_functions import *

def generate_basic_watermark(k, p_neg_ones=0.5, p_remove=0.75):
    j = int(p_neg_ones*k)
    watermark = torch.ones(k)
    watermark_neg_1_indices = torch.randperm(k)[:j]
    watermark[watermark_neg_1_indices] = -1

    j_0 = int(p_remove*k)
    watermark_remove_indices = torch.randperm(k)[:j_0]
    watermark[watermark_remove_indices] = 0
    return watermark

def get_node_indices_to_watermark(dataset_name, graph_to_watermark, subgraph_kwargs):
    if subgraph_kwargs['khop_kwargs']['autoChooseSubGs']==True:
        num_watermarked_nodes = subgraph_kwargs['numSubgraphs']
        print('num_watermarked_nodes:',num_watermarked_nodes)
        random.seed(2575)
        ranked_nodes = rank_training_nodes_by_degree(dataset_name, graph_to_watermark, max_degree=subgraph_kwargs['khop_kwargs']['max_degree'])
        node_indices_to_watermark = ranked_nodes[:num_watermarked_nodes]
    elif subgraph_kwargs['khop_kwargs']['autoChooseSubGs']==False:
        assert subgraph_kwargs['khop_kwargs']['nodeIndices'] is not None
        node_indices_to_watermark = subgraph_kwargs['khop_kwargs']['nodeIndices']
    return node_indices_to_watermark

def collect_subgraphs_within_single_graph_for_watermarking(data, dataset_name, use_train_mask, subgraph_kwargs):
    """
    Collect subgraphs within a single graph for watermarking purposes.

    Args:
        data (object): The dataset object.
        dataset_name (str): The name of the dataset.
        use_train_mask (bool): Whether to use the training mask.
        subgraph_kwargs (dict): Keyword arguments for subgraph generation.

    Returns:
        dict: A dictionary containing subgraph information with subgraph signatures as keys.
    """
    subgraph_data_dir = os.path.join(data_dir,dataset_name,'subgraphs')
    # if os.path.exists(subgraph_data_dir)==False:
    #     os.mkdir(subgraph_data_dir)
    # these_subgraphs_filename = get_subgraph_tag(dataset_name)
    # these_subgraphs_filepath = os.path.join(subgraph_data_dir,these_subgraphs_filename)
    # numSubgraphs = subgraph_kwargs['numSubgraphs']
    numSubgraphs = subgraph_kwargs['numSubgraphs']
    if subgraph_kwargs['regenerate']==False:
        subgraph_kwargs = config.subgraph_kwargs
        frac = subgraph_kwargs['fraction']
        numHops = subgraph_kwargs['khop_kwargs']['numHops']
        new_numHops = determine_whether_to_increment_numHops(dataset_name,frac,numSubgraphs,numHops)
        # need to know which subgraph_dict to actually look up
        if subgraph_kwargs['method']=='khop' and new_numHops > numHops:
            config.subgraph_kwargs['khop_kwargs']['numHops']=new_numHops
        these_subgraphs_filename = get_subgraph_tag(dataset_name)
        these_subgraphs_filepath = os.path.join(subgraph_data_dir,these_subgraphs_filename)
        try:
            subgraph_dict = pickle.load(open(these_subgraphs_filepath,'rb'))
        except:
            config.subgraph_kwargs['regenerate']=True
            subgraph_kwargs = config.subgraph_kwargs
    if subgraph_kwargs['regenerate']==True:
        if subgraph_kwargs['method']=='khop':
            ''' this method relies on node_indices_to_watermark '''
            node_indices_to_watermark = get_node_indices_to_watermark(dataset_name, data, subgraph_kwargs)
            assert node_indices_to_watermark is not None
            enumerate_over_me = node_indices_to_watermark
        elif subgraph_kwargs['method']=='random_walk_with_restart':
            ''' this method relies on numSubgraphs '''
            node_indices_to_watermark = get_node_indices_to_watermark(dataset_name, data, subgraph_kwargs)
            assert node_indices_to_watermark is not None
            enumerate_over_me = node_indices_to_watermark
        elif subgraph_kwargs['method']=='random':
            ''' this method relies on numSubgraphs '''
            assert numSubgraphs is not None
            enumerate_over_me = range(numSubgraphs)

        subgraph_dict = {}
        seen_nodes = []
        print("enumerate_over_me:",enumerate_over_me)
        for i, item in enumerate(enumerate_over_me):
            avoid_indices=seen_nodes
            # print(f'Forming subgraph {i+1} of {numSubgraphs}',end='\r')
            if subgraph_kwargs['method'] in ['khop', 'random_walk_with_restart']:
                central_node=item
                # print("central_node:",central_node)
            else:
                central_node=None

            print(f"Before generate_subgraph: numHops = {subgraph_kwargs['khop_kwargs']['numHops']}")
            data_sub, subgraph_signature, subgraph_node_indices = generate_subgraph(data, dataset_name, subgraph_kwargs, central_node, avoid_indices, use_train_mask, show=False)
            print('subgraph_kwargs afterward:',subgraph_kwargs)
            subgraph_dict[subgraph_signature] = {'subgraph': data_sub, 'nodeIndices': subgraph_node_indices}
            seen_nodes += subgraph_node_indices.tolist()

        subgraph_data_dir = os.path.join(data_dir,dataset_name,'subgraphs')
        if os.path.exists(subgraph_data_dir)==False:
            os.mkdir(subgraph_data_dir)
        these_subgraphs_filename = get_subgraph_tag(dataset_name)
        these_subgraphs_filepath = os.path.join(subgraph_data_dir,these_subgraphs_filename)
        numSubgraphs = config.subgraph_kwargs['numSubgraphs']
        with open(these_subgraphs_filepath,'wb') as f:
            pickle.dump(subgraph_dict, f)

    all_subgraph_node_indices = []
    for sig in subgraph_dict.keys():
        nodeIndices = subgraph_dict[sig]['nodeIndices'].tolist()
        print('nodeIndices:',nodeIndices)
        all_subgraph_node_indices += nodeIndices
    all_subgraph_node_indices = torch.tensor(all_subgraph_node_indices)
    return subgraph_dict, all_subgraph_node_indices

def select_fancy_watermark_indices_shared(num_features,subgraph_dict, probas, probas_dict, num_indices, use_rand, use_unimpt, use_concat, use_avg):
    regression_kwargs = config.regression_kwargs
    if use_rand:
        feature_importance, beta = [],[]
        ordered_indices = torch.randperm(num_features)
    elif use_unimpt:
        if use_concat:
            all_indices = torch.concat([subgraph_dict[k]['nodeIndices'] for k in subgraph_dict.keys()])
            all_x = torch.concat([subgraph_dict[k]['subgraph'].x for k in subgraph_dict.keys()])
            all_probas_sep_fwd = torch.concat([probas_dict[k] for k in subgraph_dict.keys()])
            all_probas = probas[all_indices]
            print('all probas sep fwd passes:\n',all_probas_sep_fwd)
            print('all probas regular:\n',all_probas)

            # if config.optimization_kwargs['separate_forward_passes_per_subgraph']==True:
            #     all_probas = torch.concat([probas_dict[k] for k in subgraph_dict.keys()])
            # else:
            #     all_probas = probas[all_indices]
            beta= regress_on_subgraph(all_x, all_probas, regression_kwargs)
        elif use_avg:
            betas = []
            for k in subgraph_dict.keys():
                indices_this_sub = subgraph_dict[k]['nodeIndices']
                x_this_sub = subgraph_dict[k]['subgraph'].x
                probas_this_sub_sep_fwd = probas_dict[k]
                probas_this_sub = probas[indices_this_sub]
                print('probas this sub sep fwd passes:\n',probas_this_sub_sep_fwd)
                print('probas this sub regular:\n',probas_this_sub)
                # if config.optimization_kwargs['separate_forward_passes_per_subgraph']==True:
                #     probas_this_sub = probas_dict[k]
                # else:
                #     probas_this_sub = probas[indices_this_sub]
                beta_this_sub= regress_on_subgraph(x_this_sub, probas_this_sub, regression_kwargs)
                betas.append(beta_this_sub)
            beta = torch.mean(torch.vstack(betas),dim=0)
        feature_importance = beta.abs()
        ordered_indices = sorted(range(num_features), key=lambda item: feature_importance[item])

    all_zero_features = [torch.where(torch.sum(subgraph_dict[k]['subgraph'].x, dim=0) == 0)[0] for k in subgraph_dict.keys()]
    indices = []
    i=0
    while len(indices)<num_indices:
        if item_not_in_any_list(ordered_indices[i],all_zero_features):
            try:
                indices.append(ordered_indices[i].item())
                print('importance:',feature_importance[ordered_indices[i].item()])

            except:
                indices.append(ordered_indices[i])
                print('importance:',feature_importance[ordered_indices[i]])

        i +=1 
    all_indices             = [indices]*len(subgraph_dict)
    all_feature_importances = [feature_importance]*len(subgraph_dict)
    all_betas               = [beta]*len(subgraph_dict)

    return all_indices, all_feature_importances, all_betas

def select_fancy_watermark_indices_individual(num_features,subgraph_dict, k, use_rand, use_unimpt, probas, probas_dict, num_indices):
    regression_kwargs = config.regression_kwargs
    # if k is None:
        # k = list(subgraph_dict.keys())[0]

    if use_rand:
        feature_importance, beta = [], []
        ordered_indices = torch.randperm(num_features)

    elif use_unimpt:
        indices_this_sub = subgraph_dict[k]['nodeIndices']
        x_this_sub = subgraph_dict[k]['subgraph'].x
        if config.optimization_kwargs['separate_forward_passes_per_subgraph']==True:
            these_probas = probas_dict[k]
        else:
            these_probas = probas[indices_this_sub]
        beta= regress_on_subgraph(x_this_sub, these_probas, regression_kwargs)
        feature_importance = beta.abs()
        ordered_indices = sorted(range(num_features), key=lambda item: feature_importance[item])

    i=0
    indices = []
    zero_features = torch.where(torch.sum(subgraph_dict[k]['subgraph'].x, dim=0) == 0)[0]
    while len(indices)<num_indices:
        if ordered_indices[i] not in zero_features:
            try:
                indices.append(ordered_indices[i].item())
            except:
                indices.append(ordered_indices[i])
        i +=1 

    return indices, feature_importance, beta


# select_fancy_watermark_indices(num_features, subgraph_dict, probas, probas_dict)
def select_fancy_watermark_indices(num_features,subgraph_dict, probas, probas_dict):
    watermark_kwargs = config.watermark_kwargs
    use_unimpt, use_rand, use_concat, use_avg, use_indiv, num_subgraphs, num_indices, message = describe_selection_config(num_features, watermark_kwargs, subgraph_dict)
    if num_subgraphs==1:
        k = list(subgraph_dict.keys())[0]
        indices, feature_importance, beta = select_fancy_watermark_indices_individual(num_features, subgraph_dict, k, use_rand, use_unimpt, probas, probas_dict, num_indices)
        all_indices, all_feature_importances, all_betas = [indices], [feature_importance], [beta]

    elif num_subgraphs>1:
        if use_indiv:
            all_indices, all_feature_importances, all_betas = [],[],[]
            for k in subgraph_dict.keys():
                indices, feature_importance, beta = select_fancy_watermark_indices_individual(num_features, subgraph_dict, k, use_rand, use_unimpt, probas, probas_dict, num_indices)
                all_indices.append(indices)
                all_feature_importances.append(feature_importance)
                all_betas.append(beta)
        else:
            all_indices, all_feature_importances, all_betas = select_fancy_watermark_indices_shared(num_features, subgraph_dict, probas, probas_dict, num_indices, use_rand, use_unimpt, use_concat, use_avg)

    assert message is not None
    print(message)
    return all_indices, all_feature_importances, all_betas

def apply_basic_watermark(data, subgraph_dict, watermark_kwargs):
    print('apply_basic_watermark')
    if watermark_kwargs['watermark_type']=='basic':
        watermark = torch.zeros(data.x.shape[1])
        p_remove = watermark_kwargs['basic_selection_kwargs']['p_remove']
        watermark = generate_basic_watermark(k=data.num_node_features, p_neg_ones=0.5, p_remove=p_remove)
        features_all_subgraphs = torch.vstack([subgraph_dict[subgraph_central_node]['subgraph'].x for subgraph_central_node in subgraph_dict.keys()]).squeeze()
        zero_features_across_subgraphs = torch.where(torch.sum(features_all_subgraphs,dim=0)==0)
        watermark[zero_features_across_subgraphs]=0
        for subgraph_sig in subgraph_dict.keys():
            subgraph_dict[subgraph_sig]['watermark']=watermark
    return subgraph_dict

def apply_fancy_watermark(num_features, subgraph_dict, probas, probas_dict):
    print('apply_fancy_watermark')

    #    if add_summary_beta==True:
    #         all_indices = torch.concat([subgraph_dict[k]['nodeIndices'] for k in subgraph_dict.keys()])
    #         sig = '_'.join(all_indices)
    #         beta = regress_on_subgraph(data, all_indices, probas, regression_kwargs)


    all_watermark_indices, all_feature_importances, _ = select_fancy_watermark_indices(num_features, subgraph_dict, probas, probas_dict)
    u = len(all_watermark_indices[0])
    h1,h2,random_order = u//2, u-u//2, torch.randperm(u)
    nonzero_watermark_values = torch.tensor([1]*h1 + [-1]*h2)[random_order].float()
    for i, subgraph_sig in enumerate(subgraph_dict.keys()):
        this_watermark = torch.zeros(num_features)
        watermarked_feature_indices = all_watermark_indices[i]
        this_watermark[watermarked_feature_indices]=nonzero_watermark_values
        subgraph_dict[subgraph_sig]['watermark']=this_watermark
        not_0 = torch.where(this_watermark!=0)[0]
    
    return subgraph_dict, all_watermark_indices, all_feature_importances


