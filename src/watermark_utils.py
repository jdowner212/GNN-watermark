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

def select_unimportant_watermark_indices_shared(num_features,subgraph_dict, probas, probas_dict, len_watermark, use_concat, use_avg):
    regression_kwargs = config.regression_kwargs
   # if watermark_type=='basic':
  #      feature_importance, beta = [],[]
 #       ordered_indices = torch.randperm(num_features)
#    elif watermark_type=='unimportant':
    if use_concat:
        all_indices = torch.concat([subgraph_dict[k]['nodeIndices'] for k in subgraph_dict.keys()])
        all_x = torch.concat([subgraph_dict[k]['subgraph'].x for k in subgraph_dict.keys()])
        if config.optimization_kwargs['separate_forward_passes_per_subgraph']==True:
            all_probas = torch.concat([probas_dict[k] for k in subgraph_dict.keys()])
        else:
            all_probas = probas[all_indices]
        beta= regress_on_subgraph(all_x, all_probas, regression_kwargs)
    elif use_avg:
        betas = []
        for k in subgraph_dict.keys():
            indices_this_sub = subgraph_dict[k]['nodeIndices']
            x_this_sub = subgraph_dict[k]['subgraph'].x
            if config.optimization_kwargs['separate_forward_passes_per_subgraph']==True:
                probas_this_sub = probas_dict[k]
            else:
                probas_this_sub = probas[indices_this_sub]
            beta_this_sub= regress_on_subgraph(x_this_sub, probas_this_sub, regression_kwargs)
            betas.append(beta_this_sub)
        beta = torch.mean(torch.vstack(betas),dim=0)
    feature_importance = beta.abs()
    ordered_indices = sorted(range(num_features), key=lambda item: feature_importance[item])

    all_zero_features = [torch.where(torch.sum(subgraph_dict[k]['subgraph'].x, dim=0) == 0)[0] for k in subgraph_dict.keys()]
    indices = []
    i=0
    while len(indices)<len_watermark:
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

def select_unimportant_watermark_indices_individual(num_features,subgraph_dict, k, probas, probas_dict, len_watermark):
    regression_kwargs = config.regression_kwargs
    # if k is None:
        # k = list(subgraph_dict.keys())[0]

    # if use_rand:
    #     feature_importance, beta = [], []
    #     ordered_watermark_indices = torch.randperm(num_features)

    # elif use_unimpt:
    indices_this_sub = subgraph_dict[k]['nodeIndices']
    x_this_sub = subgraph_dict[k]['subgraph'].x
    if config.optimization_kwargs['separate_forward_passes_per_subgraph']==True:
        these_probas = probas_dict[k]
    else:
        these_probas = probas[indices_this_sub]
    beta= regress_on_subgraph(x_this_sub, these_probas, regression_kwargs)
    feature_importance = beta.abs()
    ordered_watermark_indices = sorted(range(num_features), key=lambda item: feature_importance[item])

    i=0
    watermark_indices = []
    zero_features = torch.where(torch.sum(subgraph_dict[k]['subgraph'].x, dim=0) == 0)[0]
    while len(watermark_indices)<len_watermark:
        if ordered_watermark_indices[i] not in zero_features:
            try:
                watermark_indices.append(ordered_watermark_indices[i].item())
            except:
                watermark_indices.append(ordered_watermark_indices[i])
        i +=1 

    return watermark_indices, feature_importance, beta


def select_unimportant_watermark_indices(num_features,subgraph_dict, probas, probas_dict):
    watermark_kwargs = config.watermark_kwargs
    use_concat, use_avg, use_indiv, num_subgraphs, len_watermark, message = describe_selection_config(num_features, watermark_kwargs, subgraph_dict)
    if num_subgraphs==1:
        k = list(subgraph_dict.keys())[0]
        watermark_indices, feature_importance, beta = select_unimportant_watermark_indices_individual(num_features, subgraph_dict, k, probas, probas_dict, len_watermark)
        each_subgraph_watermark_indices, each_subgraph_feature_importances, each_subgraph_betas = [watermark_indices], [feature_importance], [beta]

    elif num_subgraphs>1:
        if use_indiv:
            each_subgraph_watermark_indices, each_subgraph_feature_importances, each_subgraph_betas = [],[],[]
            for k in subgraph_dict.keys():
                watermark_indices, feature_importance, beta = select_unimportant_watermark_indices_individual(num_features, subgraph_dict, k, probas, probas_dict, len_watermark)
                each_subgraph_watermark_indices.append(watermark_indices)
                each_subgraph_feature_importances.append(feature_importance)
                each_subgraph_betas.append(beta)
        else:
            each_subgraph_watermark_indices, each_subgraph_feature_importances, each_subgraph_betas = select_unimportant_watermark_indices_shared(num_features, subgraph_dict, probas, probas_dict, len_watermark, use_concat, use_avg)

    assert message is not None
    print(message)
    return each_subgraph_watermark_indices, each_subgraph_feature_importances, each_subgraph_betas

def create_basic_watermarks(num_features, len_watermark):# subgraph_dict, watermark_kwargs):
    watermark = torch.zeros(num_features)
    watermark_indices = torch.randperm(num_features)[:len_watermark]
    nonzero_watermark_values = collect_watermark_values(len_watermark)
    watermark[watermark_indices]=nonzero_watermark_values
    
    # p_remove = watermark_kwargs['basic_selection_kwargs']['p_remove']
    # watermark = generate_basic_watermark(k=num_features, p_neg_ones=0.5, p_remove=p_remove)
    # features_all_subgraphs = torch.vstack([subgraph_dict[subgraph_central_node]['subgraph'].x for subgraph_central_node in subgraph_dict.keys()]).squeeze()
    # zero_features_across_subgraphs = torch.where(torch.sum(features_all_subgraphs,dim=0)==0)
    # watermark[zero_features_across_subgraphs]=0

    
    
    return watermark

def collect_watermark_values(len_watermark):
    h1,h2,random_order = len_watermark//2, len_watermark-len_watermark//2, torch.randperm(len_watermark)
    nonzero_watermark_values = torch.tensor([1]*h1 + [-1]*h2)[random_order].float()
    return nonzero_watermark_values


def create_watermarks_at_unimportant_indices(num_features, subgraph_dict, probas, probas_dict):
    each_subgraph_watermark_indices, each_subgraph_feature_importances, _ = select_unimportant_watermark_indices(num_features, subgraph_dict, probas, probas_dict)
    nonzero_watermark_values = collect_watermark_values(len(each_subgraph_watermark_indices[0]))
    watermarks=[]
    for i in range(len(subgraph_dict)):
        this_watermark = torch.zeros(num_features)
        this_watermark[each_subgraph_watermark_indices[i]]=nonzero_watermark_values
        watermarks.append(this_watermark)
    return watermarks, each_subgraph_watermark_indices, each_subgraph_feature_importances

    # for i, subgraph_sig in enumerate(subgraph_dict.keys()):
    #     this_watermark = torch.zeros(num_features)
    #     this_watermark[each_subgraph_watermark_indices[i]]=nonzero_watermark_values
    #     subgraph_dict[subgraph_sig]['watermark']=this_watermark
    #     # not_0 = torch.where(this_watermark!=0)[0]
    
    # return subgraph_dict, each_subgraph_watermark_indices, each_subgraph_feature_importances

def select_most_represented_feature_indices(x, len_watermark):
    nonzero_feat_mask = x!=0
    nonzero_feat_counts = torch.sum(nonzero_feat_mask,dim=0)
    sorted_indices = torch.argsort(nonzero_feat_counts, descending=True)
    print('len sorted_indices:',len(sorted_indices))
    most_represented_indices = sorted_indices[:len_watermark]#nonzero_feat_counts[sorted_indices[:len_watermark]]
    return most_represented_indices

def create_watermarks_at_most_represented_indices(num_subgraphs, len_watermark, num_features, most_represented_indices):    
    nonzero_watermark_values = collect_watermark_values(len_watermark)
    watermarks=[]
    for _ in range(num_subgraphs):
        this_watermark = torch.zeros(num_features)
        this_watermark[most_represented_indices]=nonzero_watermark_values
        watermarks.append(this_watermark)
    return watermarks

def apply_watermark(watermark_type, num_features, len_watermark, subgraph_dict, x=None, probas=None, probas_dict=None, watermark_kwargs=None):
    each_subgraph_feature_importances=None
    if watermark_type=='basic':
        assert watermark_kwargs is not None
        # watermark = create_basic_watermarks(num_features, len_watermark, subgraph_dict, watermark_kwargs)
        watermark = create_basic_watermarks(num_features, len_watermark)
        watermarks = [watermark]*len(subgraph_dict)
        watermark_indices = torch.where(watermark!=0)[0]
        each_subgraph_watermark_indices = [watermark_indices]*(len(subgraph_dict))
    elif watermark_type=='unimportant':# or watermark_type=='most_represented':
        assert probas is not None
        assert probas_dict is not None
        watermarks, each_subgraph_watermark_indices, each_subgraph_feature_importances = create_watermarks_at_unimportant_indices(num_features, subgraph_dict, probas, probas_dict)
    elif watermark_type=='most_represented':
        assert x is not None
        most_represented_indices = select_most_represented_feature_indices(x, len_watermark)
        print('len_watermark:',len_watermark)
        print('most_represented_indices:',most_represented_indices)
        watermarks = create_watermarks_at_most_represented_indices(len(subgraph_dict), len_watermark, num_features, most_represented_indices)
        print('watermarks:',watermarks)
        watermark_indices = torch.where(watermarks[0]!=0)[0]
        print('watermark_indices:',watermark_indices)
        each_subgraph_watermark_indices = [watermark_indices]*(len(subgraph_dict))
    for i, subgraph_sig in enumerate(subgraph_dict.keys()):
        subgraph_dict[subgraph_sig]['watermark']=watermarks[i]
    return subgraph_dict, each_subgraph_watermark_indices, each_subgraph_feature_importances

# subgraph_dict, each_subgraph_watermark_indices, each_subgraph_feature_importances = apply_watermark(watermark_type, num_features, len_watermark, subgraph_dict, x=None, probas=None, probas_dict=None, watermark_kwargs=None)




