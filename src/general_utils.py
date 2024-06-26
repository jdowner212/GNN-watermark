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

# from config import *
import config
from models import *



def count_matches(features):
    num_matches = 0
    for i in range(features.shape[1]):
        if torch.all(features[:, i] == features[0, i]):
            num_matches += 1
    return num_matches

def describe_selection_config(data, watermark_kwargs, subgraph_dict):
    use_unimpt  = watermark_kwargs['fancy_selection_kwargs']['selection_strategy']=='unimportant'
    use_rand    = watermark_kwargs['fancy_selection_kwargs']['selection_strategy']=='random'
    use_concat  = watermark_kwargs['fancy_selection_kwargs']['multi_subg_strategy']=='concat'
    use_avg     = watermark_kwargs['fancy_selection_kwargs']['multi_subg_strategy']=='average'
    use_indiv   = watermark_kwargs['fancy_selection_kwargs']['evaluate_individually']
    
    num_subgraphs=len(subgraph_dict)
    perc        = watermark_kwargs['fancy_selection_kwargs']['percent_of_features_to_watermark']
    num_indices = int(perc*data.x.shape[1]/100)

    message=None
    end_tag = ', single subgraph' if num_subgraphs==1 else ', individualized for each subgraph' if (num_subgraphs>1 and use_indiv) else ', uniformly across subgraphs' if (num_subgraphs>1 and not use_indiv) else ''
    if use_unimpt:
        beta_tag = 'beta from single subgraph' if (num_subgraphs==1) else 'betas' if (num_subgraphs>1 and use_indiv) else 'betas from concatenated subgraphs' if (num_subgraphs>1 and not use_indiv and use_concat) else 'averaged betas from subgraphs' if (num_subgraphs>1 and use_unimpt and not use_indiv and use_avg) else ''
        message = 'Using ' + beta_tag + f' to identify bottom {perc}% of important feature indices for watermarking' + end_tag
    if use_rand:
        message = f"Selecting random {perc}% of feature indices for watermarking" + end_tag 

    return use_unimpt, use_rand, use_concat, use_avg, use_indiv, num_subgraphs, num_indices, message

def get_augment_tag():#augment_kwargs):
    augment_kwargs = config.augment_kwargs
    augment_tags = []
    if augment_kwargs['nodeDrop']['use']:
        augment_tags.append(f'nodeDropP{np.round(augment_kwargs["nodeDrop"]["p"], 5)}')
    if augment_kwargs['nodeMixUp']['use']:
        augment_tags.append(f'nodeMixUp{np.round(augment_kwargs["nodeMixUp"]["lambda"], 5)}')
    if augment_kwargs['nodeFeatMask']['use']:
        augment_tags.append(f'nodeFeatMask{np.round(augment_kwargs["nodeFeatMask"]["p"], 5)}')
    if augment_kwargs['edgeDrop']['use']:
        augment_tags.append(f'edgeDrop{np.round(augment_kwargs["edgeDrop"]["p"], 5)}')    
    augment_tag = '_'.join(augment_tags)
    return augment_tag

def get_model_tag():#node_classifier_kwargs):
    node_classifier_kwargs = config.node_classifier_kwargs
    arch = node_classifier_kwargs['arch']
    act = node_classifier_kwargs['activation']
    nLayers = node_classifier_kwargs['nLayers']
    hDim = node_classifier_kwargs['hDim']
    dropout = node_classifier_kwargs['dropout']
    skip = node_classifier_kwargs['skip_connections']
    model_tag = f'arch{arch}_{act}_nLayers{nLayers}_hDim{hDim}_drop{dropout}_skip{skip}'
    if arch == 'GAT':
        heads_1 = node_classifier_kwargs['heads_1']
        heads_2 = node_classifier_kwargs['heads_2']
        model_tag += f'_heads_{heads_1}_{heads_2}'
    return model_tag

def get_optimization_tag():#optimization_kwargs):
    optimization_kwargs = config.optimization_kwargs
    [lr, epochs, coef_wmk] = [optimization_kwargs[k] for k in ['lr','epochs','coefWmk']]
    tag = f'lr{lr}_epochs{epochs}_coefWmk{coef_wmk}'
    if optimization_kwargs['sacrifice_kwargs']['method'] == 'subgraph_node_indices':
        tag += f'_sacrifice{optimization_kwargs['sacrifice_kwargs']["percentage"]}subNodes'
    elif optimization_kwargs['sacrifice_kwargs']['method'] == 'train_node_indices':
        tag += f'_sacrifice{optimization_kwargs['sacrifice_kwargs']["percentage"]}trainNodes'
    if optimization_kwargs['regularization_type'] is not None:
        tag += f'_{optimization_kwargs["regularization_type"]}'
        if optimization_kwargs['regularization_type']=='L2':
            tag += f'_lambdaL2{optimization_kwargs["lambda_l2"]}'
    return tag

def get_regression_tag():##regression_kwargs):
    regression_kwargs = config.regression_kwargs
    lambda_ = regression_kwargs['lambda']
    regression_tag = f'regressionLambda{lambda_}'
    return regression_tag

def get_subgraph_tag(#subgraph_kwargs, 
                     dataset_name
                     ):
    subgraph_kwargs = config.subgraph_kwargs
    subgraph_tag = ''
    fraction = subgraph_kwargs['fraction']
    numSubgraphs = subgraph_kwargs['numSubgraphs']
    method = subgraph_kwargs['method']
    if method == 'khop':
        khop_kwargs = subgraph_kwargs['khop_kwargs']
        autoChooseSubGs = khop_kwargs['autoChooseSubGs']
        if autoChooseSubGs:
            pass
        else:
            nodeIndices = khop_kwargs['nodeIndices']
            num_nodes = dataset_attributes[dataset_name]['num_nodes']
            fraction = np.round(len(nodeIndices) / num_nodes, 5)
        numHops = khop_kwargs['numHops']
        max_degree = khop_kwargs['max_degree']
        subgraph_tag = f'{method}{numHops}_fraction{fraction}_numSubgraphs{numSubgraphs}_maxDegree{max_degree}'
    elif method == 'random':
        subgraph_tag = f'{method}_fraction{fraction}_numSubgraphs{numSubgraphs}'
    elif method == 'random_walk_with_restart':
        rwr_kwargs = subgraph_kwargs['rwr_kwargs']
        restart_prob = rwr_kwargs['restart_prob']
        max_steps = rwr_kwargs['max_steps']
        subgraph_tag = f'{method}_fraction{fraction}_numSubgraphs{numSubgraphs}_restart_prob{restart_prob}_maxSteps{max_steps}'
    return subgraph_tag

def get_watermark_loss_tag():#watermark_loss_kwargs):
    watermark_loss_kwargs = config.watermark_loss_kwargs
    tag = f'eps{watermark_loss_kwargs['epsilon']}_'
    if watermark_loss_kwargs['scale_beta_method'] is None:
        tag+='raw_beta'
    else:
        if watermark_loss_kwargs['scale_beta_method']=='tanh':
            tag+= f'tanh_{watermark_loss_kwargs["alpha"]}*beta'
        elif watermark_loss_kwargs['scale_beta_method']=='tan':
            tag+= f'tan_{watermark_loss_kwargs["alpha"]}*beta'
        elif watermark_loss_kwargs['scale_beta_method']=='clip':
            tag+= f'clipped_beta'
    if watermark_loss_kwargs['balance_beta_weights']==True:
        tag += '_balanced_beta_weights'
    return tag

def get_watermark_tag(#watermark_kwargs, 
                      dataset_name):
    watermark_kwargs = config.watermark_kwargs
    single_or_multi_graph = dataset_attributes[dataset_name]['single_or_multi_graph']
    pGraphs = watermark_kwargs['pGraphs']
    wmk_tag = ''

    if single_or_multi_graph == 'multi':
        wmk_tag = f'pGraphs{pGraphs}_' + wmk_tag

    if watermark_kwargs['watermark_type']=='fancy':
        percent_wmk = watermark_kwargs['fancy_selection_kwargs']['percent_of_features_to_watermark']
        strategy = watermark_kwargs['fancy_selection_kwargs']['selection_strategy'].capitalize()
        wmk_tag += f'_{percent_wmk}%{strategy}Indices'
        if watermark_kwargs['fancy_selection_kwargs']['evaluate_individually']:
            handle_multiple = 'individualized'
        else:
            handle_multiple = watermark_kwargs['fancy_selection_kwargs']['multi_subg_strategy']
        wmk_tag += f'_{handle_multiple}'
        wmk_tag += f'_{watermark_kwargs['fancy_selection_kwargs']['clf_only_epochs']}ClfEpochs'

    elif watermark_kwargs['watermark_type']=='basic':
        percent_wmk = np.round(100 * (1 - watermark_kwargs['basic_selection_kwargs']['p_remove']), 3)
        wmk_tag += f'_{percent_wmk}%BasicIndices'
    
    return wmk_tag

def get_results_folder_name(dataset_name):#, optimization_kwargs, node_classifier_kwargs, watermark_kwargs, subgraph_kwargs, augment_kwargs, watermark_loss_kwargs, regression_kwargs):
    """
    Generate the folder name for storing results based on various configuration parameters.

    Args:
        dataset_name (str): The name of the dataset.
        lr (float): Learning rate.
        epochs (int): Number of training epochs.
        node_classifier_kwargs (dict): Keyword arguments for the node classifier.
        watermark_kwargs (dict): Keyword arguments for watermarking.
        subgraph_kwargs (dict): Keyword arguments for subgraph generation.
        augment_kwargs (dict): Keyword arguments for data augmentation.

    Returns:
        str: The generated folder name for storing results.
    """
    # global optimization_kwargs, node_classifier_kwargs, watermark_kwargs, subgraph_kwargs, augment_kwargs, watermark_loss_kwargs, regression_kwargs
    # print('optimization_kwargs:',optimization_kwargs)
    # Model Tag Construction
    model_tag = get_model_tag()#node_classifier_kwargs)
    wmk_tag = get_watermark_tag(#watermark_kwargs, 
                                dataset_name)
    subgraph_tag = get_subgraph_tag(#subgraph_kwargs, 
                                    dataset_name)
    loss_tag = get_watermark_loss_tag()#watermark_loss_kwargs)
    augment_tag = get_augment_tag()#augment_kwargs)
    optimization_tag = get_optimization_tag()#optimization_kwargs)
    regression_tag = get_regression_tag()#regression_kwargs)


    # Combine All Tags into Config Name
    model_folder_name = model_tag
    config_name = f'{wmk_tag}_{subgraph_tag}_{loss_tag}_{augment_tag}_{optimization_tag}_{regression_tag}'
    dataset_folder_name = os.path.join(results_dir, dataset_name)

    if os.path.exists(os.path.join(dataset_folder_name, model_folder_name))==False:
        os.mkdir(os.path.join(dataset_folder_name, model_folder_name))

    return os.path.join(dataset_folder_name, model_folder_name, config_name)

def item_not_in_any_list(item, list_of_lists):
    for sublist in list_of_lists:
        if item in sublist:
            return False
    return True
    
def name_compare_dict(dataset_name, optimization_kwargs, node_classifier_kwargs, watermark_kwargs, subgraph_kwargs, augment_kwargs, watermark_loss_kwargs):
    folder_name = get_results_folder_name(dataset_name, optimization_kwargs, node_classifier_kwargs, watermark_kwargs, subgraph_kwargs, augment_kwargs, watermark_loss_kwargs)
    compare_dict_name = f"compare_dicts_{folder_name}"
    return compare_dict_name

def save_results(node_classifier, history, subgraph_dict, all_feature_importances, all_watermark_indices, probas, dataset_name):#, node_classifier_kwargs, watermark_kwargs, subgraph_kwargs, optimization_kwargs, augment_kwargs, watermark_loss_kwargs,regression_kwargs):
    results_folder_name = get_results_folder_name(dataset_name)#, optimization_kwargs, node_classifier_kwargs, watermark_kwargs, subgraph_kwargs, augment_kwargs, watermark_loss_kwargs, regression_kwargs)
    if os.path.exists(results_folder_name)==False:
        os.mkdir(results_folder_name)
    for object_name, object in zip(['node_classifier','history','subgraph_dict','all_feature_importances','all_watermark_indices','probas'],
                                   [ node_classifier,  history,  subgraph_dict,  all_feature_importances,  all_watermark_indices,  probas]):
        with open(os.path.join(results_folder_name,object_name),'wb') as f:
            pickle.dump(object,f)
    print('Node classifier, history, subgraph dict, feature importances, watermark indices, and probas saved in:')
    print(results_folder_name)

def unpack_dict(dictionary,keys):
    return [dictionary[k] for k in keys]

def update_dict(dict_,keys,values):
    for k,v in zip(keys,values):
        dict_[k]=v
    return dict_




