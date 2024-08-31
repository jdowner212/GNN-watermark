import numpy as np
import os
import numpy as np 
import pickle
import textwrap
from   tqdm.notebook import tqdm
import torch


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

def describe_selection_config(num_features,watermark_kwargs, subgraph_dict):
    watermark_type = watermark_kwargs['watermark_type']
    use_concat  = watermark_kwargs['unimportant_selection_kwargs']['multi_subg_strategy']=='concat'
    use_avg     = watermark_kwargs['unimportant_selection_kwargs']['multi_subg_strategy']=='average'
    use_indiv   = watermark_kwargs['unimportant_selection_kwargs']['evaluate_individually']
    
    num_subgraphs=len(subgraph_dict)
    perc        = watermark_kwargs['percent_of_features_to_watermark']
    len_watermark = int(perc*num_features/100)

    message=None
    end_tag = ', single subgraph' if num_subgraphs==1 else ', individualized for each subgraph' if (num_subgraphs>1 and use_indiv) else ', uniformly across subgraphs' if (num_subgraphs>1 and not use_indiv) else ''
    if watermark_type=='basic':
        message = f"Selecting random {perc}% of feature indices for watermarking" + end_tag 
    if watermark_type=='unimportant':
        beta_tag = 'beta from single subgraph' if (num_subgraphs==1) else 'betas' if (num_subgraphs>1 and use_indiv) else 'betas from concatenated subgraphs' if (num_subgraphs>1 and not use_indiv and use_concat) else 'averaged betas from subgraphs' if (num_subgraphs>1 and watermark_type=='unimportant' and not use_indiv and use_avg) else ''
        message = 'Using ' + beta_tag + f' to identify bottom {perc}% of important feature indices for watermarking' + end_tag
    if watermark_type=='most_represented':
        message = f'Selecting {perc}% of most-represented feature indices for watermarking' + end_tag
    return use_concat, use_avg, use_indiv, num_subgraphs, len_watermark, message

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




def get_model_tag():
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

def get_optimization_tag():
    optimization_kwargs = config.optimization_kwargs
    clf_only = optimization_kwargs['clf_only']
    [lr, epochs] = [optimization_kwargs[k] for k in ['lr','epochs']]
    coef_wmk = optimization_kwargs['coefWmk_kwargs']['coefWmk']
    tag = f'lr{lr}_epochs{epochs}'
    
    if clf_only==False:
        tag += f'_coefWmk{coef_wmk}'
        # if optimization_kwargs['sacrifice_kwargs']['method'] == 'subgraph_node_indices':
            # tag += f'_sacrifice{optimization_kwargs['sacrifice_kwargs']["percentage"]}subNodes'
        # elif optimization_kwargs['sacrifice_kwargs']['method'] == 'train_node_indices':
            # tag += f'_sacrifice{optimization_kwargs['sacrifice_kwargs']["percentage"]}trainNodes'
    if optimization_kwargs['regularization_type'] is not None:
        tag += f'_{optimization_kwargs["regularization_type"]}'
        if optimization_kwargs['regularization_type']=='L2':
            tag += f'_lambdaL2{optimization_kwargs["lambda_l2"]}'
    if clf_only==False:
        if optimization_kwargs['penalize_similar_subgraphs']==True:
            p_swap = optimization_kwargs['p_swap']
            coef = optimization_kwargs['shifted_subgraph_loss_coef']
            tag += f'_penalizeSimilar{p_swap}X{coef}'  
        if optimization_kwargs['use_pcgrad']==True:
            tag+='_pcgrad'                    
    if optimization_kwargs['use_sam']==True:
        tag += f'_sam_mom{optimization_kwargs['sam_momentum']}_rho{optimization_kwargs['sam_rho']}'
    return tag

def get_regression_tag():
    regression_kwargs = config.regression_kwargs
    lambda_ = regression_kwargs['lambda']
    regression_tag = f'regressionLambda{lambda_}'
    return regression_tag

def get_subgraph_tag(dataset_name):
    subgraph_kwargs = config.subgraph_kwargs
    subgraph_tag = ''
    sub_size_as_fraction = subgraph_kwargs['subgraph_size_as_fraction']
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
            sub_size_as_fraction = np.round((len(nodeIndices) / num_nodes)/numSubgraphs, 5)
        numHops = khop_kwargs['numHops']
        max_degree = khop_kwargs['max_degree']
        subgraph_tag = f'{method}{numHops}_sub_size_as_fraction{sub_size_as_fraction}_numSubgraphs{numSubgraphs}_maxDegree{max_degree}'

    elif method == 'random':
        subgraph_tag = f'{method}_sub_size_as_fraction{sub_size_as_fraction}_numSubgraphs{numSubgraphs}'
    elif method == 'random_walk_with_restart':
        rwr_kwargs = subgraph_kwargs['rwr_kwargs']
        restart_prob = rwr_kwargs['restart_prob']
        max_steps = rwr_kwargs['max_steps']
        subgraph_tag = f'{method}_sub_size_as_fraction{sub_size_as_fraction}_numSubgraphs{numSubgraphs}_restart_prob{restart_prob}_maxSteps{max_steps}'
    return subgraph_tag

def get_watermark_loss_tag():#watermark_loss_kwargs):
    watermark_loss_kwargs = config.watermark_loss_kwargs
    tag = f'eps{watermark_loss_kwargs['epsilon']}'
    return tag

def get_watermark_tag(#watermark_kwargs, 
                      dataset_name):
    watermark_kwargs = config.watermark_kwargs
    single_or_multi_graph = dataset_attributes[dataset_name]['single_or_multi_graph']
    pGraphs = watermark_kwargs['pGraphs']
    wmk_tag = ''

    if single_or_multi_graph == 'multi':
        wmk_tag = f'pGraphs{pGraphs}_' + wmk_tag
    percent_wmk = watermark_kwargs['percent_of_features_to_watermark']
    
    if len(wmk_tag)>0:
        wmk_tag += f'_{np.round(percent_wmk,2)}pct'
    else:
        wmk_tag += f'{np.round(percent_wmk,2)}pct'


    if watermark_kwargs['watermark_type']=='basic':
        wmk_tag +='BasicIndices'

    elif watermark_kwargs['watermark_type']=='unimportant':
        wmk_tag +='UnimptIndices'
        #strategy = watermark_kwargs['fancy_selection_kwargs']['selection_strategy'].capitalize()
        if watermark_kwargs['unimportant_selection_kwargs']['evaluate_individually']:
            handle_multiple = 'individualized'
        else:
            handle_multiple = watermark_kwargs['unimportant_selection_kwargs']['multi_subg_strategy']
        wmk_tag += f'_{handle_multiple}'
        wmk_tag += f'_{watermark_kwargs['unimportant_selection_kwargs']['clf_only_epochs']}ClfEpochs'

    elif watermark_kwargs['watermark_type']=='most_represented':
        wmk_tag +='MostRepIndices'

        #percent_wmk = np.round(100 * (1 - watermark_kwargs['basic_selection_kwargs']['p_remove']), 3)
    #    wmk_tag += f'_{percent_wmk}%BasicIndices'
    
    return wmk_tag

def get_seed_tag():
    seed_tag = 'seed'+str(config.seed)
    return seed_tag


def get_config_name(dataset_name):
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
    # model_tag = get_model_tag()#node_classifier_kwargs)
    # seed_tag = get_seed_tag()
    wmk_tag = get_watermark_tag(#watermark_kwargs, 
                                dataset_name)
    subgraph_tag = get_subgraph_tag(#subgraph_kwargs, 
                                    dataset_name)
    watermark_loss_tag = get_watermark_loss_tag()#watermark_loss_kwargs)
    augment_tag = get_augment_tag()#augment_kwargs)
    optimization_tag = get_optimization_tag()#optimization_kwargs)
    regression_tag = get_regression_tag()#regression_kwargs)


    # Combine All Tags into Config Name
    optimization_kwargs = config.optimization_kwargs
    clf_only = optimization_kwargs['clf_only']
    if clf_only==False:
        config_name = f'{wmk_tag}_{subgraph_tag}_{watermark_loss_tag}_{augment_tag}_{optimization_tag}_{regression_tag}'#_{seed_tag}'
    elif clf_only==True:
        config_name = f'clf_only_{augment_tag}_{optimization_tag}'#_{seed_tag}'
    return config_name

def get_results_folder_name(dataset_name):
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
    config_name = get_config_name(dataset_name)
    seed_tag = get_seed_tag()
    model_tag = get_model_tag()
    dataset_folder_name = os.path.join(results_dir, dataset_name)
    model_folder_name = os.path.join(dataset_folder_name, model_tag)
    model_folder_seed_version_name = os.path.join(model_folder_name, seed_tag)

    if os.path.exists(dataset_folder_name)==False:
        os.mkdir(dataset_folder_name)
    if os.path.exists(model_folder_name)==False:
        os.mkdir(model_folder_name)
    if os.path.exists(model_folder_seed_version_name)==False:
        os.mkdir(model_folder_seed_version_name)
    # if os.path.exists(os.path.join(dataset_folder_name, model_folder_name))==False:
    #     os.mkdir(os.path.join(dataset_folder_name, model_folder_name))
    # if os.path.exists(os.path.join(dataset_folder_name, model_folder_name,))==False:
    #     os.mkdir(os.path.join(dataset_folder_name, model_folder_name))

    return model_folder_seed_version_name #os.path.join(dataset_folder_name, model_folder_name, config_name, seed_tag)

def item_not_in_any_list(item, list_of_lists):
    for sublist in list_of_lists:
        if item in sublist:
            return False
    return True


def save_results(dataset_name, node_classifier, history, subgraph_dict=None, all_feature_importances=None, all_watermark_indices=None):#, probas=None):
    results_folder_name = get_results_folder_name(dataset_name)
    if os.path.exists(results_folder_name)==False:
        os.mkdir(results_folder_name)
    config_dict = {'node_classifier_kwargs':config.node_classifier_kwargs,
                'optimization_kwargs':config.optimization_kwargs,
                'watermark_kwargs':config.watermark_kwargs,
                'subgraph_kwargs':config.subgraph_kwargs,
                'regression_kwargs':config.regression_kwargs,
                'watermark_loss_kwargs':config.watermark_loss_kwargs,
                'augment_kwargs':config.augment_kwargs,
                'seed':config.seed}
    for object_name, object in zip(['node_classifier','history','subgraph_dict','all_feature_importances','all_watermark_indices',#'probas',
                                    'config_dict'],
                                [ node_classifier,  history,  subgraph_dict,  all_feature_importances,  all_watermark_indices,  #probas,  
                                 config_dict]):
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



def merge_kwargs_dicts():
    node_classifier_kwargs = config.node_classifier_kwargs
    optimization_kwargs = config.optimization_kwargs
    watermark_kwargs = config.watermark_kwargs
    subgraph_kwargs = config.subgraph_kwargs
    regression_kwargs = config.regression_kwargs
    watermark_loss_kwargs = config.watermark_loss_kwargs
    augment_kwargs = config.augment_kwargs

    merged_dict = {}

    # Add node_classifier_kwargs to merged_dict
    for k, v in node_classifier_kwargs.items():
        merged_dict[f"node_classifier_{k}"] = v
    
    # Add optimization_kwargs to merged_dict
    for k, v in optimization_kwargs.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                merged_dict[f"optimization_{k}_{sub_k}"] = sub_v
        else:
            merged_dict[f"optimization_{k}"] = v
    
    # Add watermark_kwargs to merged_dict
    for k, v in watermark_kwargs.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                merged_dict[f"watermark_{k}_{sub_k}"] = sub_v
        else:
            merged_dict[f"watermark_{k}"] = v
    
    # Add subgraph_kwargs to merged_dict
    for k, v in subgraph_kwargs.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                merged_dict[f"subgraph_{k}_{sub_k}"] = sub_v
        else:
            merged_dict[f"subgraph_{k}"] = v
    
    # Add regression_kwargs to merged_dict
    for k, v in regression_kwargs.items():
        merged_dict[f"regression_{k}"] = v
    
    # Add watermark_loss_kwargs to merged_dict
    for k, v in watermark_loss_kwargs.items():
        merged_dict[f"watermark_loss_{k}"] = v
    
    # Add augment_kwargs to merged_dict
    for k, v in augment_kwargs.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                merged_dict[f"augment_{k}_{sub_k}"] = sub_v
        else:
            merged_dict[f"augment_{k}"] = v
    
    return merged_dict


def wrap_title(title, width=40):
    return "\n".join(textwrap.wrap(title, width))


def update_seed(current_seed, max_value=10000):
    # Generate a pseudo-random value based on the current seed
    new_seed = ((current_seed*43+101)//17)%max_value
    return new_seed


def check_grads(node_classifier, epoch, tag='A'):
    grad_norm = 0
    for param in node_classifier.parameters():
        if param.grad is not None:
            grad_norm += param.grad.norm().item()
    print(f"Epoch {epoch} {tag}: Gradient norm = {grad_norm}")