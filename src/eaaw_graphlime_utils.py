from   config import *
import copy
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
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
from transform_functions import *
from models import *
import os

torch.manual_seed(2)


import grafog.transforms as T



def give_subgraph_example(dataset_name, graph_to_watermark, numHops, compare_to_full=False, max_degree=None):
    if dataset_name=='PubMed':
        node_indices_to_watermark = [18745, 18728, 18809]
    else:
        ranked_nodes = rank_training_nodes_by_degree(dataset_name, graph_to_watermark, max_degree=max_degree)
        node_indices_to_watermark = ranked_nodes[:1]



    for node_index_to_watermark in node_indices_to_watermark:
        print(node_index_to_watermark)
        data_sub, _, subgraph_node_idx = generate_subgraph(graph_to_watermark, dataset_name, numHops, node_index_to_watermark=node_index_to_watermark, show=True)
        if compare_to_full==True:
            subgraph_node_idx, subgraph_edge_idx, _, _ = k_hop_subgraph(node_index_to_watermark, numHops, edge_index=graph_to_watermark.edge_index, num_nodes=graph_to_watermark.num_nodes, relabel_nodes=True)
            data_sub = Data(x=graph_to_watermark.x[subgraph_node_idx], edge_index=subgraph_edge_idx, y=graph_to_watermark.y[subgraph_node_idx])
            G_sub = to_networkx(data_sub, to_undirected=True)
            plt.figure(figsize=(5, 3))
            nx.draw_networkx(G_sub, with_labels=False,  node_color = 'blue', node_size=30)
            plt.title(f'{numHops}-hop subgraph centered at node {node_index_to_watermark} -- training mask not applied')
            plt.show()


def prep_data(dataset_name='CORA', 
              location='default', 
              batch_size='default',
              transform_list = 'default' #= NormalizeFeatures())
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
            transform_list.append(CreateMaskTransform(0.6, 0.2, 0.2))
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

def get_1_hop_edge_index(data, central_node, mask=None):
    edge_index_sub = torch.tensor([(n0,n1) for (n0,n1) in data.edge_index.T if n0==central_node or n1==central_node]).T
    if mask is not None:
        edge_index_sub =  torch.tensor([(n0,n1) for (n0,n1) in edge_index_sub.T if mask[n0].item()==True and mask[n1].item()==True]).T
    return edge_index_sub


def get_masked_subgraph_nodes(data, central_node, hops=2, mask=None):
    ''' 
    In some graphs, nodes are split train/val/test. This extracts the 
    subgraph nodes belonging to a certain subset, and *then* handing this
    these nodes to k_hop_subgraph to construct the result.
    '''
    seen_nodes = set()
    nodes_to_explore = set([central_node])
    subgraph_edge_list = []

    for hop in range(hops):
        nodes_to_explore_temp = set()  
        for this_central_node in nodes_to_explore:
            if this_central_node not in seen_nodes:
                seen_nodes.add(this_central_node)
                this_edge_index = get_1_hop_edge_index(data, this_central_node, mask=mask)
                for [n0, n1] in this_edge_index.T.tolist():
                    if [n0, n1] not in subgraph_edge_list and [n1, n0] not in subgraph_edge_list:
                        subgraph_edge_list.append([n0, n1])
                    if n0 != this_central_node and n0 not in seen_nodes:
                        nodes_to_explore_temp.add(n0)
                    if n1 != this_central_node and n1 not in seen_nodes:
                        nodes_to_explore_temp.add(n1)

        nodes_to_explore = nodes_to_explore.union(nodes_to_explore_temp)


    subgraph_edge_index = torch.tensor(subgraph_edge_list).T
    if hops==0 or len(subgraph_edge_list)==0:
        original_node_ids = torch.concat([torch.unique(subgraph_edge_index),torch.tensor([central_node])]).int()
    else:
        original_node_ids = torch.unique(subgraph_edge_index)
    original_node_ids, _ = torch.sort(original_node_ids)
    return original_node_ids


def create_random_subgraph(data, subgraph_size, mask=None, avoid_nodes=None, verbose=True):
        num_nodes = data.num_nodes
        num_selected_nodes = subgraph_size
        nodes_random_order = torch.randperm(num_nodes)
        if mask is not None:
            nodes_random_order = torch.tensor([n.item() for n in nodes_random_order if mask[n.item()] is not False])
        if avoid_nodes is not None:
            nodes_random_order = torch.tensor([n.item() for n in nodes_random_order if n not in avoid_nodes])
        selected_nodes = nodes_random_order[:num_selected_nodes]
        if verbose==True:
            print('selected_nodes:',selected_nodes)


        sub_edge_index, _ = subgraph(selected_nodes, data.edge_index, relabel_nodes=True, num_nodes=num_nodes)
        sub_data = Data(
            x=data.x[selected_nodes] if data.x is not None else None,
            edge_index=sub_edge_index,
            y=data.y[selected_nodes] if data.y is not None else None,
            train_mask=data.train_mask[selected_nodes] if data.train_mask is not None else None,
            test_mask=data.test_mask[selected_nodes] if data.test_mask is not None else None,
            val_mask=data.val_mask[selected_nodes] if data.val_mask is not None else None,
        )

        return sub_data, selected_nodes



def generate_subgraph(data, dataset_name, kwargs, central_node=None, avoid_nodes=[], use_train_mask=True, show=True):
    data = copy.deepcopy(data)
        
    if kwargs['method']=='khop':
        assert central_node is not None
        max_degree  = kwargs['khop_kwargs']['max_degree']
        numHops     = kwargs['khop_kwargs']['numHops']

        G = to_networkx(data, to_undirected=True)
        degrees = dict(nx.degree(G))
        if central_node is None:
            ranked_node_indices = rank_training_nodes_by_degree(dataset_name, data, max_degree)
            central_node = ranked_node_indices[0]
        if dataset_name in ['CORA','CiteSeer','PubMed','Reddit','Reddit2','CS','Flickr','computers','photo']:
            mask = torch.tensor([True]*len(data.x)) # initially, "mask" blocks nothing
            if use_train_mask is True:
                mask = copy.deepcopy(data.train_mask)
            if avoid_nodes is not None and len(avoid_nodes)>0:
                mask[avoid_nodes] = False
            subgraph_node_idx = get_masked_subgraph_nodes(data, central_node, hops=numHops, mask=mask)
            subgraph_node_idx, subgraph_edge_idx, _, _ = k_hop_subgraph(subgraph_node_idx, 0, edge_index=data.edge_index, num_nodes=data.num_nodes, relabel_nodes=True)
        elif dataset_name=='PPI':
            subgraph_node_idx, subgraph_edge_idx, _, _ = k_hop_subgraph(central_node, numHops, edge_index=data.edge_index, num_nodes=data.num_nodes, relabel_nodes=True)
        data_sub = Data(x=data.x[subgraph_node_idx], edge_index=subgraph_edge_idx, y=data.y[subgraph_node_idx])
        subgraph_signature = central_node
        
    elif kwargs['method']=='random':
        fraction      = kwargs['random_kwargs']['fraction']
        numSubgraphs = kwargs['random_kwargs']['numSubgraphs']
        num_watermarked_nodes = int(fraction*(data.x.shape[0]))
        subgraph_size = int(num_watermarked_nodes/numSubgraphs)

        data_sub, subgraph_node_idx = create_random_subgraph(data, subgraph_size, data.train_mask, avoid_nodes)
        subgraph_signature = '_'.join([str(s) for s in subgraph_node_idx.tolist()])


    if show==True:
        if kwargs['method']=='random':
            title = f'random {fraction}-fraction subgraph'
        elif kwargs['method']=='khop':
            title = f'{numHops}-hop subgraph centered at node {central_node} (degree={degrees[central_node]})'
        G_sub = to_networkx(data_sub, to_undirected=True)
        plt.figure(figsize=(5, 3))
        nx.draw_networkx(G_sub, with_labels=False,  node_color = 'blue', node_size=30)
        plt.title(title)
        plt.show()

    return data_sub, subgraph_signature, subgraph_node_idx



def generate_basic_watermark(k, p_neg_ones=0.5, p_remove=0.75):
    j = int(p_neg_ones*k)
    watermark = torch.ones(k)
    watermark_neg_1_indices = torch.randperm(k)[:j]
    watermark[watermark_neg_1_indices] = -1

    j_0 = int(p_remove*k)
    watermark_remove_indices = torch.randperm(k)[:j_0]
    watermark[watermark_remove_indices] = 0
    return watermark

def compute_kernel(x, reduce):
    ''' Gaussian (RBF) Kernel '''
    assert x.ndim == 2, x.shape
    n, d = x.shape

    if x.shape[0]==0:
       print('Computing gram matrix for single-node subgraph -- betas will be zero!')

    ## Graph Fourier Transform will result in all zeros unless you normalize for numerical stability:
    x = (x-torch.mean(x))/torch.std(x)
    ## 

    dist = x.reshape(1, n, d) - x.reshape(n, 1, d)  # (n, n, d)
    dist = dist ** 2
    if reduce:
        dist = torch.sum(dist, dim=-1, keepdim=True)
    std = np.sqrt(d)            
    K = torch.exp(-dist / (2 * std ** 2 * 0.1 + 1e-10))
    return K

def compute_gram_matrix(x):
    # Centers and normalizes
    if x.shape[0]==0:
        print('Computing gram matrix for single-node subgraph -- betas will be zero!')
    G = x - torch.mean(x, axis=0, keepdims=True)
    G = G - torch.mean(G, axis=1, keepdims=True)
    G_norm = torch.norm(G, p='fro', dim=(0, 1), keepdim=True) + 1e-10
    G = G / G_norm
    return G

def solve_regression(x,y, lambda_=1e-1):
    n, d = x.shape

    K       = compute_kernel(x, reduce=False)
    K_bar   = compute_gram_matrix(K).reshape(n ** 2, d)
    L       = compute_kernel(y, reduce=True)
    L_bar   = compute_gram_matrix(L).reshape(n ** 2,)
    KtK     = torch.matmul(K_bar.T,K_bar)
    I       = torch.eye(KtK.shape[0])
    KtL     = torch.matmul(K_bar.T,L_bar)
    beta    = torch.matmul((KtK+lambda_*I).inverse(),KtL)      
    beta    = beta.reshape(-1)
    return beta



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

def unpack_dict(dictionary,keys):
    return [dictionary[k] for k in keys]


def rank_training_nodes_by_degree(dataset_name, graph_to_watermark, max_degree=None):
    '''
    nodes_list may or may not equal all nodes in graph 
    (e.g., graph could be split into train/test/val sections)
    '''
    single_or_multi_graph = dataset_attributes[dataset_name]['single_or_multi_graph']
    assert single_or_multi_graph in ['single','multi'] # can remove later, just including at first to make sure code is correct
    if single_or_multi_graph=='single':
        nodes_list = torch.where(graph_to_watermark.train_mask==True)[0].tolist()
    else:
        nodes_list = list(range(len(graph_to_watermark.x)))
    G = to_networkx(graph_to_watermark, to_undirected=True)
    all_degrees = dict(nx.degree(G))
    if max_degree is None and max(list(all_degrees.values()))>200:
        print('Warning: building subgraph at central node with degree larger than 200. Consider setting "max_degree" argument of rank_training_nodes_by_degree() to a none-None value.')
    if max_degree is not None:
        all_degrees = {k:v for (k,v) in all_degrees.items() if v<=max_degree}
    these_degrees = {k:v for (k,v) in all_degrees.items() if k in nodes_list}
    these_degrees_sorted = {k: v for k, v in sorted(these_degrees.items(), key=lambda item: item[1], reverse=True)}
    ranked_node_indices = list(these_degrees_sorted.keys())
    return ranked_node_indices



def get_node_indices_to_watermark(dataset_name, graph_to_watermark, subgraph_kwargs):
    if subgraph_kwargs['khop_kwargs']['autoChooseSubGs']==True:
        assert subgraph_kwargs['khop_kwargs']['pNodes'] is not None
        p = subgraph_kwargs['khop_kwargs']['pNodes']
        random.seed(2575)
        num_watermarked_nodes = int(p*len(graph_to_watermark.train_mask.tolist()))
        print('num_watermarked_nodes:',num_watermarked_nodes)
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
    if os.path.exists(subgraph_data_dir)==False:
        os.mkdir(subgraph_data_dir)
    these_subgraphs_filename = None
    if subgraph_kwargs['method']=='khop':
        kwargs = subgraph_kwargs['khop_kwargs']
        [numHops, maxDegree, pNodes] = [kwargs[k] for k in ['numHops','max_degree','pNodes']]
        these_subgraphs_filename = f'khop_numHops{numHops}_maxDegree{maxDegree}_pNodes{pNodes}'
    elif subgraph_kwargs['method']=='random':
        kwargs = subgraph_kwargs['random_kwargs']
        [fraction, numSubgraphs] = [kwargs[k] for k in ['fraction','numSubgraphs']]
        these_subgraphs_filename = f'random_fraction{fraction}_numSubgraphs{numSubgraphs}'
    these_subgraphs_filepath = os.path.join(subgraph_data_dir,these_subgraphs_filename)


    try:
        subgraph_dict = pickle.load(open(these_subgraphs_filepath,'rb'))
    except:
        if subgraph_kwargs['method']=='khop':
            ''' this method relies on node_indices_to_watermark '''
            node_indices_to_watermark = get_node_indices_to_watermark(dataset_name, data, subgraph_kwargs)
            assert node_indices_to_watermark is not None
            enumerate_over_me = node_indices_to_watermark
        elif subgraph_kwargs['method']=='random':
            ''' this method relies on numSubgraphs '''
            numSubgraphs = subgraph_kwargs['random_kwargs']['numSubgraphs']
            assert numSubgraphs is not None
            enumerate_over_me = range(numSubgraphs)


        subgraph_dict = {}
        seen_nodes = []
        print("enumerate_over_me:",enumerate_over_me)
        for i, item in enumerate(enumerate_over_me):
            avoid_indices=seen_nodes
            print(f'Forming subgraph {i+1} of {numSubgraphs}',end='\r')
            if subgraph_kwargs['method']=='khop':
                central_node=item
                print("central_node:",central_node)
            else:
                central_node=None
            data_sub, subgraph_signature, subgraph_node_indices = generate_subgraph(data, dataset_name, subgraph_kwargs, central_node, avoid_indices, use_train_mask, show=False)
            subgraph_dict[subgraph_signature] = {'subgraph': data_sub, 'nodeIndices': subgraph_node_indices}
            seen_nodes += subgraph_node_indices.tolist()

        with open(these_subgraphs_filepath,'wb') as f:
            pickle.dump(subgraph_dict, f)

    return subgraph_dict



def get_results_folder_name(dataset_name, lr, epochs, node_classifier_kwargs, watermark_kwargs, subgraph_kwargs, augment_kwargs):
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
    # Model Tag Construction
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

    # Watermark Tag Construction
    single_or_multi_graph = dataset_attributes[dataset_name]['single_or_multi_graph']
    coef_wmk = watermark_kwargs['coefWmk']
    pGraphs = watermark_kwargs['pGraphs']
    wmk_tag = f'coefWmk{coef_wmk}'
    
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
    elif watermark_kwargs['watermark_type']=='basic':
        percent_wmk = np.round(100 * (1 - watermark_kwargs['basic_selection_kwargs']['p_remove']), 3)
        wmk_tag += f'_{percent_wmk}%BasicIndices'

    # Subgraph Tag Construction
    subgraph_tag = ''
    if subgraph_kwargs['method'] == 'khop':
        khop_kwargs = subgraph_kwargs['khop_kwargs']
        autoChooseSubGs = khop_kwargs['autoChooseSubGs']
        
        if autoChooseSubGs:
            pNodes = khop_kwargs['pNodes']
        else:
            nodeIndices = khop_kwargs['nodeIndices']
            num_nodes = dataset_attributes[dataset_name]['num_nodes']
            pNodes = np.round(len(nodeIndices) / num_nodes, 5)
        
        numHops = khop_kwargs['numHops']
        max_degree = khop_kwargs['max_degree']
        subgraph_tag = f'khop{numHops}_pNodes{pNodes}_maxDeg{max_degree}'

    elif subgraph_kwargs['method'] == 'random':
        fraction = subgraph_kwargs['random_kwargs']['fraction']
        numSubgraphs = subgraph_kwargs['random_kwargs']['numSubgraphs']
        subgraph_tag = f'random_fraction{fraction}_numSubgraphs{numSubgraphs}'

    # Augment Tag Construction
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

    # Combine All Tags into Config Name
    config_name = f'{model_tag}_{wmk_tag}_{subgraph_tag}_{augment_tag}_lr{lr}_epochs{epochs}'
    dataset_folder_name = os.path.join(results_dir, dataset_name)

    return os.path.join(dataset_folder_name, config_name)

    # Example usage (assuming you have the necessary configurations and directories defined):
    # folder_name = get_results_folder_name(dataset_name, lr, epochs, node_classifier_kwargs, watermark_kwargs, subgraph_kwargs, augment_kwargs)
    # print(folder_name)

def save_results(node_classifier, history, subgraph_dict, all_feature_importances, all_watermark_indices, probas, dataset_name, node_classifier_kwargs, watermark_kwargs, subgraph_kwargs,
                 lr, epochs, augment_kwargs):
    results_folder_name = get_results_folder_name(dataset_name, lr, epochs, node_classifier_kwargs, watermark_kwargs, subgraph_kwargs, augment_kwargs)
    if os.path.exists(results_folder_name)==False:
        os.mkdir(results_folder_name)
    for object_name, object in zip(['node_classifier','history','subgraph_dict','all_feature_importances','all_watermark_indices','probas'],
                                   [ node_classifier,  history,  subgraph_dict,  all_feature_importances,  all_watermark_indices,  probas]):
        with open(os.path.join(results_folder_name,object_name),'wb') as f:
            pickle.dump(object,f)
    print('Node classifier, history, subgraph dict, feature importances, watermark indices, and probas saved in:')
    print(results_folder_name)



def extract_results_random_subgraphs(data, dataset_name, fraction, numSubgraphs, alpha, watermark, probas, node_classifier, subgraph_kwargs, use_train_mask=False):
    if subgraph_kwargs['method']=='khop':
        num_nodes = data.x.shape[0]
        numSubgraphs = int(fraction*num_nodes)
        subgraph_kwargs['pNodes']=fraction
        node_indices_to_watermark = random.sample(list(range(num_nodes)),numSubgraphs) 
        subgraph_kwargs['khop_kwargs']['nodeIndices'] = node_indices_to_watermark
    elif subgraph_kwargs['method']=='random':
        subgraph_kwargs['random_kwargs']['fraction']=fraction
        subgraph_kwargs['random_kwargs']['numSubgraphs']=numSubgraphs

    subgraph_dict = collect_subgraphs_within_single_graph_for_watermarking(data, dataset_name, use_train_mask, subgraph_kwargs)
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

        omit_indices = get_omit_indices(x_sub, watermark,ignore_zeros_from_subgraphs=False)
        beta = process_beta(solve_regression(x_sub, y_sub), alpha, omit_indices)
        betas_dict[sig].append(beta.clone().detach())
        beta_similarities_dict[sig] = torch.sum(beta*watermark)

    return betas_dict, beta_similarities_dict

def get_complement_subgraph(data, subgraph_dict):
    all_node_indices = torch.cat([subgraph_dict[k]['nodeIndices'] for k in subgraph_dict]).tolist()
    complement_indices = torch.tensor([i for i in range(len(data.x)) if i not in all_node_indices])

    num_nodes = data.num_nodes
    comp_sub_edge_index, _ = subgraph(complement_indices, data.edge_index, relabel_nodes=True, num_nodes=num_nodes)
    data_comp_sub = Data(
        x=data.x[complement_indices] if data.x is not None else None,
        edge_index=comp_sub_edge_index,
        y=data.y[complement_indices] if data.y is not None else None,
        train_mask=data.train_mask[complement_indices] if data.train_mask is not None else None,
        test_mask=data.test_mask[complement_indices] if data.test_mask is not None else None,
        val_mask=data.val_mask[complement_indices] if data.val_mask is not None else None,
    )
    return data_comp_sub, complement_indices

def collect_augmentations(augment_kwargs, outDim):
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

def regress_on_subgraph(data, nodeIndices, probas):
    x_this_sub = data.x[nodeIndices]
    y_this_sub = probas[nodeIndices]
    beta_this_sub = solve_regression(x_this_sub, y_this_sub).clone().detach()
    return beta_this_sub

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



def item_not_in_any_list(item, list_of_lists):
    for sublist in list_of_lists:
        if item in sublist:
            return False
    return True



def name_compare_dict(dataset_name, watermark_kwargs):
    compare_dict_name = f"compare_dicts_{dataset_name}_{watermark_kwargs['watermark_type']}"
    if watermark_kwargs['watermark_type']=='fancy':
        compare_dict_name += f"{watermark_kwargs['fancy_selection_kwargs']['selection_strategy'].capitalize()}"
        if watermark_kwargs['fancy_selection_kwargs']['evaluate_individually']==True:
            compare_dict_name+='Individualized'
        else:
            if watermark_kwargs['fancy_selection_kwargs']['selection_strategy']!='random':
                compare_dict_name+=watermark_kwargs['fancy_selection_kwargs']['multi_subg_strategy'].capitalize()
    compare_dict_name += 'Indices'
    return compare_dict_name




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




def get_omit_indices(x_sub, watermark, ignore_zeros_from_subgraphs=True):
    if ignore_zeros_from_subgraphs==True:
        zero_features_within_subgraph = torch.where(torch.sum(x_sub, dim=0) == 0)
    else:
        zero_features_within_subgraph = torch.tensor([[]])
    zero_indices_within_watermark = torch.where(watermark==0)
    omit_indices = torch.tensor(list(set(zero_features_within_subgraph[0].tolist() + zero_indices_within_watermark[0].tolist())))
    return omit_indices

def process_beta(beta, tanh_alpha, omit_indices):
    beta = torch.tanh(tanh_alpha*beta)
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


def select_fancy_watermark_indices_individual(data, subgraph_dict, k, use_rand, use_unimpt, probas, num_indices):
    k = list(subgraph_dict.keys())[0]

    if use_rand:
        feature_importance, beta = [], []
        ordered_indices = torch.randperm(data.x.shape[1])

    elif use_unimpt:
        nodeIndices = subgraph_dict[k]['nodeIndices']
        beta = regress_on_subgraph(data, nodeIndices, probas)
        feature_importance = beta.abs()
        ordered_indices = sorted(range(data.x.shape[1]), key=lambda item: feature_importance[item])

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

def select_fancy_watermark_indices_shared(data, subgraph_dict, probas, num_indices, use_rand, use_unimpt, use_concat, use_avg):
    if use_rand:
        feature_importance, beta = [],[]
        ordered_indices = torch.randperm(data.x.shape[1])
    elif use_unimpt:
        if use_concat:
            all_indices = torch.concat([subgraph_dict[k]['nodeIndices'] for k in subgraph_dict.keys()])
            beta = regress_on_subgraph(data, all_indices, probas)
        elif use_avg:
            betas = []
            for k in subgraph_dict.keys():
                beta_this_sub = regress_on_subgraph(data, subgraph_dict[k]['nodeIndices'], probas)
                betas.append(beta_this_sub)
            beta = torch.mean(torch.vstack(betas),dim=0)
        feature_importance = beta.abs()
        ordered_indices = sorted(range(data.x.shape[1]), key=lambda item: feature_importance[item])

    all_zero_features = [torch.where(torch.sum(subgraph_dict[k]['subgraph'].x, dim=0) == 0)[0] for k in subgraph_dict.keys()]
    indices = []
    i=0
    while len(indices)<num_indices:
        if item_not_in_any_list(ordered_indices[i],all_zero_features):
            try:
                indices.append(ordered_indices[i].item())
            except:
                indices.append(ordered_indices[i])
        i +=1 
    all_indices             = [indices]*len(subgraph_dict)
    all_feature_importances = [feature_importance]*len(subgraph_dict)
    all_betas               = [beta]*len(subgraph_dict)

    return all_indices, all_feature_importances, all_betas

def select_fancy_watermark_indices(watermark_kwargs, data, subgraph_dict, probas):
    use_unimpt, use_rand, use_concat, use_avg, use_indiv, num_subgraphs, num_indices, message = describe_selection_config(data, watermark_kwargs, subgraph_dict)
    if num_subgraphs==1:
        k = list(subgraph_dict.keys())[0]
        indices, feature_importance, beta = select_fancy_watermark_indices_individual(data, subgraph_dict, k, use_rand, use_unimpt, probas, num_indices)
        all_indices, all_betas, all_feature_importances = [[indices]], [[feature_importance]], [[beta]]

    elif num_subgraphs>1:
        if use_indiv:
            all_indices, all_feature_importances, all_betas = [],[],[]
            for k in subgraph_dict.keys():
                indices, feature_importance, beta = select_fancy_watermark_indices_individual(data, subgraph_dict, k, use_rand, use_unimpt, probas, num_indices)
                all_indices.append(indices)
                all_feature_importances.append(feature_importance)
                all_betas.append(beta)
        else:
            all_indices, all_feature_importances, all_betas = select_fancy_watermark_indices_shared(data, subgraph_dict, probas, num_indices, use_rand, use_unimpt, use_concat, use_avg)

    assert message is not None
    print(message)
    return all_indices, all_feature_importances, all_betas


def apply_basic_watermark(data, subgraph_dict, watermark_kwargs):
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

def apply_fancy_watermark(data, subgraph_dict, probas, watermark_kwargs):
    all_watermark_indices, all_feature_importances, _ = select_fancy_watermark_indices(watermark_kwargs, data, subgraph_dict, probas)
    
    u = len(all_watermark_indices[0])
    h1,h2,random_order = u//2, u-u//2, torch.randperm(u)
    nonzero_watermark_values = torch.tensor([1]*h1 + [-1]*h2)[random_order].float()
    for i, subgraph_sig in enumerate(subgraph_dict.keys()):
        this_watermark = torch.zeros(data.x.shape[1])
        watermarked_feature_indices = all_watermark_indices[i]
        this_watermark[watermarked_feature_indices]=nonzero_watermark_values
        subgraph_dict[subgraph_sig]['watermark']=this_watermark
    
    return subgraph_dict, all_watermark_indices, all_feature_importances

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


def initialize_training(data, node_classifier_kwargs, lr):
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

def setup_subgraph_dict(data, dataset_name, subgraph_kwargs, watermark_kwargs):
    subgraph_dict = collect_subgraphs_within_single_graph_for_watermarking(data, dataset_name, use_train_mask=True, subgraph_kwargs=subgraph_kwargs)
    subgraph_signatures = list(subgraph_dict.keys())
    if watermark_kwargs['watermark_type']=='basic':
        subgraph_dict = apply_basic_watermark(data, subgraph_dict, watermark_kwargs)
    return subgraph_dict, subgraph_signatures

def print_epoch_status(epoch, loss_primary, loss_watermark, beta_similarity, acc_trn, acc_val, condition_met):
    if condition_met:
        epoch_printout = f'Epoch: {epoch:3d}, loss_primary = {loss_primary:.3f}, loss_watermark = {loss_watermark:.3f}, B*W = {beta_similarity:.5f}, train acc = {acc_trn:.3f}, val acc = {acc_val:.3f}'
    else:
        epoch_printout = f'Epoch: {epoch:3d}, loss_primary = {loss_primary:.3f}, loss_watermark = n/a, B*W = n/a, train acc = {acc_trn:.3f}, val acc = {acc_val:.3f}'
    print(epoch_printout)


def optimize_watermark(loss_watermark, beta_similarity, x_sub, y_sub, this_watermark, betas_dict, beta_similarities_dict, sig, alpha, epoch_condition, ignore_zeros_from_subgraphs=False):
    ''' epoch condtion: epoch==epoch-1'''
    omit_indices        = get_omit_indices(x_sub, this_watermark,ignore_zeros_from_subgraphs=ignore_zeros_from_subgraphs)
    beta                = process_beta(solve_regression(x_sub, y_sub), alpha, omit_indices)
    one_minus_B_x_W     = get_one_minus_B_x_W(beta, this_watermark, omit_indices)
    this_loss_watermark = torch.sum(one_minus_B_x_W)/len(torch.where(beta!=0)[0])
    betas_dict[sig].append(beta.clone().detach())
    loss_watermark += this_loss_watermark
    beta_similarity += torch.sum(beta*this_watermark)
    if epoch_condition:
        beta_similarities_dict[sig]=torch.sum(beta*this_watermark)
    return loss_watermark, beta_similarity, betas_dict, beta_similarities_dict


def train(data, dataset_name, lr, epochs, node_classifier_kwargs, watermark_kwargs, subgraph_kwargs, augment_kwargs, save=True):

    validate_kwargs(node_classifier_kwargs, subgraph_kwargs, augment_kwargs, watermark_kwargs)
    
    node_classifier, optimizer                  = initialize_training(data, node_classifier_kwargs, lr)
    subgraph_dict, subgraph_signatures          = setup_subgraph_dict(data, dataset_name, subgraph_kwargs, watermark_kwargs)
    node_aug, edge_aug                          = collect_augmentations(augment_kwargs, node_classifier_kwargs['outDim'])
    history, betas_dict, beta_similarities_dict = setup_history(subgraph_signatures)

    all_feature_importances, all_watermark_indices, coef_wmk = None, None, watermark_kwargs['coefWmk']
    # beta = torch.zeros(data.x.shape[1],dtype=torch.float)
    for epoch in tqdm(range(epochs)):

        optimizer.zero_grad()

        edge_index, x, y    = augment_data(data, node_aug, edge_aug)
        log_logits          = node_classifier(x, edge_index)
        loss_primary        = F.nll_loss(log_logits[data.train_mask], y[data.train_mask])
        loss_watermark, beta_similarity = torch.tensor(0,dtype=float), torch.tensor(0,dtype=float)


        wmk_optimization_condtion_met = (watermark_kwargs['watermark_type']=='basic') or (watermark_kwargs['watermark_type']=='fancy' and epoch>=watermark_kwargs['clf_only_epochs']) 
        if not wmk_optimization_condtion_met:
            ''' if designing based on coefficients but the time hasn't come to optimize watermark, hold off '''
            loss = loss_primary
            loss.backward()

        elif wmk_optimization_condtion_met:
            ''' otherwise if not doing a fancy watermark or if the epoch has been reached, optimize watermark '''
            
            probas = log_logits.clone().exp()

            if watermark_kwargs['watermark_type']=='fancy' and epoch==watermark_kwargs['clf_only_epochs']:
                ''' Define watermark at subset of feature indices. If `False`, then watermark was previously-defined. '''
                subgraph_dict, all_watermark_indices, all_feature_importances = apply_fancy_watermark(data, subgraph_dict, probas, watermark_kwargs)

            for sig in subgraph_signatures:
                this_watermark, data_sub, subgraph_node_indices = [subgraph_dict[sig][k] for k in ['watermark','subgraph','nodeIndices']]
                x_sub, y_sub = data_sub.x, probas[subgraph_node_indices]

                loss_watermark, beta_similarity, betas_dict, beta_similarities_dict = optimize_watermark(loss_watermark, beta_similarity, x_sub, y_sub, this_watermark, betas_dict, beta_similarities_dict, sig, 1e3, epoch==epochs-1, ignore_zeros_from_subgraphs=False)
            
            loss_watermark /= len(subgraph_dict)
            beta_similarity /= len(subgraph_dict)
            loss=loss_primary+coef_wmk*loss_watermark
            loss.backward()

        optimizer.step()

        acc_trn = accuracy(log_logits[data.train_mask], y[data.train_mask])
        acc_val = accuracy(log_logits[data.val_mask],   y[data.val_mask])
        history = update_history_one_epoch(history, loss, loss_primary, loss_watermark, acc_trn, acc_val)
        print_epoch_status(epoch, loss_primary, loss_watermark, beta_similarity, acc_trn, acc_val, wmk_optimization_condtion_met)
    
    history['betas']=betas_dict
    history['beta_similarities'] = beta_similarities_dict

    if save==True:
        save_results(node_classifier, history, subgraph_dict, all_feature_importances, all_watermark_indices, probas, dataset_name, node_classifier_kwargs, watermark_kwargs, subgraph_kwargs, lr, epochs, augment_kwargs)
    return node_classifier, history, subgraph_dict, all_feature_importances, all_watermark_indices, probas





def update_history_one_epoch(history, loss, loss_primary, loss_watermark, acc_trn, acc_val):
    history['losses'].append(loss.clone().detach())
    history['losses_primary'].append(loss_primary.clone().detach())
    history['losses_watermark'].append(loss_watermark.clone().detach())
    history['train_accs'].append(acc_trn)
    history['val_accs'].append(acc_val)
    return history




def gather_random_subgraphs_for_testing(data, dataset_name, max_degrees_choices=[20,50,100],pNodes_choices=[0.005,0.001,0.01],frac_choices = [0.001,0.004,0.005,0.01], nHops_choices=[1,2,3]):
    subgraph_kwargs =   {'method': 'random',  
                                'khop_kwargs':   {'autoChooseSubGs': True,   'nodeIndices':  None,   'numHops': None,   'max_degree': None,   'pNodes': None},
                                'random_kwargs': {'fraction': None,         'numSubgraphs': 1}
                            }
    
    if dataset_name=='computers':
        nHops_choices = [1]
        max_degrees_choices = [20]
        pNodes_choices = [0.001]
        frac_choices = [0.001]


    use_train_mask=True
    num_ = 50
    avoid_indices = []
    subgraphs = []
    for i in range(num_):
        subgraph_kwargs['method'] = np.random.choice(['khop','random'])
        print(f'Forming subgraph {i+1} of {num_}: {subgraph_kwargs['method']}',end='\r')
        if subgraph_kwargs['method']=='khop':
            subgraph_kwargs['khop_kwargs']['numHops'] = np.random.choice(nHops_choices)
            maxDegree = subgraph_kwargs['khop_kwargs']['maxDegree'] = np.random.choice(max_degrees_choices)
            pNodes = subgraph_kwargs['khop_kwargs']['pNodes'] = np.random.choice(pNodes_choices)

            random.seed(2575)
            num_watermarked_nodes = int(pNodes*len(data.train_mask.tolist()))
            ranked_nodes = torch.tensor(rank_training_nodes_by_degree(dataset_name, data, max_degree=maxDegree)[:100])
            idxs = torch.randperm(len(ranked_nodes))
            ranked_nodes = ranked_nodes[idxs]
            node_indices_to_watermark = ranked_nodes[:num_watermarked_nodes]
            central_node = node_indices_to_watermark[0]


        if subgraph_kwargs['method']=='random':
            subgraph_kwargs['random_kwargs']['fraction'] = np.random.choice(frac_choices)
            central_node=None


        data_sub, _, subgraph_node_indices = generate_subgraph(data, dataset_name, subgraph_kwargs, central_node, avoid_indices, use_train_mask, show=False)
        subgraphs.append((data_sub,subgraph_node_indices))

        avoid_indices += subgraph_node_indices

    return subgraphs


def count_matches(features):
    num_matches = 0
    for i in range(features.shape[1]):
        if torch.all(features[:, i] == features[0, i]):
            num_matches += 1
    return num_matches