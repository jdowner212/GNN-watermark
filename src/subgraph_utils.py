import ast
import config
import copy
import matplotlib.pyplot as plt
import networkx as nx
import os
import numpy as np 
import random
import torch



from torch_geometric.data import Data  
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx, k_hop_subgraph, subgraph
from torch_geometric.transforms import BaseTransform, NormalizeFeatures, Compose

from general_utils import *
from models import *
from transform_functions import *

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
    del G, nodes_list
    return ranked_node_indices

def determine_whether_to_increment_numHops(dataset_name,sub_size_as_fraction,numSubgraphs,numHops):
    file_name = os.path.join(data_dir,dataset_name,'subgraphs','must_increment_numHops.txt')
    if os.path.exists(file_name)==False:
        with open(file_name,'w') as f:
            pass
    with open(file_name,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = ast.literal_eval(line)
            if line[:3]==[sub_size_as_fraction,numSubgraphs,numHops]:
                f.close()
                return line[3]
        return numHops

def create_khop_subgraph(data, dataset_name, central_node, numHops, max_num_nodes, use_train_mask, avoid_nodes, seed):
    original_numHops = numHops
    if dataset_name in ['CORA','CiteSeer','PubMed','Reddit','Reddit2','CS','Flickr','computers','photo']:
        mask = torch.tensor([True]*len(data.x)) # initially, "mask" blocks nothing
        if use_train_mask is True:
            mask = copy.deepcopy(data.train_mask)
        if avoid_nodes is not None and len(avoid_nodes)>0:
            mask[avoid_nodes] = False
        subgraph_node_idx = get_masked_subgraph_nodes(data, central_node, hops=numHops, mask=mask)
        print('numHops:',numHops)
        print('len subgraph_node_idx:',len(subgraph_node_idx))
        print('max_num_nodes:',max_num_nodes)
        incremented=False
        can_break=False
        while len(subgraph_node_idx)<max_num_nodes and can_break==False:
            old_len = len(subgraph_node_idx)
            incremented=True
            print(f'len subgraph_node_idx={len(subgraph_node_idx)} but need {max_num_nodes}: adding 1 to numHop')
            numHops+=1
            subgraph_node_idx = get_masked_subgraph_nodes(data, central_node, hops=numHops, mask=mask)
            new_len = len(subgraph_node_idx)
            if old_len==new_len:
                numHops-=1
                can_break=True
                print(f'Incrementing numHops didn\'t add any more nodes -- stopped at numHops={numHops}.')
                print(f'Some subgraphs may be smaller than desired (this one is length {new_len}).')
        if max_num_nodes is not None:
            try:
                random.seed(seed)
                subgraph_node_idx = torch.tensor(random.sample(subgraph_node_idx.tolist(),max_num_nodes))
            except:
                print('max_num_nodes exceeds subgraph sizes; no need to cap number of nodes')
                print(f'num subgraph node indices: {subgraph_node_idx.shape}')
                print(f'max_num_nodes: {max_num_nodes}')
        if incremented==True:
            file_name = os.path.join(data_dir,dataset_name,'subgraphs','must_increment_numHops.txt')
            with open(file_name,'a+') as f:
                f.seek(0)
                # Read the content of the file
                file_content = f.read()
                if file_content:
                    f.seek(0, 2)  # Move the cursor to the end of the file
                # f.write("%s\n" % [config.subgraph_kwargs['fraction'],config.subgraph_kwargs['numSubgraphs'],original_numHops,numHops])
                f.write("%s\n" % [config.subgraph_kwargs['subgraph_size_as_fraction'],config.subgraph_kwargs['numSubgraphs'],original_numHops,numHops])
            f.close()



        subgraph_node_idx, subgraph_edge_idx, _, _ = k_hop_subgraph(subgraph_node_idx, 0, edge_index=data.edge_index, num_nodes=data.num_nodes, relabel_nodes=True)
    elif dataset_name=='PPI':
        subgraph_node_idx, subgraph_edge_idx, _, _ = k_hop_subgraph(central_node, numHops, edge_index=data.edge_index, num_nodes=data.num_nodes, relabel_nodes=True)
    data_sub = Data(x=data.x[subgraph_node_idx], edge_index=subgraph_edge_idx, y=data.y[subgraph_node_idx])
    return data_sub, subgraph_node_idx, numHops

def create_random_subgraph(data, subgraph_size, mask=None, avoid_nodes=None, verbose=False, seed=0):
    # torch.manual_seed(seed)
    # num_nodes = data.num_nodes
    # num_selected_nodes = subgraph_size
    # nodes_random_order = torch.randperm(num_nodes)


    # if mask is not None:
    #     nodes_random_order = torch.tensor([n.item() for n in nodes_random_order if mask[n.item()] is not False])
    # if avoid_nodes is not None:
    #     nodes_random_order = torch.tensor([n.item() for n in nodes_random_order if n not in avoid_nodes])
    # selected_nodes = nodes_random_order[:num_selected_nodes]
    # if verbose==True:
    #     print('selected_nodes:',selected_nodes)
    torch.manual_seed(seed)
    random.seed(seed)
    num_nodes = data.num_nodes

    # Convert mask and avoid_nodes to sets for efficient lookups
    mask_set = set(mask.nonzero(as_tuple=True)[0].tolist()) if mask is not None else set(range(num_nodes))
    avoid_nodes_set = set(avoid_nodes) if avoid_nodes is not None else set()

    # Eligible nodes are those in the mask and not in the avoid list
    eligible_nodes = list(mask_set - avoid_nodes_set)

    # Randomly sample nodes from the eligible set
    selected_nodes = torch.tensor(random.sample(eligible_nodes, min(subgraph_size, len(eligible_nodes))))

    if verbose:
        print('selected_nodes:', selected_nodes)

    sub_edge_index, _ = subgraph(selected_nodes, data.edge_index, relabel_nodes=True, num_nodes=num_nodes)
    sub_data = Data(
        x=data.x[selected_nodes] if data.x is not None else None,
        edge_index=sub_edge_index,
        y=data.y[selected_nodes] if data.y is not None else None,
        train_mask=data.train_mask[selected_nodes] if data.train_mask is not None else None,
        test_mask=data.test_mask[selected_nodes] if data.test_mask is not None else None,
        val_mask=data.val_mask[selected_nodes] if data.val_mask is not None else None)

    del eligible_nodes, avoid_nodes_set, sub_edge_index
    return sub_data, selected_nodes


def create_rwr_subgraph(data, start_node, restart_prob=0.15, subgraph_size=50, max_steps=1000, mask=None, avoid_nodes=None, seed=0):
    G = to_networkx(data, to_undirected=True)
    subgraph_nodes = set([start_node])
    current_node = start_node
    random.seed(seed)

    for _ in range(max_steps):
        try:
            current_node = current_node.item()
        except:
            current_node = current_node
        if len(subgraph_nodes) >= subgraph_size:
            break
        if random.random() < restart_prob:
            current_node = start_node
        else:
            neighbors = list(G.neighbors(current_node))
            if neighbors:
                current_node = random.choice(neighbors)
                if (mask is None or mask[current_node]) and (avoid_nodes is None or current_node not in avoid_nodes):
                    subgraph_nodes.add(current_node)
            else:
                current_node = start_node

    subgraph_node_idx = torch.tensor(list(subgraph_nodes))
    sub_edge_index, _ = subgraph(subgraph_node_idx, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    sub_data = Data(
        x=data.x[subgraph_node_idx] if data.x is not None else None,
        edge_index=sub_edge_index,
        y=data.y[subgraph_node_idx] if data.y is not None else None,
        train_mask=data.train_mask[subgraph_node_idx] if data.train_mask is not None else None,
        test_mask=data.test_mask[subgraph_node_idx] if data.test_mask is not None else None,
        val_mask=data.val_mask[subgraph_node_idx] if data.val_mask is not None else None,
    )

    return sub_data, subgraph_node_idx

def generate_subgraph(data, dataset_name, kwargs, central_node=None, avoid_nodes=[], use_train_mask=True, overrule_size_info=False, explicit_size_choice=10,show=True, seed=0):
    #data = copy.deepcopy(data)
    data = data.clone()
    sub_size_as_fraction = kwargs['subgraph_size_as_fraction']
    total_num_nodes = sum(data.train_mask)
    subgraph_size   = int(sub_size_as_fraction*total_num_nodes)

    G = None

    if overrule_size_info==True:
        assert isinstance(explicit_size_choice,int)
        assert explicit_size_choice<sum(data.train_mask)
        subgraph_size = explicit_size_choice

    if kwargs['method']=='khop':
        assert central_node is not None
        max_degree  = kwargs['khop_kwargs']['max_degree']
        numHops     = kwargs['khop_kwargs']['numHops']
        G = to_networkx(data, to_undirected=True)
        degrees = dict(nx.degree(G))
        if central_node is None:
            ranked_node_indices = rank_training_nodes_by_degree(dataset_name, data, max_degree)
            central_node = ranked_node_indices[0]
        data_sub, subgraph_node_idx, numHops = create_khop_subgraph(data, dataset_name, central_node, numHops, subgraph_size, use_train_mask, avoid_nodes, seed=seed)
        print(f"numHops inside generate_subgraph before update: {numHops}")
        kwargs['khop_kwargs']['numHops']=numHops # may have changed in create_khop_subgraph if the numHops was insufficient for desired subgraph size
        print(f"kwargs['khop_kwargs']['numHops'] inside generate_subgraph after update: {kwargs['khop_kwargs']['numHops']}")
        subgraph_signature = central_node
        del G
    elif kwargs['method']=='random':
        data_sub, subgraph_node_idx = create_random_subgraph(data, subgraph_size, data.train_mask, avoid_nodes, seed=seed)
        subgraph_signature = '_'.join([str(s) for s in subgraph_node_idx.tolist()])
    elif kwargs['method']=='random_walk_with_restart':
        assert central_node is not None
        restart_prob = kwargs['rwr_kwargs']['restart_prob']
        max_steps    = kwargs['rwr_kwargs']['max_steps']
        data_sub, subgraph_node_idx = create_rwr_subgraph(data, central_node, restart_prob=restart_prob, subgraph_size=subgraph_size, max_steps=max_steps, mask=data.train_mask, avoid_nodes=avoid_nodes, seed=seed)
        subgraph_signature = '_'.join([str(s) for s in subgraph_node_idx.tolist()])

    # print(f'Subgraph size: {len(subgraph_node_idx)} ({np.round(len(data_sub.x)/sum(data.train_mask),4)} of training data)')

    if show==True:
        if G==None:
            G=to_networkx(data, to_undirected=True)
        if kwargs['method']=='khop':
            title = f'{numHops}-hop subgraph centered at node {central_node} (degree={degrees[central_node]})'
        elif kwargs['method']=='random':
            # title = f'random {fraction}-fraction subgraph'
            title = f'random {sub_size_as_fraction}-fraction subgraph'
        elif kwargs['method']=='random_walk_with_restart':
            title = f'random walk w/ restart subgraph at node {central_node}, max steps={kwargs["rwr_kwargs"]["max_steps"]} and restart_prob={kwargs["rwr_kwargs"]["restart_prob"]}'
        G_sub = to_networkx(data_sub, to_undirected=True)
        plt.figure(figsize=(5, 3))
        nx.draw_networkx(G_sub, with_labels=False,  node_color = 'blue', node_size=30)
        plt.title(title)
        plt.show()
        del G_sub, G
    

    return data_sub, subgraph_signature, subgraph_node_idx




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

    del subgraph_edge_index, nodes_to_explore, subgraph_edge_list
    return original_node_ids

