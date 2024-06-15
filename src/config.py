import numpy as np
from   torch_geometric.datasets import PPI, Reddit, Reddit2, Planetoid, Coauthor, Flickr, Amazon

root_dir = '/Users/janedowner/Desktop/Desktop/IDEAL/Project_2'
src_dir  = f'{root_dir}/src'
data_dir = f'{root_dir}/data'
results_dir  = f'{root_dir}/training_results'
compare_dicts_dir = f'{root_dir}/compare_dicts'

dataset_attributes = { 
    'CORA': {
        'single_or_multi_graph': 'single',
        'class': Planetoid,
        'num_classes':7,
        'num_nodes':2708
    },
    'CiteSeer': {
        'single_or_multi_graph': 'single',
        'class': Planetoid,
        'num_classes':6,
        'num_nodes':3327,
    },
    'PubMed': {
        'single_or_multi_graph': 'single',
        'class': Planetoid,
        'num_classes':3,
        'num_nodes':19717,
    },
    'CS': {
        'single_or_multi_graph': 'single',
        'class': Coauthor,
        'num_classes':3,
        'num_nodes':19717,
    },
    'Reddit': {
        'single_or_multi_graph': 'single',
        'class': Reddit,
        'num_classes': 41,
        'num_nodes': 232965
    },
    'Reddit2': {
        'single_or_multi_graph': 'single',
        'class': Reddit2,
        'num_classes': 41,
        'num_nodes': 232965
    },
    'Flickr': {
        'single_or_multi_graph': 'single',
        'class': Flickr,
        'num_classes':7,
        'num_nodes':89250
    },
    'computers': {
        'single_or_multi_graph': 'single',
        'class': Amazon,
        'num_classes':10,
        'num_nodes':13752
    },
    'photo': {
        'single_or_multi_graph': 'single',
        'class': Amazon,
        'num_classes':8,
        'num_nodes':7650
    },
    'PPI': {
        'single_or_multi_graph': 'multi',
        'class': PPI,
        'num_classes':121,
        'num_nodes':np.nan
    }
}



def get_presets(dataset, dataset_name):
    if dataset_name=='default':
        lr=0.01
        epochs=200

        node_classifier_kwargs  = {'arch': 'SAGE',  'activation': 'elu',        'nLayers':3,    'hDim':256, 
                                   'dropout': 0,    'skip_connections':True,    'heads_1':8,    'heads_2':1,    
                                   'inDim': dataset.num_features,  
                                   'outDim': dataset.num_classes} 

        watermark_kwargs        = {'coefWmk': 1, 
                                    'pGraphs': 1, 'p_remove':0.75, 
                                    'clf_only_epochs':100, 
                                    'subset_indices':False,
                                    'selection_kwargs': {'percent_of_features_to_watermark':100,
                                                         'evaluate_individually':False,
                                                         'selection_strategy': 'unimportant',
                                                         'multi_subg_strategy': 'average'
                                                         } 
                                    }
        

        subgraph_kwargs         =   {'method': 'random',  
                                     'khop_kwargs':   {'autoChooseSubGs': True,   'nodeIndices':  None,   'numHops': 1,   'max_degree': 50,   'pNodes': 0.005},
                                     'random_kwargs': {'fraction': 0.003,         'numSubgraphs': 1}
                                    }

        augment_kwargs = {'nodeDrop':{'use':True,'p':0.45}, 
                          'nodeMixUp':{'use':True,'lambda':100},  
                          'nodeFeatMask':{'use':True,'p':0.2},    
                          'edgeDrop':{'use':True,'p':0.9}}

    elif dataset_name=='Flickr':
        lr=0.01
        epochs=200

        node_classifier_kwargs  = {'arch': 'SAGE',  'activation': 'elu',        'nLayers':3,    'hDim':256, 
                                   'dropout': 0,    'skip_connections':True,    'heads_1':8,    'heads_2':1,    
                                   'inDim': dataset.num_features,  
                                   'outDim': dataset.num_classes} 

        watermark_kwargs        = {'coefWmk': 1, 
                                   '#restrict_watermark_to_present_features':False,     
                                   'pGraphs': 1, 'p_remove':0.75, 'clf_only_epochs':100, 
                                   'subset_indices':False,
                                   'selection_kwargs': {'percent_of_features_to_watermark':100,
                                                         'evaluate_individually':False,
                                                         'selection_strategy': 'unimportant',
                                                         'multi_subg_strategy': 'average'
                                                        }} 


        subgraph_kwargs         =   {'method': 'random',  
                                     'khop_kwargs':   {'autoChooseSubGs': True,   'nodeIndices':  None,   'numHops': 1,   'max_degree': 50,   'pNodes': 0.005},
                                     'random_kwargs': {'fraction': 0.003,         'numSubgraphs': 1}
                                    }
                    

        augment_kwargs = {'nodeDrop':{'use':True,'p':0.45}, 
                          'nodeMixUp':{'use':True,'lambda':100},  
                          'nodeFeatMask':{'use':True,'p':0.2},    
                          'edgeDrop':{'use':True,'p':0.9}}
        
    
    elif dataset_name=='photo':
        lr= 0.001
        epochs=100

        node_classifier_kwargs['arch']='GCN'
        node_classifier_kwargs['hDim']=256
        watermark_kwargs['p_remove']=0

        watermark_kwargs['restrict_watermark_to_present_features']=False
        augment_kwargs = {'nodeDrop': {'use': True, 'p': 0.3},
                          'nodeMixUp': {'use': True, 'lambda': 40},
                          'nodeFeatMask': {'use': True, 'p': 0.3},
                          'edgeDrop': {'use': True, 'p': 0.3}}
        subgraph_kwargs['random_kwargs']['fraction']=0.005


    assert watermark_kwargs['clf_only_epochs']<=epochs

    return lr, epochs, node_classifier_kwargs, watermark_kwargs, subgraph_kwargs, augment_kwargs



def validate_node_classifier_kwargs(node_classifier_kwargs):
    assert node_classifier_kwargs['arch'] in ['GAT','GCN','GraphConv','SAGE']
    assert isinstance(node_classifier_kwargs['nLayers'],int)
    assert isinstance(node_classifier_kwargs['inDim'],int)
    assert isinstance(node_classifier_kwargs['hDim'],int)
    assert isinstance(node_classifier_kwargs['outDim'],int)
    assert node_classifier_kwargs['dropout']>=0 and node_classifier_kwargs['dropout']<=1
    assert node_classifier_kwargs['skip_connections'] in [True,False]

def validate_subgraph_kwargs(subgraph_kwargs):
    assert set(list(subgraph_kwargs.keys()))=={'method','khop_kwargs','random_kwargs'}
    assert subgraph_kwargs['method'] in ['random','khop']
    if subgraph_kwargs['method']=='khop':
        khop_kwargs = subgraph_kwargs['khop_kwargs']
        assert set(list(khop_kwargs.keys()))=={'autoChooseSubGs','nodeIndices','numHops','max_degree','pNodes'}
        assert khop_kwargs['autoChooseSubGs'] in [True,False]
        if khop_kwargs['autoChooseSubGs']==False:
            assert khop_kwargs['nodeIndices'] is not None
        assert isinstance(khop_kwargs['numHops'],int)
        assert isinstance(khop_kwargs['max_degree'],int)
        assert khop_kwargs['pNodes']>0 and khop_kwargs['pNodes']<1
    elif subgraph_kwargs['method']=='random':
        random_kwargs = subgraph_kwargs['random_kwargs']
        assert set(list(random_kwargs.keys()))=={'fraction','numSubgraphs'}
        assert random_kwargs['fraction']>0 and random_kwargs['fraction']<1
        assert isinstance(random_kwargs['numSubgraphs'],int)

def validate_augment_kwargs(augment_kwargs):
    for k in augment_kwargs.keys():
        if k in ['nodeDrop','nodeFeatMask','edgeDrop']:
            assert set(list(augment_kwargs[k].keys()))=={'use','p'}
            assert augment_kwargs[k]['use'] in [True,False]
            assert augment_kwargs[k]['p'] >= 0 and augment_kwargs[k]['p'] <= 1
        elif k=='nodeMixUp':
            assert set(list(augment_kwargs[k].keys()))=={'use','lambda'}
            assert augment_kwargs[k]['use'] in [True,False]
            assert isinstance(augment_kwargs[k]['lambda'],int) or isinstance(augment_kwargs[k]['lambda'],float)

def validate_watermark_kwargs(watermark_kwargs):
    assert set(list(watermark_kwargs.keys()))=={'coefWmk', 'coefWmk_star', 'restrict_watermark_to_present_features', 'pGraphs', 'p_remove', 'clf_only_epochs', 'subset_indices', 'selection_kwargs'}
    assert watermark_kwargs['coefWmk']>0 and watermark_kwargs['coefWmk']<1
    assert isinstance(watermark_kwargs['clf_only_epochs'],int)
    assert watermark_kwargs['subset_indices'] in [True,False]
    assert set(list(watermark_kwargs['selection_kwargs'].keys()))=={'percent_of_features_to_watermark', 'selection_strategy', 'multi_subg_strategy'}
    assert watermark_kwargs['selection_kwargs']['percent_of_features_to_watermark'] >= 0 and watermark_kwargs['selection_kwargs']['percent_of_features_to_watermark'] <= 100
    assert watermark_kwargs['selection_kwargs']['evaluate_individually'] in [True, False]
    assert watermark_kwargs['selection_kwargs']['selection_strategy'] in ['unimportant','random']
    assert watermark_kwargs['selection_kwargs']['multi_subg_strategy'] in ['concat','average']