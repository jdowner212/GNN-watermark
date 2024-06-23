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
                                    'pGraphs': 1, 
                                    'clf_only_epochs':100, 
                                    'watermark_type':'fancy',
                                    'basic_selection_kwargs': {'p_remove':0.75},
                                    'fancy_selection_kwargs': {'percent_of_features_to_watermark':100,
                                                         'evaluate_individually':False,
                                                         'selection_strategy': 'unimportant',
                                                         'multi_subg_strategy': 'average'
                                                         } 
                                    }
        

        subgraph_kwargs         =   {'regenerate': False,
                                     'method': 'random',
                                     'fraction':0.003,
                                     'numSubgraphs': 1,
                                     'khop_kwargs':   {'autoChooseSubGs': True,   'nodeIndices':  None,   'numHops': 1,   'max_degree': 50},
                                     'random_kwargs': {},
                                     'rwr_kwargs': {'restart_prob':0.15, 'max_steps':1000}
                                    }
        
        watermark_loss_kwargs = {'epsilon': 0.001,
                                 'scale_beta_method': None,
                                 'balance_beta_weights':False,
                                 'alpha': None,
                                 'regularization_type': None,
                                 'lambda_l2': 0.01}

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
                                   'pGraphs': 1, 
                                   'clf_only_epochs':100, 
                                    'watermark_type':'basic',
                                    'basic_selection_kwargs': {'p_remove':0.75},
                                    'fancy_selection_kwargs': {'percent_of_features_to_watermark':100,
                                                         'evaluate_individually':False,
                                                         'selection_strategy': 'unimportant',
                                                         'multi_subg_strategy': 'average'
                                                        }} 

        watermark_loss_kwargs = {'epsilon': 0.001,
                                 'balance_beta_weights':False,
                                 'scale_beta_method': None,
                                 'alpha': None,
                                 'regularization_type': None,
                                 'lambda_l2': 0.01}

        subgraph_kwargs         =   {'regenerate':False,
                                     'method': 'random',  
                                     'fraction': 0.003,
                                     'numSubgraphs': 1,
                                     'khop_kwargs':   {'autoChooseSubGs': True,   'nodeIndices':  None,   'numHops': 1,   'max_degree': 50},
                                     'random_kwargs': {},
                                     'rwr_kwargs': {'restart_prob':0.15, 'max_steps':1000},

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

        augment_kwargs = {'nodeDrop': {'use': True, 'p': 0.3},
                          'nodeMixUp': {'use': True, 'lambda': 40},
                          'nodeFeatMask': {'use': True, 'p': 0.3},
                          'edgeDrop': {'use': True, 'p': 0.3}}
        subgraph_kwargs['fraction']=0.005


    assert watermark_kwargs['clf_only_epochs']<=epochs

    return lr, epochs, node_classifier_kwargs, watermark_kwargs, subgraph_kwargs, augment_kwargs, watermark_loss_kwargs






def validate_node_classifier_kwargs(node_classifier_kwargs):
    assert node_classifier_kwargs['arch'] in ['GAT','GCN','GraphConv','SAGE']
    assert isinstance(node_classifier_kwargs['nLayers'],int)
    assert isinstance(node_classifier_kwargs['inDim'],int)
    assert isinstance(node_classifier_kwargs['hDim'],int)
    assert isinstance(node_classifier_kwargs['outDim'],int)
    assert node_classifier_kwargs['dropout']>=0 and node_classifier_kwargs['dropout']<=1
    assert isinstance(node_classifier_kwargs['skip_connections'],bool)

def validate_subgraph_kwargs(subgraph_kwargs):
    assert set(list(subgraph_kwargs.keys()))=={'regenerate','method','numSubgraphs','fraction', 'khop_kwargs','random_kwargs','rwr_kwargs'}
    assert isinstance(subgraph_kwargs['regenerate'],bool)
    assert isinstance(subgraph_kwargs['numSubgraphs'],int)
    assert isinstance(subgraph_kwargs['fraction'], (int, float, np.integer, np.floating))
    assert subgraph_kwargs['fraction']>0 and subgraph_kwargs['fraction']<1
    assert subgraph_kwargs['method'] in ['random','khop','random_walk_with_restart']
    if subgraph_kwargs['method']=='khop':
        khop_kwargs = subgraph_kwargs['khop_kwargs']
        assert set(list(khop_kwargs.keys()))=={'autoChooseSubGs','nodeIndices','numHops','max_degree'}
        assert isinstance(khop_kwargs['autoChooseSubGs'], bool)
        if khop_kwargs['autoChooseSubGs']==False:
            assert khop_kwargs['nodeIndices'] is not None
        assert isinstance(khop_kwargs['numHops'],int)
        assert isinstance(khop_kwargs['max_degree'],int)
    elif subgraph_kwargs['method']=='random_walk_with_restart':
        rwr_kwargs = subgraph_kwargs['rwr_kwargs']                       
        assert set(list(rwr_kwargs.keys()))=={'restart_prob','max_steps'}
        assert isinstance(rwr_kwargs['restart_prob'],(int,float,np.integer,np.floating))
        assert rwr_kwargs['restart_prob']>=0 and rwr_kwargs['restart_prob']<=1
        assert isinstance(rwr_kwargs['max_steps'],(int))

def validate_augment_kwargs(augment_kwargs):
    for k in augment_kwargs.keys():
        assert isinstance(augment_kwargs[k]['use'],bool)
        if k in ['nodeDrop','nodeFeatMask','edgeDrop']:
            assert set(list(augment_kwargs[k].keys()))=={'use','p'}
            assert augment_kwargs[k]['p'] >= 0 and augment_kwargs[k]['p'] <= 1 and isinstance(augment_kwargs[k]['p'], (int, float, np.integer, np.floating))
        elif k=='nodeMixUp':
            assert set(list(augment_kwargs[k].keys()))=={'use','lambda'}
            assert isinstance(augment_kwargs[k]['lambda'],int) or isinstance(augment_kwargs[k]['lambda'],float)

def validate_watermark_kwargs(watermark_kwargs):
    assert set(list(watermark_kwargs.keys()))=={'coefWmk', 'pGraphs', 'clf_only_epochs', 'watermark_type', 'basic_selection_kwargs', 'fancy_selection_kwargs'}
    assert watermark_kwargs['coefWmk']>0 and isinstance(watermark_kwargs['coefWmk'], (int, float, np.integer, np.floating))
    assert isinstance(watermark_kwargs['clf_only_epochs'],int)
    assert watermark_kwargs['watermark_type'] in ['fancy','basic']
    if watermark_kwargs['watermark_type']=='basic':
        assert set(list(watermark_kwargs['watermark_type']['basic_selection_kwargs'].keys()))=={'p_remove'}
        assert isinstance(watermark_kwargs['watermark_type']['basic_selection_kwargs'], (int, float, np.integer, np.floating))
    if watermark_kwargs['watermark_type']=='fancy':
        assert watermark_kwargs['fancy_selection_kwargs']['percent_of_features_to_watermark'] >= 0 and watermark_kwargs['fancy_selection_kwargs']['percent_of_features_to_watermark'] <= 100 and isinstance(watermark_kwargs['fancy_selection_kwargs']['percent_of_features_to_watermark'], (int, float, complex, np.integer, np.floating))
        assert set(list(watermark_kwargs['fancy_selection_kwargs'].keys()))=={'percent_of_features_to_watermark', 'evaluate_individually', 'selection_strategy', 'multi_subg_strategy'}
        assert isinstance(watermark_kwargs['fancy_selection_kwargs']['evaluate_individually'],bool)
        assert watermark_kwargs['fancy_selection_kwargs']['selection_strategy'] in ['unimportant','random']
        if watermark_kwargs['fancy_selection_kwargs']['evaluate_individually']==False and watermark_kwargs['fancy_selection_kwargs']['selection_strategy']!='random':
            assert watermark_kwargs['fancy_selection_kwargs']['multi_subg_strategy'] in ['concat','average']


def validate_watermark_loss_kwargs(watermark_loss_kwargs):
    assert set(list(watermark_loss_kwargs.keys()))=={'epsilon','balance_beta_weights','scale_beta_method','alpha','regularization_type','lambda_l2'}
    assert(isinstance(watermark_loss_kwargs['balance_beta_weights'],bool))
    assert isinstance(watermark_loss_kwargs['epsilon'],(int, float, np.integer, np.floating))
    assert watermark_loss_kwargs['epsilon']>=0
    assert watermark_loss_kwargs['scale_beta_method'] in [None, 'tanh','tan','clip']
    if watermark_loss_kwargs['scale_beta_method'] in ['tanh','tan']:
        assert isinstance(watermark_loss_kwargs['alpha'],(int, float, np.integer, np.floating))
        assert watermark_loss_kwargs['alpha']>=0
    assert watermark_loss_kwargs['regularization_type'] in [None, 'L2','beta_var']
    if watermark_loss_kwargs['regularization_type']=='L2':
        assert isinstance(watermark_loss_kwargs['lambda_l2'],(int, float, np.integer, np.floating))
        assert watermark_loss_kwargs['lambda_l2']>=0

def validate_kwargs(node_classifier_kwargs, subgraph_kwargs, augment_kwargs, watermark_kwargs, watermark_loss_kwargs):
    validate_node_classifier_kwargs(node_classifier_kwargs)
    validate_subgraph_kwargs(subgraph_kwargs)
    validate_augment_kwargs(augment_kwargs)
    validate_watermark_kwargs(watermark_kwargs)
    validate_watermark_loss_kwargs(watermark_loss_kwargs)