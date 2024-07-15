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
    global node_classifier_kwargs, optimization_kwargs, watermark_kwargs, subgraph_kwargs, augment_kwargs, watermark_loss_kwargs, regression_kwargs

    node_classifier_kwargs  = {'arch': 'SAGE',  'activation': 'elu',        'nLayers':3,    'hDim':256, 
                                'dropout': 0,    'dropout_subgraphs': 0, 'skip_connections':True,    'heads_1':8,    'heads_2':1,    
                                'inDim': dataset.num_features,  
                                'outDim': dataset.num_classes} 
    
    optimization_kwargs = {'lr': 0.01,
                            'epochs': 200,
                            'sacrifice_kwargs':{'method':None,'percentage':None},
                            'clf_only':False,
                            'coefWmk_kwargs': {
                                               'coefWmk':1,
                                               'schedule_coef_wmk': False,
                                               'min_coefWmk_scheduled': 0,
                                               'reach_max_coef_wmk_by_epoch':100,
                                               },
                            # 'coefWmk': 1,
                            'regularization_type': None,
                            'lambda_l2': 0.01,
                            'use_pcgrad':False,
                            'use_sam':False,
                            'sam_momentum':0.5,
                            'sam_rho':0.005,
                            'use_gradnorm': False,
                            'gradnorm_alpha': 0.5,
                            'use_summary_beta':False,
                            'ignore_subgraph_neighbors':False,
                            'separate_forward_passes_per_subgraph':False}
    
    watermark_kwargs        = {'pGraphs': 1, 
                                'watermark_type':'fancy',
                                'basic_selection_kwargs': {'p_remove':0.75},
                                'fancy_selection_kwargs': {'percent_of_features_to_watermark':100,
                                                            'clf_only_epochs':100, 
                                                            'evaluate_individually':False,
                                                            'selection_strategy': 'unimportant',
                                                            'multi_subg_strategy': 'average'}}
    

    
    subgraph_kwargs         =   {'regenerate': False,
                                    'method': 'random',
                                    'fraction':0.003,
                                    'numSubgraphs': 1,
                                    'khop_kwargs':   {'autoChooseSubGs': True,   'nodeIndices':  None,   'numHops': 1,   'max_degree': 50},
                                    'random_kwargs': {},
                                    'rwr_kwargs': {'restart_prob':0.15, 'max_steps':1000}
                                }
    
    regression_kwargs = {'lambda': 0.1}

    watermark_loss_kwargs = {'epsilon': 0.001,'scale_beta_method': None,'balance_beta_weights':False,'alpha': None}

    
    augment_kwargs = {'separate_trainset_from_subgraphs':True, 
                      'ignore_subgraphs':True,
                      'nodeDrop':{'use':True,'p':0.45}, 
                      'nodeMixUp':{'use':True,'lambda':100},  
                      'nodeFeatMask':{'use':True,'p':0.2},    
                      'edgeDrop':{'use':True,'p':0.9}}
    
    if dataset_name=='default':
        pass # above settings are fine

    elif dataset_name=='Flickr':
        pass # default settings are fine
        

    elif dataset_name=='photo':
        optimization_kwargs['lr']=0.001
        optimization_kwargs['epochs']=100

        node_classifier_kwargs['arch']='GCN'
        watermark_kwargs['p_remove']=0

        augment_kwargs = {'nodeDrop': {'use': True, 'p': 0.3},
                          'nodeMixUp': {'use': True, 'lambda': 40},
                          'nodeFeatMask': {'use': True, 'p': 0.3},
                          'edgeDrop': {'use': True, 'p': 0.3}}
        subgraph_kwargs['fraction']=0.005
    
    elif dataset_name=='computers':
        node_classifier_kwargs['arch']='GCN'
        augment_kwargs['nodeDrop']['p']=0.35
        augment_kwargs['nodeMixUp']['lambda']=40
        augment_kwargs['edgeDrop']['p']=0.4
        watermark_loss_kwargs['epsilon']=0.01
        watermark_loss_kwargs['alpha']=None
        optimization_kwargs['lr']=0.002
        subgraph_kwargs['khop_kwargs']['max_degree']=40

    assert watermark_kwargs['fancy_selection_kwargs']['clf_only_epochs']<=optimization_kwargs['epochs']

#    return optimization_kwargs, node_classifier_kwargs, watermark_kwargs, subgraph_kwargs, augment_kwargs, watermark_loss_kwargs, regression_kwargs


def validate_regression_kwargs():#regression_kwargs):
    assert set(list(regression_kwargs.keys()))=={'lambda'}
    assert isinstance(regression_kwargs['lambda'],(int,float,np.integer,np.floating))
    assert regression_kwargs['lambda']>=0


def validate_optimization_kwargs():#optimization_kwargs):
    assert set(list(optimization_kwargs.keys()))=={'lr','epochs','sacrifice_kwargs','coefWmk_kwargs','clf_only','regularization_type','lambda_l2','use_pcgrad','use_sam','sam_momentum','sam_rho','use_gradnorm','gradnorm_alpha','use_summary_beta','ignore_subgraph_neighbors','separate_forward_passes_per_subgraph'}
    assert isinstance(optimization_kwargs['lr'],(int, float, np.integer, np.floating)) and optimization_kwargs['lr']>=0
    assert isinstance(optimization_kwargs['epochs'],int) and optimization_kwargs['epochs']>=0
    assert isinstance(optimization_kwargs['sacrifice_kwargs'],dict)
    assert set(list(optimization_kwargs['sacrifice_kwargs'].keys()))=={'method','percentage'}
    assert optimization_kwargs['sacrifice_kwargs']['method'] in [None,'subgraph_node_indices','train_node_indices']
    if optimization_kwargs['sacrifice_kwargs']['method'] is not None:
        assert isinstance(optimization_kwargs['sacrifice_kwargs']['percentage'],(int, float, np.integer, np.floating)) and optimization_kwargs['sacrifice_kwargs']['percentage']>=0 and optimization_kwargs['sacrifice_kwargs']['percentage']<=1
    assert isinstance(optimization_kwargs['coefWmk_kwargs'],dict)
    assert set(list(optimization_kwargs['coefWmk_kwargs'].keys()))=={'coefWmk','min_coefWmk_scheduled','schedule_coef_wmk','reach_max_coef_wmk_by_epoch'}
    assert isinstance(optimization_kwargs['coefWmk_kwargs']['coefWmk'],(int, float, np.integer, np.floating)) and optimization_kwargs['coefWmk_kwargs']['coefWmk']>=0
    assert isinstance(optimization_kwargs['coefWmk_kwargs']['min_coefWmk_scheduled'],(int, float, np.integer, np.floating)) and optimization_kwargs['coefWmk_kwargs']['min_coefWmk_scheduled']>=0
    assert isinstance(optimization_kwargs['coefWmk_kwargs']['schedule_coef_wmk'],bool)
    assert isinstance(optimization_kwargs['coefWmk_kwargs']['reach_max_coef_wmk_by_epoch'],int) and optimization_kwargs['coefWmk_kwargs']['reach_max_coef_wmk_by_epoch']>=0
    if optimization_kwargs['coefWmk_kwargs']['schedule_coef_wmk']==True:
        assert optimization_kwargs['coefWmk_kwargs']['coefWmk']>optimization_kwargs['coefWmk_kwargs']['min_coefWmk_scheduled']
        assert optimization_kwargs['coefWmk_kwargs']['reach_max_coef_wmk_by_epoch'] <= optimization_kwargs['epochs']

    assert isinstance(optimization_kwargs['clf_only'], bool)
    assert optimization_kwargs['regularization_type'] in [None, 'L2','beta_var']
    if optimization_kwargs['regularization_type']=='L2':
        assert isinstance(optimization_kwargs['lambda_l2'],(int, float, np.integer, np.floating))
        assert optimization_kwargs['lambda_l2']>=0
    assert isinstance(optimization_kwargs['use_pcgrad'],bool)
    assert isinstance(optimization_kwargs['use_sam'],bool)
    assert isinstance(optimization_kwargs['sam_momentum'],(int, float, np.integer, np.floating))
    assert isinstance(optimization_kwargs['sam_rho'],(int, float, np.integer, np.floating))
    assert isinstance(optimization_kwargs['use_gradnorm'],bool)
    assert isinstance(optimization_kwargs['gradnorm_alpha'],(int, float, np.integer, np.floating))
    assert optimization_kwargs['gradnorm_alpha']>=0
    assert isinstance(optimization_kwargs['use_summary_beta'],bool)
    assert isinstance(optimization_kwargs['ignore_subgraph_neighbors'],bool)
    assert isinstance(optimization_kwargs['separate_forward_passes_per_subgraph'],bool)
    if optimization_kwargs['sacrifice_kwargs']['method']=='train_node_indices' and optimization_kwargs['sacrifice_kwargs']['percentage']==1:
        assert optimization_kwargs['clf_only']==False  

def validate_node_classifier_kwargs():#node_classifier_kwargs):
    assert set(list(node_classifier_kwargs.keys()))=={'arch','activation','nLayers','hDim','dropout','dropout_subgraphs','skip_connections','heads_1','heads_2','inDim','outDim'}
    assert node_classifier_kwargs['arch'] in ['GAT','GCN','GraphConv','SAGE']
    assert isinstance(node_classifier_kwargs['nLayers'],int)
    assert isinstance(node_classifier_kwargs['inDim'],int)
    assert isinstance(node_classifier_kwargs['hDim'],int)
    assert isinstance(node_classifier_kwargs['outDim'],int)
    assert node_classifier_kwargs['dropout']>=0 and node_classifier_kwargs['dropout']<=1
    assert node_classifier_kwargs['dropout_subgraphs']>=0 and node_classifier_kwargs['dropout_subgraphs']<=1
    assert isinstance(node_classifier_kwargs['skip_connections'],bool)

def validate_subgraph_kwargs():#subgraph_kwargs):
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

def validate_augment_kwargs():#augment_kwargs):
    assert set(list(augment_kwargs.keys()))=={'separate_trainset_from_subgraphs', 'ignore_subgraphs',
                                                 'nodeDrop', 'nodeFeatMask','edgeDrop','nodeMixUp'}
    assert isinstance(augment_kwargs['separate_trainset_from_subgraphs'],bool)
    assert isinstance(augment_kwargs['ignore_subgraphs'],bool)
    for k in ['nodeDrop', 'nodeFeatMask','edgeDrop','nodeMixUp']:
        assert isinstance(augment_kwargs[k]['use'],bool)
        if k in ['nodeDrop','nodeFeatMask','edgeDrop']:
            assert set(list(augment_kwargs[k].keys()))=={'use','p'}
            assert augment_kwargs[k]['p'] >= 0 and augment_kwargs[k]['p'] <= 1 and isinstance(augment_kwargs[k]['p'], (int, float, np.integer, np.floating))
        elif k=='nodeMixUp':
            assert set(list(augment_kwargs[k].keys()))=={'use','lambda'}
            assert isinstance(augment_kwargs[k]['lambda'],int) or isinstance(augment_kwargs[k]['lambda'],float)

def validate_watermark_kwargs():#watermark_kwargs):
    assert set(list(watermark_kwargs.keys()))=={'pGraphs', 'watermark_type', 'basic_selection_kwargs', 'fancy_selection_kwargs'}
    assert watermark_kwargs['watermark_type'] in ['fancy','basic']
    if watermark_kwargs['watermark_type']=='basic':
        basic_kwargs = watermark_kwargs['basic_selection_kwargs']
        assert set(list(basic_kwargs.keys()))=={'p_remove'}
        assert isinstance(basic_kwargs['p_remove'], (int, float, np.integer, np.floating))
    if watermark_kwargs['watermark_type']=='fancy':
        fancy_kwargs = watermark_kwargs['fancy_selection_kwargs']
        assert set(list(fancy_kwargs.keys()))=={'percent_of_features_to_watermark', 'clf_only_epochs', 'evaluate_individually', 'selection_strategy', 'multi_subg_strategy'}
        assert fancy_kwargs['percent_of_features_to_watermark'] >= 0 and fancy_kwargs['percent_of_features_to_watermark'] <= 100 and isinstance(fancy_kwargs['percent_of_features_to_watermark'], (int, float, complex, np.integer, np.floating))
        assert isinstance(fancy_kwargs['clf_only_epochs'],int) and fancy_kwargs['clf_only_epochs']>=0
        assert isinstance(fancy_kwargs['evaluate_individually'],bool)
        assert fancy_kwargs['selection_strategy'] in ['unimportant','random']
        if fancy_kwargs['evaluate_individually']==False and fancy_kwargs['selection_strategy']!='random':
            assert fancy_kwargs['multi_subg_strategy'] in ['concat','average']


def validate_watermark_loss_kwargs():#watermark_loss_kwargs):
    assert set(list(watermark_loss_kwargs.keys()))=={'epsilon','balance_beta_weights','scale_beta_method','alpha'}
    assert isinstance(watermark_loss_kwargs['balance_beta_weights'],bool)
    assert isinstance(watermark_loss_kwargs['epsilon'],(int, float, np.integer, np.floating))
    assert watermark_loss_kwargs['epsilon']>=0
    assert watermark_loss_kwargs['scale_beta_method'] in [None, 'tanh','tan','clip']
    if watermark_loss_kwargs['scale_beta_method'] in ['tanh','tan']:
        assert isinstance(watermark_loss_kwargs['alpha'],(int, float, np.integer, np.floating))
        assert watermark_loss_kwargs['alpha']>=0


def validate_kwargs():#optimization_kwargs, node_classifier_kwargs, subgraph_kwargs, augment_kwargs, watermark_kwargs, watermark_loss_kwargs, regression_kwargs):
    # optimization_kwargs, node_classifier_kwargs, subgraph_kwargs, augment_kwargs, watermark_kwargs, watermark_loss_kwargs, regression_kwargs
    validate_regression_kwargs()#regression_kwargs)
    validate_optimization_kwargs()#optimization_kwargs)
    validate_node_classifier_kwargs()#node_classifier_kwargs)
    validate_subgraph_kwargs()#subgraph_kwargs)
    validate_augment_kwargs()#augment_kwargs)
    validate_watermark_kwargs()#watermark_kwargs)
    validate_watermark_loss_kwargs()#watermark_loss_kwargs)
    assert watermark_kwargs['fancy_selection_kwargs']['clf_only_epochs']<=optimization_kwargs['epochs']
