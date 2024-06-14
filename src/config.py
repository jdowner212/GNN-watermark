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

        watermark_kwargs        = {'coefWmk': 1, 'coefWmk_star': 1, 'restrict_watermark_to_present_features':False,     
                                    'pGraphs': 1, 'p_remove':0.75, 
                                    'clf_only_epochs':100, 
                                    'unimportant_percentile':20, 
                                    'subset_indices':False,
                                    'selection_kwargs': {'percent_of_features_to_watermark':100,
                                                         'personalized_indices':False,
                                                         'strategy': 'unimportant_features',
                                                         'merge_betas_method': 'average'
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

        watermark_kwargs        = {'coefWmk': 1, 'coefWmk_star': 1, 'restrict_watermark_to_present_features':False,     
                                   'pGraphs': 1, 'p_remove':0.75, 'clf_only_epochs':100, 'unimportant_percentile':20,
                                   'subset_indices':False,
                                   'selection_kwargs': {'percent_of_features_to_watermark':100,
                                                        'personalized_indices':False,
                                                        'strategy': 'unimportant_features',
                                                        'merge_betas_method': 'average'
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


def validate_watermark_kwargs(watermark_kwargs):
    watermark_kwargs        = {'coefWmk': 1, 
                               'coefWmk_star': 1, 
                               'restrict_watermark_to_present_features':False,     
                               'pGraphs': 1, 
                               'p_remove':0.75, 
                               'clf_only_epochs':100, 
                               'subset_indices':False,
                               'selection_kwargs': {'percent_of_features_to_watermark':100,
                                                    'personalized_indices':False,
                                                    'strategy': 'unimportant_features',
                                                    'merge_betas_method': 'average'
                                                }} 

    assert set(list(watermark_kwargs.keys()))=={'coefWmk', 'coefWmk_star', 'restrict_watermark_to_present_features', 'pGraphs', 'p_remove', 'clf_only_epochs', 'subset_indices', 'selection_kwargs'}
    assert watermark_kwargs['restrict_watermark_to_present_features'] in [True,False]
    assert watermark_kwargs['subset_indices'] in [True,False]
    assert set(list(watermark_kwargs['selection_kwargs'].keys()))=={'percent_of_features_to_watermark', 'personalized_indices', 'strategy', 'merge_betas_method'}
    assert watermark_kwargs['selection_kwargs']['strategy'] in ['unimportant_features','random']
    assert watermark_kwargs['selection_kwargs']['merge_betas_method'] in ['concat','average']