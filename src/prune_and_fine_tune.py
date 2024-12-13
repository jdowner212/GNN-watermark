from embed_and_verify import *
from prune_and_fine_tune_utils import *

import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Test pruning settings')
    parser.add_argument('--dataset', type=str, default='PubMed',help='Dataset Name')
    parser.add_argument('--prune', action='store_true', help='Whether to test effectiveness after pruning.')
    parser.add_argument('--fine_tune', action='store_true', help='Whether to test effectiveness after fine tuning.')
    parser.add_argument('--seed', type=int, default=0, help='random_seed')
    parser.add_argument('--model_path',type=str,default=f'{parent_dir}/training_results/PubMed/archSAGE_elu_nLayers3_hDim256_drop0.1_skipTrue/6.0pctMostRepIndices_random_sub_size_as_fraction0.005_numSubgraphs5_eps0.1_nodeDropP0.1_nodeMixUp5_edgeDrop0.1_lr0.001_epochs200_coefWmk70_sacrifice1subNodes_pcgrad_regressionLambda0.1_seed4',help='Path to model')
    parser.add_arugment('--distribution_method', type=str, default='from_records')
    args = parser.parse_args()
    
    dataset_name = args.dataset
    parser.add_argument('--train_ratio',      type=float,   default=dataset_attributes[dataset_name]['train_ratio'],  help='Ratio of dataset comprising train set.')
    args, _ = parser.parse_known_args()
    val_ratio = test_ratio = (1-args.train_ratio)/2
    if dataset_attributes[dataset_name]['single_or_multi_graph']=='single':
        dataset = prep_data(dataset_name=dataset_name, location='default', batch_size='default', transform_list='default',  train_val_test_split=[args.train_ratio, val_ratio, test_ratio], seed=args.seed,load=True,save=False)
        graph_to_watermark = data = dataset[0]
    elif dataset_attributes[dataset_name]['single_or_multi_graph']=='multi':
        [train_dataset, val_dataset, test_dataset], [train_loader, val_loader, test_loader] = prep_data(dataset_name=dataset_name, location='default', batch_size='default', transform_list='default', train_val_test_split=[args.train_ratio, val_ratio, test_ratio],load=True,save=False)
        graph_to_watermark = train_dataset[0]
    get_presets(dataset,dataset_name)

    config_dict = pickle.load(open(os.path.join(args.model_path,'config_dict'),'rb'))
    subgraph_dict = pickle.load(open(os.path.join(args.model_path,'subgraph_dict'),'rb'))
    subgraph_dict_not_watermarked, _ = collect_subgraphs_within_single_graph_for_watermarking(data, dataset_name, use_train_mask=True, subgraph_kwargs=config_dict['subgraph_kwargs'], not_watermarked=True, seed=2575)

    assert args.distribution_method in ['from_records','predict']
    if args.distribution_method=='from_records':
        distribution_records_file = os.path.join(args.model_path,'distribution.txt')
        with open(distribution_records_file,'r') as f:
            lines = f.readlines()
        f.close()
        mu_natural = float(lines.split('\n')[0].split('mu_natural=')[1].split(',')[0])
        sigma_natural = float(lines.split('\n')[0].split('sigma_natural=')[1])
    elif args.distribution_method=='predict':
        mu_natural, sigma_natural = get_natural_match_distribution(data.x.shape[1], len(subgraph_dict))

    if args.prune==True:
        run_prune(data, mu_natural, sigma_natural, args.model_path, subgraph_dict, subgraph_dict_not_watermarked, config_dict, args.seed, 
                  pruning_type='structured',
                  target_confidence=0.99)

    if args.fine_tune==True:
        run_fine_tune(data, mu_natural, sigma_natural , args.model_path, subgraph_dict, subgraph_dict_not_watermarked, config_dict, args.seed, target_confidence=0.99)
