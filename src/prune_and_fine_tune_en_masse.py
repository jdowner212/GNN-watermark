from eaaw_graphlime_utils import *
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
    args = parser.parse_args()

    
    dataset_name = args.dataset
    if dataset_attributes[dataset_name]['single_or_multi_graph']=='single':
        dataset = prep_data(dataset_name=dataset_name, location='default', batch_size='default', transform_list='default',  train_val_test_split=[0.9,0.05,0.05], seed=args.seed)
        graph_to_watermark = data = dataset[0]
    elif dataset_attributes[dataset_name]['single_or_multi_graph']=='multi':
        [train_dataset, val_dataset, test_dataset], [train_loader, val_loader, test_loader] = prep_data(dataset_name=dataset_name, location='default', batch_size='default', transform_list='default', train_val_test_split=[0.9,0.05,0.05])
        graph_to_watermark = train_dataset[0]
    get_presets(dataset,dataset_name)


    data_training_path = f'{parent_dir}/training_results/{args.dataset}/'
    architecture_folders = [os.path.join(data_training_path, folder) for folder in os.listdir(data_training_path) if folder[0]!='.']
    for architecture_folder in architecture_folders:
        model_paths = [os.path.join(architecture_folder, folder) for folder in os.listdir(architecture_folder) if folder[0]!='.' ]
        for model_path in model_paths:
            assert os.path.exists(model_path)
            seeds = [seed_name for seed_name in os.listdir(model_path) if seed_name[0]!='.' and '.png' not in seed_name]
            for seed in seeds:
                full_model_path = os.path.join(model_path,seed)
                print('seed:',seed)
                print('path:',os.path.join(full_model_path,'config_dict'))
                assert os.path.exists(os.path.join(full_model_path,'config_dict'))
                config_dict = pickle.load(open(os.path.join(full_model_path,'config_dict'),'rb'))
                subgraph_dict = pickle.load(open(os.path.join(full_model_path,'subgraph_dict'),'rb'))
                subgraph_dict_not_watermarked, _ = collect_subgraphs_within_single_graph_for_watermarking(data, dataset_name, use_train_mask=True, subgraph_kwargs=config_dict['subgraph_kwargs'], not_watermarked=True, seed=2575)

                if args.prune==True:
                    run_prune(data, full_model_path, subgraph_dict, subgraph_dict_not_watermarked, config_dict, args.seed)

                if args.fine_tune==True:
                    run_fine_tune(data, full_model_path, subgraph_dict, subgraph_dict_not_watermarked, config_dict, args.seed)
