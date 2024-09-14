from eaaw_graphlime_utils import *
from prune_and_fine_tune_utils import *

import argparse
from config import *

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Test pruning settings')
    parser.add_argument('--dataset', type=str, default='PubMed',help='Dataset Name')
    parser.add_argument('--prune', action='store_true', help='Whether to test effectiveness after pruning.')
    parser.add_argument('--fine_tune', action='store_true', help='Whether to test effectiveness after fine tuning.')
    parser.add_argument('--distribution_method', type=str, default='from_records')
    parser.add_argument('--test_at_confidence', type=float, default=0.99, help='confidence level for testing results with prune/fine-tune')
    # parser.add_argument('--pruning_type', type=str, default='structured', help='Whether to apply "structured" or "unstructured" pruning.')
    parser.add_argument('--arch', type=str, default='GCN', help='GNN architecture type')
    parser.add_argument('--seed', type=int, default=0, help='random_seed')
    parser.add_argument('--numSubgraphs', type=int, default=None)
    parser.add_argument('--numLayers',type=int,default=None)
    parser.add_argument('--hDim',type=int,default=None)
    parser.add_argument('--epsilon',type=float,default=None)
    parser.add_argument('--coefWmk',type=float,default=None)
    parser.add_argument('--dropout',type=float,default=None)
    parser.add_argument('--epochs',type=int,default=None)
    parser.add_argument('--lr',type=float,default=None)



    args = parser.parse_args()
    print('args2:',args)


    # assert args.pruning_type in ['structured','unstructured']
    dataset_name = args.dataset
    train_ratio = dataset_attributes[dataset_name]['train_ratio']
    val_ratio = test_ratio = (1-train_ratio)/2
    saved_data_filename = f'load_this_dataset_trn_{train_ratio:.2f}_val_{val_ratio:2f}_test_{test_ratio:2f}.pkl'
    print('loading data:',saved_data_filename)
    data_path = os.path.join(parent_dir, 'data', dataset_name, saved_data_filename)
    dataset = pickle.load(open(data_path,'rb'))

    data = dataset[0]


    get_presets(dataset,dataset_name)
    if args.numSubgraphs==None:
        args.numSubgraphs = config.subgraph_kwargs['numSubgraphs']
    if args.numLayers==None:
        args.numLayers = config.node_classifier_kwargs['nLayers']
    if args.hDim==None:
        args.hDim= config.node_classifier_kwargs['hDim']
    if args.epsilon==None:
        args.epsilon = config.watermark_loss_kwargs['epsilon']
    if args.coefWmk==None:
        args.coefWmk = config.optimization_kwargs['coefWmk_kwargs']['coefWmk']
    if args.dropout==None:
        args.dropout = config.node_classifier_kwargs['dropout']
    if args.epochs==None:
        args.epochs = config.optimization_kwargs['epochs']
    if args.lr==None:
        args.lr = config.optimization_kwargs['lr']

    # args = parser.parse_args()

    config.subgraph_kwargs['numSubgraphs']=args.numSubgraphs
    config.node_classifier_kwargs['nLayers'] = args.numLayers
    config.node_classifier_kwargs['hDim'] = args.hDim
    config.watermark_loss_kwargs['epsilon'] = args.epsilon
    config.optimization_kwargs['coefWmk_kwargs']['coefWmk'] = args.coefWmk
    config.node_classifier_kwargs['dropout'] = args.dropout
    config.optimization_kwargs['epochs'] = args.epochs
    config.optimization_kwargs['lr'] = args.lr
    config.seed = args.seed

    assert args.distribution_method in ['from_records','predict']
    def get_fraction_from_folder_name(folder_name):
        return float(folder_name.split('fraction')[1].split('_')[0])
    def get_numSubgraphs_from_folder_name(folder_name):
        return int(folder_name.split('numSubgraphs')[1].split('_')[0])
    def get_nLayers_from_folder_name(folder_name):
        return int(folder_name.split('nLayers')[1].split('_')[0])
    def get_hDim_from_folder_name(folder_name):
        return int(folder_name.split('hDim')[1].split('_')[0])
    def get_eps_from_folder_name(folder_name):
        return float(folder_name.split('eps')[1].split('_')[0])
    def get_coefWmk_from_folder_name(folder_name):
        return float(folder_name.split('coefWmk')[1].split('_')[0])
    def get_dropout_from_folder_name(folder_name):
        return float(folder_name.split('drop')[1].split('_')[0])
    def get_epochs_from_folder_name(folder_name):
        return int(folder_name.split('epochs')[1].split('_')[0])
    def get_lr_from_folder_name(folder_name):
        return float(folder_name.split('lr')[1].split('_')[0])
    def get_fraction_from_folder_name(folder_name):
        return float(folder_name.split('fraction')[1].split('_')[0])



    data_training_path = f'{parent_dir}/training_results/{args.dataset}/'
    print('data_training_path:',data_training_path)
    print('os.listdir(data_training_path):',os.listdir(data_training_path) )
    print('args.arch:',args.arch)
    architecture_folder = [os.path.join(data_training_path, folder) for folder in os.listdir(data_training_path) if folder[0]!='.' and "clf_only" not in folder and args.arch in folder][0]
# for architecture_folder in architecture_folders:
    model_paths = [os.path.join(architecture_folder, folder) for folder in os.listdir(architecture_folder) if folder[0]!='.' and 'ignore' not in folder]# and get_coefWmk_from_folder_name(folder)==args.coefWmk and get_numSubgraphs_from_folder_name(folder)==args.numSubgraphs][0]

    try:
        model_path = [f for f in model_paths if \
                                                    get_numSubgraphs_from_folder_name(f)==args.numSubgraphs and \
                                                    get_nLayers_from_folder_name(f)==args.numLayers and \
                                                    get_hDim_from_folder_name(f)==args.hDim and \
                                                    get_eps_from_folder_name(f)==args.epsilon and \
                                                    get_coefWmk_from_folder_name(f)==args.coefWmk and \
                                                    get_dropout_from_folder_name(f)==args.dropout and \
                                                    get_epochs_from_folder_name(f)==args.epochs and \
                                                    get_lr_from_folder_name(f)==args.lr \
                                                    ][0]
    except:
        print('****')
        # print('model_paths:',model_paths)
        for m in model_paths:
            print('model path:',m)
            print('args.numSubgraphs==get_numSubgraphs_from_folder_name(f):',args.numSubgraphs,get_numSubgraphs_from_folder_name(m))
            print('args.numLayers==get_nLayers_from_folder_name(f):',args.numLayers,get_nLayers_from_folder_name(m))
            print('args.hDim==get_hDim_from_folder_name(f):',args.hDim,get_hDim_from_folder_name(m))
            print('args.epsilon==get_eps_from_folder_name(f):',args.epsilon,get_eps_from_folder_name(m))
            print('args.coefWmk==get_coefWmk_from_folder_name(f):',args.coefWmk,get_coefWmk_from_folder_name(m))
            print('args.dropout==get_dropout_from_folder_name(f):',args.dropout,get_dropout_from_folder_name(m))
            print('args.epochs==get_epochs_from_folder_name(f):',args.epochs,get_epochs_from_folder_name(m))
            print('args.lr==get_lr_from_folder_name(f):',args.lr,get_lr_from_folder_name(m))
    #print('model_paths:',model_paths)
#        for model_path in model_paths:
    assert os.path.exists(model_path)
#       seeds = [seed_name for seed_name in os.listdir(model_path) if seed_name[0]!='.' and '.png' not in seed_name and 'ignore' not in seed_name]
#        for seed in seeds:
    seed = args.seed
    full_model_path = os.path.join(model_path,f'seed{seed}')
    assert os.path.exists(os.path.join(full_model_path,'config_dict'))
    config_dict = pickle.load(open(os.path.join(full_model_path,'config_dict'),'rb'))
    subgraph_dict = pickle.load(open(os.path.join(full_model_path,'subgraph_dict'),'rb'))
    subgraph_dict_not_watermarked, _ = collect_subgraphs_within_single_graph_for_watermarking(data, dataset_name, use_train_mask=True, subgraph_kwargs=config_dict['subgraph_kwargs'], not_watermarked=True, seed=2575)

    Trainer_object = pickle.load(open(os.path.join(model_path,f'seed{seed}','Trainer'),'rb'))

    if args.distribution_method=='from_records':
        distribution_records_file = os.path.join(full_model_path,'distribution.txt')
        with open(distribution_records_file,'r') as f:
            lines = f.readlines()
        f.close()
        mu_natural = float(lines[1].split('mu_natural=')[1].split(',')[0])
        sigma_natural = float(lines[1].split('sigma_natural=')[1])
    elif args.distribution_method=='predict':
        mu_natural, sigma_natural = get_natural_match_distribution(data.x.shape[1], len(subgraph_dict))

    if args.prune==True:
        run_prune(Trainer_object, data, mu_natural, sigma_natural, full_model_path, subgraph_dict, subgraph_dict_not_watermarked, config_dict, args.seed,
                    pruning_type='structured',target_confidence=args.test_at_confidence)
        run_prune(Trainer_object, data, mu_natural, sigma_natural, full_model_path, subgraph_dict, subgraph_dict_not_watermarked, config_dict, args.seed,
                    pruning_type='unstructured',target_confidence=args.test_at_confidence)
    if args.fine_tune==True:
        run_fine_tune(Trainer_object, data, mu_natural, sigma_natural, full_model_path, subgraph_dict, subgraph_dict_not_watermarked, config_dict, args.seed,target_confidence=args.test_at_confidence)
