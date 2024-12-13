from eaaw_graphlime_utils import *
import argparse
import os
import random
import config

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

data_dir = os.path.join(parent_dir, 'data')
subgraph_dir = os.path.join(data_dir,'random_subgraphs')

# def get_largest_continuation_file(file_list, base_file):
#     continue_files = [file for file in file_list if "continuation_from_" in file]
#     if continue_files:
#         largest_file = max(continue_files, key=lambda x: int(re.search(r'continuation_from_(\d+)', x).group(1)))
#         return largest_file
#     return base_file


try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except:
    current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)

# def count_beta_matches(dataset_name, c, subgraph_size_as_fraction,  model_folder, #arch_path='default',
#                      numLayers=3, hDim=256, dropout=0.1, skip=True, seed=0,save=True):
# def count_beta_matches(model_folder, c, subgraph_size_as_fraction, save=True, continuation=False, starting_epoch=0):

#     beta_path_filename = f'raw_betas_list_size_{subgraph_size_as_fraction}' if continuation==False else f'raw_betas_list_size_{subgraph_size_as_fraction}_continuation_from_{starting_epoch}'
#     beta_path = os.path.join(model_folder, beta_path_filename)
#     betas = pickle.load(open(beta_path,'rb'))
#     all_match_counts = []
#     for i in range(1000):
#         betas_ = random.sample(betas, c)
#         match_counts = count_matches(torch.sign(torch.vstack(betas_)))
#         all_match_counts.append(match_counts)
#         print(f'{i}/1000, {match_counts} matches',end='\r')
#     mu_natural = np.mean(all_match_counts)
#     sigma_natural = np.std(all_match_counts)
#     print('mu, sigma =', mu_natural, sigma_natural)
#     distribution_folder = 'distribution.txt' if continuation==False else f'distribution_continuation_from_{starting_epoch}.txt'
#     if save==True:
#         with open(os.path.join(model_folder,distribution_folder),'w') as f:
#             f.write(f'\nDistribution obtained by 1000 iterations of counting matches across {c} regression tensors: mu_natural={mu_natural},sigma_natural={sigma_natural}')
#         f.close()
#     with open(os.path.join(model_folder,'all_match_counts.pkl'),'wb') as f:
#         pickle.dump(all_match_counts,f)
#     return mu_natural, sigma_natural

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Gathering information about natural distributions of matching indices across regression results')
    parser.add_argument('--dataset_name', type=str, default='computers',help='Dataset Name')
    parser.add_argument('--create_random_subgraphs', action='store_true', help='whether to create/gather the random subgraphs (may have already happened in previous run)')
    parser.add_argument('--compute_regression_results', action='store_true', help='whether to compute match distribution')
    parser.add_argument('--get_match_distribution', action='store_true')
    parser.add_argument('--get_count_list', action='store_true')
    parser.add_argument('--num_iters', type=int, default=100, help='number of random subgraphs to compute')
    parser.add_argument('--subgraph_size', type=float,  default=0.005, help='List of values representing subgraph size (as fraction of the training data).')
    parser.add_argument('--seed',  type=int,  default=0, help='Seed')
    parser.add_argument('--continuation',  action='store_true',  help='Whether or not to use model continued training after the first failed')
    parser.add_argument('--starting_epoch',  type=int, default=0,  help='Epoch at which training continued, if applicable')


    args, _ = parser.parse_known_args()

    ### load data
    dataset_name = args.dataset_name
    train_ratio = dataset_attributes[dataset_name]['train_ratio']
    val_ratio = test_ratio = (1-train_ratio)/2
    saved_data_filename = f'load_this_dataset_trn_{train_ratio:.2f}_val_{val_ratio:2f}_test_{test_ratio:2f}.pkl'
    print('loading data:',saved_data_filename)
    data_path = os.path.join(parent_dir, 'data', dataset_name, saved_data_filename)
    dataset = pickle.load(open(data_path,'rb'))
    
    data = dataset[0]

    get_presets(dataset,dataset_name)
    parser.add_argument('--numSubgraphs', type=int, default=config.subgraph_kwargs['numSubgraphs'], help='corresponds to number of subgraphs model was trained on')
    parser.add_argument('--numLayers',type=int,default=config.node_classifier_kwargs['nLayers'],help='Number of layers associated with model architecture')
    parser.add_argument('--hDim',type=int,default=config.node_classifier_kwargs['hDim'],help='Number of hidden features associated with model architecture')
    parser.add_argument('--epsilon',type=float,default=config.watermark_loss_kwargs['epsilon'],help='wmk loss epsilon')    
    parser.add_argument('--coefWmk',type=float,default=config.optimization_kwargs['coefWmk_kwargs']['coefWmk'],help='coef wmk loss')
    parser.add_argument('--dropout',type=float,default=config.node_classifier_kwargs['dropout'],help='GNN dropout rate')
    parser.add_argument('--epochs',type=int,default=config.optimization_kwargs['epochs'],help='number of epochs')
    parser.add_argument('--lr',type=float,default=config.optimization_kwargs['lr'],help='learning rate')
    parser.add_argument('--arch',type=str,default=config.node_classifier_kwargs['arch'],help='Model architecture')

    args = parser.parse_args()

    if args.numSubgraphs==None:
        args.numSubgraphs = config.subgraph_kwargs['numSubgraphs']
    else:
        config.subgraph_kwargs['numSubgraphs'] = args.numSubgraphs
    if args.numLayers==None:
        args.numLayers = config.subgraph_kwargs['numSubgraphs']
    else:
        config.node_classifier_kwargs['nLayers'] = args.numLayers
    if args.hDim==None:
        args.hDim = config.node_classifier_kwargs['nLayers']
    else:
        config.node_classifier_kwargs['hDim'] = args.hDim
    if args.epsilon==None:
        args.epsilon = config.node_classifier_kwargs['hDim']
    else:
        config.watermark_loss_kwargs['epsilon'] = args.epsilon
    if args.coefWmk==None:
        args.coefWmk = config.watermark_loss_kwargs['epsilon']
    else:
        config.optimization_kwargs['coefWmk_kwargs']['coefWmk'] = args.coefWmk
    if args.dropout==None:
        args.dropout = config.optimization_kwargs['coefWmk_kwargs']['coefWmk']
    else:
        config.node_classifier_kwargs['dropout'] = args.dropout
    if args.epochs==None:
        args.epoch = config.optimization_kwargs['epochs']
    else:
        config.optimization_kwargs['epochs'] = args.epochs
    if args.lr==None:
        args.lr = config.optimization_kwargs['lr']
    else:
        config.optimization_kwargs['lr'] = args.lr
    if args.seed==None:
        args.seed = config.optimization_kwargs['lr']
    else:
        config.seed = args.seed
    if args.arch==None:
        args.arch = config.node_classifier_kwargs['arch']
    else:
        config.node_classifier_kwargs['arch'] = args.arch

    get_count_list = args.get_count_list



    ###
    def get_numSubgraphs_from_folder_name(folder_name):
        return int(folder_name.split('numSubgraphs')[1].split('_')[0])
    def get_nLayers_from_folder_name(folder_name):
        print('folder name:',folder_name)
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
    def get_arch_from_folder_name(folder_name):
        return folder_name.split('arch')[1].split('_')[0]
    
    subgraph_method = 'random'
    subgraph_size = args.subgraph_size
    train_folder = os.path.join(parent_dir, 'training_results',dataset_name)

    print('os.listdir(train_folder:',os.listdir(train_folder))
    arch_folder = []
    for f in os.listdir(train_folder):
        print('f:',f)
        if len(f)>3 and f[:4]=='arch':
            c1 = get_arch_from_folder_name(f)==args.arch
            c2 = get_nLayers_from_folder_name(f)==args.numLayers
            c3 = get_nLayers_from_folder_name(f)==args.numLayers
            if c1+c2+c3==3:
                arch_folder.append(f)
    arch_folder = arch_folder[0]
    arch_folder = os.path.join(train_folder, arch_folder)
    model_paths = [os.path.join(arch_folder,f) for f in os.listdir(arch_folder) if f[0]!='.' and 'ignore' not in f]

    try:
        model_folder = [f for f in model_paths if \
                                                    get_numSubgraphs_from_folder_name(f)==args.numSubgraphs and \
                                                    # get_hDim_from_folder_name(f)==args.hDim and \
                                                    get_eps_from_folder_name(f)==args.epsilon and \
                                                    get_coefWmk_from_folder_name(f)==args.coefWmk and \
                                                    get_dropout_from_folder_name(f)==args.dropout and \
                                                    get_epochs_from_folder_name(f)==args.epochs and \
                                                    get_lr_from_folder_name(f)==args.lr and \
                                                    get_fraction_from_folder_name(f)==subgraph_size #and \ 
                                                    ][0]
    except:
        print('****')
        print('model_paths:',model_paths)
        print('args.numSubgraphs==get_numSubgraphs_from_folder_name(f):',args.numSubgraphs,get_numSubgraphs_from_folder_name(model_paths[0]))
        print('args.numLayers==get_nLayers_from_folder_name(f):',args.numLayers,get_nLayers_from_folder_name(model_paths[0]))
        print('args.hDim==get_hDim_from_folder_name(f):',args.hDim,get_hDim_from_folder_name(model_paths[0]))
        print('args.epsilon==get_eps_from_folder_name(f):',args.epsilon,get_eps_from_folder_name(model_paths[0]))
        print('args.coefWmk==get_coefWmk_from_folder_name(f):',args.coefWmk,get_coefWmk_from_folder_name(model_paths[0]))
        print('args.dropout==get_dropout_from_folder_name(f):',args.dropout,get_dropout_from_folder_name(model_paths[0]))
        print('args.epochs==get_epochs_from_folder_name(f):',args.epochs,get_epochs_from_folder_name(model_paths[0]))
        print('args.lr==get_lr_from_folder_name(f):',args.lr,get_lr_from_folder_name(model_paths[0]))
        print('args.subgraph_size==get_fraction_from_folder_name(f):',subgraph_size,get_fraction_from_folder_name(model_paths[0]))

    # model_folder = get_results_folder_name(dataset_name)
    model_folder = os.path.join(arch_folder, model_folder, f'seed{args.seed}')
    watermarked_subgraph_dict_path = os.path.join(model_folder,'subgraph_dict')
    model_path = os.path.join(model_folder,'node_classifier')
    if args.continuation==True:
        watermarked_subgraph_dict_path = f'{watermarked_subgraph_dict_path}_continuation_from_{args.starting_epoch}'
        model_path = f'{model_path}_continuation_from_{args.starting_epoch}'

    print('model_path:',model_path)
        
    node_classifier = pickle.load(open(model_path,'rb'))

    this_subgraph_dir = os.path.join(subgraph_dir,dataset_name)
    
    results_filename = f'distribution_values_{dataset_name}_all_sizes.txt'
    results_path = os.path.join(model_folder, f'distribution_values_{dataset_name}_all_sizes.txt')

    watermarked_subgraph_dict = pickle.load(open(watermarked_subgraph_dict_path,'rb'))
    all_watermarked_sign_betas = []
    for i, k in enumerate(watermarked_subgraph_dict.keys()):
        node_classifier.eval()
        subgraph = watermarked_subgraph_dict[k]['subgraph']

        y_sub = node_classifier(subgraph.x, subgraph.edge_index, 0).clone().exp()
        raw_beta = solve_regression(subgraph.x, y_sub, config.regression_kwargs['lambda'])
        sign_beta = torch.sign(raw_beta)
        all_watermarked_sign_betas.append(raw_beta)       
    watermarked_match_counts = count_matches(torch.sign(torch.vstack(all_watermarked_sign_betas)))
    print('match count, watermarked betas:',watermarked_match_counts)



    ### gather subgraphs
    for size in [subgraph_size]:
        subgraph_dir_path = os.path.join(subgraph_dir, dataset_name, f'random_subgraphs_{dataset_name}_size_{size}.pkl')
        try:
            subgraphs_ = pickle.load(open(subgraph_dir_path,'rb'))
        except:
            subgraphs_=None
            
        if args.create_random_subgraphs==True or (args.create_random_subgraphs==False and subgraphs_==None):
            if (args.create_random_subgraphs==False and subgraphs_==None):
                print('Subgraphs dont exist yet -- gathering now')
            subgraphs_ = gather_random_subgraphs_not_wmk(data, dataset_name,  subgraph_method = subgraph_method, size = size, size_type='subgraph_size_as_fraction',seed=0, num_subgraphs=args.num_iters)
            with open(subgraph_dir_path,'wb') as f:
                pickle.dump(subgraphs_,f)

        compute_regression_results=args.compute_regression_results
        all_raw_betas=None
        raw_betas_files = [f for f in os.listdir(model_folder) if f'raw_betas_list_size_{size}' in f]

        beta_path_filename = f'raw_betas_list_size_{size}' if args.continuation==False else f'raw_betas_list_size_{size}_continuation_from_{args.starting_epoch}'
        beta_path = os.path.join(model_folder, beta_path_filename)
        if args.get_match_distribution==True and os.path.exists(beta_path)==False:
            ret_str = f'Havent yet computed raw betas -- doing that now.'
            print(ret_str)
            compute_regression_results=True
        if compute_regression_results==True:
            num_nodes = int(size*sum(data.train_mask))
            print(f'Size: {size} ({num_nodes} nodes)')
            ###
            node_classifier.eval()
            ###
            all_raw_betas = []
            p0s = []
            p1s = []
            pn1s = []
            for i, subgraph_dict in enumerate(subgraphs_):
                print(f'subgraph {i}/{len(subgraphs_)}')
                subgraph_ = subgraph_dict['subgraph']

                y_sub = node_classifier(subgraph_.x, subgraph_.edge_index, 0).clone().exp()
                raw_beta = solve_regression(subgraph_.x, y_sub, config.regression_kwargs['lambda'])
                sign_beta = torch.sign(raw_beta)
                p0s.append((sign_beta == 0).float().mean())
                p1s.append((sign_beta == 1).float().mean())
                pn1s.append((sign_beta == -1).float().mean())
                all_raw_betas.append(raw_beta)

            p0_avg = np.mean(p0s)
            p1_avg = np.mean(p1s)
            pn1_avg = np.mean(pn1s)
            with open(results_path, 'a') as f:
                f.write(f'\nSubgraph size {size} ({num_nodes} nodes):')
                f.write(f'\np0_avg:{p0_avg:.4f},p1_avg:{p1_avg:.4f},pn1_avg:{pn1_avg:.4f}')
            f.close()

            beta_path = os.path.join(model_folder, f'raw_betas_list_size_{size}')
            if args.continuation==True:
                beta_path = f'{beta_path}_continuation_from_{args.starting_epoch}'
            pickle.dump(all_raw_betas, open(beta_path, 'wb'))

        if args.get_match_distribution==True or args.get_count_list==True:
            numLayers=args.numLayers
            hDim=args.hDim
            dropout=config.node_classifier_kwargs['dropout']
            skip = config.node_classifier_kwargs['skip_connections']
            save=False if args.get_match_distribution==False else True
            mu_natural, sigma_natural = count_beta_matches(model_folder, args.numSubgraphs, args.subgraph_size, save=True, continuation=args.continuation,starting_epoch=args.starting_epoch)