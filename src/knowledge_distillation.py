from knowledge_distillation_utils import *
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Watermarking settings')
    parser.add_argument('--dataset', type=str, default='computers',help='Dataset Name')
    parser.add_argument('--seed',    type=int, default=0,  help='Random seed.')
    args, _ = parser.parse_known_args()
    dataset_name = args.dataset
    seed = args.seed
    load_data=True
    save_data=False

    parser.add_argument('--train_ratio', type=float, default=dataset_attributes[dataset_name]['train_ratio'],  help='Ratio of dataset comprising train set.')
    args, _ = parser.parse_known_args()
    val_ratio = test_ratio = (1-args.train_ratio)/2
    if dataset_attributes[dataset_name]['single_or_multi_graph']=='single':
        dataset = prep_data(dataset_name=dataset_name, location='default', batch_size='default', transform_list='default',  train_val_test_split=[args.train_ratio,val_ratio,test_ratio], seed=seed, save=save_data, load=load_data)
        graph_to_watermark = data = dataset[0]
    elif dataset_attributes[dataset_name]['single_or_multi_graph']=='multi':
        [train_dataset, val_dataset, test_dataset], [train_loader, val_loader, test_loader] = prep_data(dataset_name=dataset_name, location='default', batch_size='default', transform_list='default', train_val_test_split=[args.train_ratio,val_ratio,test_ratio])
        graph_to_watermark = train_dataset[0]


    get_presets(dataset,dataset_name)
    config.is_kd_attack=True
    parser.add_argument('--num_iters',                                        type=int,               default=1,                                                                                  help='Number of times to run the experiment, so we can obtain an average.')                                                                                       
    parser.add_argument('--confidence', type=float, default=0.999999, help='confidence value for recommending watermark size')
    parser.add_argument('--continuation',  action='store_true',  help='Whether or not to use model continued training after the first failed')
    parser.add_argument('--starting_epoch',  type=int, default=0,  help='Epoch at which training continued, if applicable')
    parser.add_argument('--KD_alpha',  type=float, default=0.5,  help='Epoch at which training continued, if applicable')
    parser.add_argument('--KD_temp',  type=float, default=1.0,  help='KD temperature. Higher smooths probabilities.')
    parser.add_argument('--kd_train_on_subgraphs',action='store_true')
    parser.add_argument('--get_p_val',   action='store_true',  help='Whether to compute significance of results')       
    parser.add_argument('--preserve_edges_between_subsets', action='store_true')
    parser.add_argument('--kd_subgraphs_only',action='store_true')
                                                                                                                                                
    ## teacher
    parser.add_argument('--arch',                                             type=str,               default=config.node_classifier_kwargs['arch'],                                              help='GNN architecture (GAT, GCN, GraphConv, SAGE).')
    parser.add_argument('--activation',                                       type=str,               default=config.node_classifier_kwargs['activation'],                                        help='relu or elu.')
    parser.add_argument('--nLayers',                                          type=int,               default=config.node_classifier_kwargs['nLayers'],                                           help='Number of layers in GNN.')
    parser.add_argument('--hDim',                                             type=int,               default=config.node_classifier_kwargs['hDim'],                                              help='Number of hidden dimensions in GNN.')
    parser.add_argument('--dropout',                                          type=float,             default=config.node_classifier_kwargs['dropout'],                                           help='Dropout rate for classification training.')
    parser.add_argument('--skip_connections',                                 type=str2bool,          default=config.node_classifier_kwargs['skip_connections'],                                  help='Whether to include skip connections in GNN architecture: True (yes, true, t, y, 1) or False (no, false, f, n, 0).')
    parser.add_argument('--heads_1',                                          type=int,               default=config.node_classifier_kwargs['heads_1'],                                           help='Number of attention heads to use in first layer of GAT architecture.')
    parser.add_argument('--heads_2',                                          type=int,               default=config.node_classifier_kwargs['heads_2'],                                           help='Number of attention heads to use in intermediate layers of GAT architecture.')
    parser.add_argument('--lr',                                           type=float,             default=config.optimization_kwargs['lr'],                                                   help='Learning rate.')
    parser.add_argument('--epochs',                                       type=int,               default=config.optimization_kwargs['epochs'],                                               help='Epochs.')
    parser.add_argument('--regularization_type',                          type=str,               default=config.optimization_kwargs['regularization_type'],                                  help='The regularization type to use during training.')
    parser.add_argument('--lambda_l2',                                    type=float,             default=config.optimization_kwargs['lambda_l2'],                                            help='If using L2 regularization, the lambda term.')
    parser.add_argument('--use_pcgrad',                                   type=str2bool,          default=config.optimization_kwargs['use_pcgrad'],                                           help='Whether to use PCGrad to help mitigate conflicting gradients from multi-task learning.')
    parser.add_argument('--use_sam',                                      type=str2bool,               default=config.optimization_kwargs['use_sam'],                                              help='Whether or not to use sharpness-aware minimization during training.')
    parser.add_argument('--sam_momentum',                                 type=float,               default=config.optimization_kwargs['sam_momentum'],                                         help='If using sharpness-aware minimization, the momentum value to use.')
    parser.add_argument('--sam_rho',                                      type=float,               default=config.optimization_kwargs['sam_rho'],                                              help='If using sharpness-aware minimization, the rho value to use.')
    
    parser.add_argument('--coefWmk',                                       type=float,             default = config.optimization_kwargs['coefWmk_kwargs']['coefWmk'],                          help='coefWmk of teacher model.')
    parser.add_argument('--percent_of_features_to_watermark',             type=float,             default=config.watermark_kwargs['percent_of_features_to_watermark'],                        help='The percentage of node features to watermark, if overriding automatic calculation.')
    parser.add_argument('--watermark_type',                               type=str,               default=config.watermark_kwargs['watermark_type'],                                          help='Watermark type ("unimportant" indices vs "most_represented" indices).')
    parser.add_argument('--subgraph_method',                                  type=str,               default=config.subgraph_kwargs['method'],                                                   help='Subgraph method (khop, random, rwr).')
    parser.add_argument('--subgraph_size_as_fraction',                                type=float,             default=config.subgraph_kwargs['subgraph_size_as_fraction'],                        help='Fraction of possible subgraph nodes comprising each watermarked subgraph.')
    parser.add_argument('--numSubgraphs',                                     type=int,               default=config.subgraph_kwargs['numSubgraphs'],                                             help='Number of subgraphs to watermark.')
    parser.add_argument('--subgraph_random_kwargs',                           type=dict,              default=config.subgraph_kwargs['random_kwargs'],                                            help='Empty dict -- no kwargs in current implementation.')
    parser.add_argument('--regression_lambda',                                type=float,             default=config.regression_kwargs['lambda'],                                                 help='The lambda value to use for regresion when getting GraphLIME explanation.')
    parser.add_argument('--watermark_loss_epsilon',                           type=float,             default=config.watermark_loss_kwargs['epsilon'],                                            help='Caps the influence of each nod feature index on the watermark loss. Smaller epislon = stricter cap.')
    parser.add_argument('--sacrifice_method',                             type=str,               default=config.optimization_kwargs['sacrifice_kwargs']['method'],                           help='If sacrificing some nodes from training, the method to use.')


    args, _ = parser.parse_known_args()
    config.kd_alpha=args.KD_alpha
    config.kd_temp=args.KD_temp
    config.kd_train_on_subgraphs = args.kd_train_on_subgraphs
    config.kd_subgraphs_only = args.kd_subgraphs_only
    config.node_classifier_kwargs['arch'] = args.arch
    config.node_classifier_kwargs['activation'] = args.activation
    config.node_classifier_kwargs['nLayers'] = args.nLayers
    config.node_classifier_kwargs['hDim'] = args.hDim
    config.node_classifier_kwargs['dropout'] = args.dropout
    config.node_classifier_kwargs['skip_connections'] = args.skip_connections
    config.node_classifier_kwargs['heads_1'] = args.heads_1
    config.node_classifier_kwargs['heads_2'] = args.heads_2
    config.optimization_kwargs['lr'] = args.lr
    config.optimization_kwargs['epochs'] = args.epochs
    config.optimization_kwargs['regularization_type'] = args.regularization_type
    config.optimization_kwargs['lambda_l2'] = args.lambda_l2
    config.optimization_kwargs['use_pcgrad'] = args.use_pcgrad
    config.optimization_kwargs['use_sam'] = args.use_sam
    config.optimization_kwargs['sam_momentum'] = args.sam_momentum
    config.optimization_kwargs['sam_rho'] = args.sam_rho
    config.optimization_kwargs['sacrifice_kwargs']['method']=args.sacrifice_method
    config.watermark_kwargs['percent_of_features_to_watermark'] = args.percent_of_features_to_watermark
    config.watermark_kwargs['watermark_type'] = args.watermark_type
    config.subgraph_kwargs['method'] = args.subgraph_method
    config.subgraph_kwargs['subgraph_size_as_fraction'] = args.subgraph_size_as_fraction
    config.subgraph_kwargs['numSubgraphs'] = args.numSubgraphs
    config.subgraph_kwargs['random_kwargs'] = args.subgraph_random_kwargs
    config.regression_kwargs['lambda'] = args.regression_lambda
    ## student
    parser.add_argument('--arch_student',               type=str,               default=args.arch,                       help='GNN architecture (GAT, GCN, GraphConv, SAGE).')
    parser.add_argument('--activation_student',         type=str,               default=args.activation,                 help='relu or elu.')
    parser.add_argument('--nLayers_student',            type=int,               default=args.nLayers-1,                  help='Number of layers in GNN.')
    parser.add_argument('--hDim_student',               type=int,               default=args.hDim//2,                    help='Number of hidden dimensions in GNN.')
    parser.add_argument('--dropout_student',            type=float,             default=args.dropout,                    help='Dropout rate for classification training.')
    parser.add_argument('--skip_connections_student',   type=str2bool,          default=args.skip_connections,           help='Whether to include skip connections in GNN architecture: True (yes, true, t, y, 1) or False (no, false, f, n, 0).')
    parser.add_argument('--heads_1_student',            type=int,               default=args.heads_1,                    help='Number of attention heads to use in first layer of GAT architecture.')
    parser.add_argument('--heads_2_student',            type=int,               default=args.heads_2,                    help='Number of attention heads to use in intermediate layers of GAT architecture.')
    parser.add_argument('--lr_student',                 type=float,             default=args.lr,                         help='Learning rate.')
    parser.add_argument('--epochs_student',             type=int,               default=args.epochs,                     help='Epochs.')
    parser.add_argument('--lambda_l2_student',                  type=float,             default=args.lambda_l2,    help='If using L2 regularization, the lambda term.')
    parser.add_argument('--use_pcgrad_student',                 type=str2bool,          default=args.use_pcgrad,   help='Whether to use PCGrad to help mitigate conflicting gradients from multi-task learning.')
    parser.add_argument('--use_sam_student',                    type=str2bool,          default=args.use_sam,      help='Whether or not to use sharpness-aware minimization during training.')
    parser.add_argument('--sam_momentum_student',               type=float,             default=args.sam_momentum, help='If using sharpness-aware minimization, the momentum value to use.')
    parser.add_argument('--sam_rho_student',                    type=float,             default=args.sam_rho,      help='If using sharpness-aware minimization, the rho value to use.')
    parser.add_argument('--sacrifice_method_student',          type=str,               default=args.sacrifice_method,  help='If sacrificing some nodes from training, the method to use.')
    args = parser.parse_args()
    config.KD_student_node_classifier_kwargs['arch']=args.arch_student
    config.KD_student_node_classifier_kwargs['activation']=args.activation_student
    config.KD_student_node_classifier_kwargs['nLayers']=args.nLayers_student
    config.KD_student_node_classifier_kwargs['hDim']=args.hDim_student
    config.KD_student_node_classifier_kwargs['dropout']=args.dropout_student
    config.KD_student_node_classifier_kwargs['skip_connections']=args.skip_connections_student
    config.KD_student_node_classifier_kwargs['heads_1']=args.heads_1_student
    config.KD_student_node_classifier_kwargs['heads_2']=args.heads_2_student
    config.KD_student_optimization_kwargs['lr']=args.lr_student
    config.KD_student_optimization_kwargs['epochs']=args.epochs_student
    config.KD_student_optimization_kwargs['lambda_l2']=args.lambda_l2_student
    config.KD_student_optimization_kwargs['use_pcgrad']=args.use_pcgrad_student
    config.KD_student_optimization_kwargs['use_sam']=args.use_sam_student
    config.KD_student_optimization_kwargs['sam_momentum']=args.sam_momentum_student
    config.KD_student_optimization_kwargs['sam_rho']=args.sam_rho_student
    config.KD_student_optimization_kwargs['sacrifice_kwargs']['method']=args.sacrifice_method_student
    config.preserve_edges_between_subsets=args.preserve_edges_between_subsets
    config.seed = args.seed

    # if args.wmk_size_auto_compute==True:
    n_features = data.x.shape[1]
    c = config.subgraph_kwargs['numSubgraphs']
    mu_natural, sigma_natural = get_natural_match_distribution(n_features, c)
    print('mu_natural:',mu_natural)
    print('sigma_natural:',sigma_natural)
    c_LB=args.confidence
    c_t=args.confidence
    recommended_watermark_length = find_min_n_uncertain(n_features, mu_natural, sigma_natural, c_LB, c_t, test_effective=True, verbose=True)
    recommended_percent = 100*recommended_watermark_length/n_features
    print(f'recommended_watermark_length for confidence={c_t}: {recommended_watermark_length}')
    print(f'recommended_percent: {recommended_percent:.3f}')
    config.watermark_kwargs['percent_of_features_to_watermark']=recommended_percent

    title_ = f'Separate Forward Passes -- {config.watermark_kwargs['watermark_type']} feature indices'
    title = f'{title_}.\n{config.subgraph_kwargs['numSubgraphs']} subgraphs.\nWatermarking {config.watermark_kwargs['percent_of_features_to_watermark']}% of node features'
    
    z_t=norm.ppf(args.confidence)
    target_number_matches = np.ceil(min(mu_natural +z_t*sigma_natural,data.x.shape[1]))

    data_original = copy.deepcopy(data)


    for _ in range(args.num_iters):
        original_results_folder_name = get_results_folder_name(dataset_name)
        print('original_results_folder_name:',original_results_folder_name)
        data_training_path = f'{parent_dir}/training_results/{args.dataset}/'
        architecture_folder = get_arch_folder(args,data_training_path)
        print('architecture_folder:',architecture_folder)
        teacher_model_path = get_model_path(args, architecture_folder, verbose=False)
        print('teacher_model_path:',teacher_model_path)
        assert os.path.exists(teacher_model_path)
        seed = args.seed
        full_teacher_model_path = os.path.join(teacher_model_path,f'seed{seed}')
        assert os.path.exists(os.path.join(full_teacher_model_path,'config_dict'))
        config_dict = pickle.load(open(os.path.join(full_teacher_model_path,'config_dict'),'rb'))
        subgraph_dict_filename = 'subgraph_dict' if args.continuation==False else f'subgraph_dict_continuation_from_{args.starting_epoch}'
        subgraph_dict = pickle.load(open(os.path.join(full_teacher_model_path,subgraph_dict_filename),'rb'))
        subgraph_dict_not_watermarked, _ = collect_subgraphs_within_single_graph_for_watermarking(data, dataset_name, use_train_mask=True, subgraph_kwargs=config_dict['subgraph_kwargs'], not_watermarked=True, seed=2575)

        teacher_Trainer_object_filename = 'Trainer' if args.continuation==False else f'Trainer_continuation_from_{args.starting_epoch}'
        teacher_Trainer_object = pickle.load(open(os.path.join(full_teacher_model_path,teacher_Trainer_object_filename),'rb'))
        teacher_node_classifier_filename = 'node_classifier' if args.continuation==False else f'node_classifier_continuation_from_{args.starting_epoch}'
        teacher_node_classifier = pickle.load(open(os.path.join(full_teacher_model_path,teacher_node_classifier_filename),'rb'))

        teacher_test_node_indices = teacher_Trainer_object.data.test_mask.nonzero(as_tuple=True)[0]
        teacher_val_node_indices = teacher_Trainer_object.data.val_mask.nonzero(as_tuple=True)[0]
        data = copy.deepcopy(data_original)
        student_node_classifier = Net(**config.KD_student_node_classifier_kwargs)
        Trainer_KD_ = Trainer_KD(student_node_classifier,teacher_node_classifier,data, dataset_name, subgraph_dict, subgraph_dict_not_watermarked, args.KD_alpha, args.KD_temp,
                                 data.train_mask,
                                 teacher_test_node_indices,
                                 teacher_val_node_indices)
        results = Trainer_KD_.train_KD(save=True, print_every=1, continuation=args.continuation, starting_epoch=args.starting_epoch)
        Trainer_KD_.test_watermark()
        results_folder_name = get_results_folder_name(dataset_name)


        if args.get_p_val==True:

            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)

            data_dir = os.path.join(parent_dir, 'data')
            subgraph_dir = os.path.join(data_dir,'random_subgraphs')
            this_subgraph_dir = os.path.join(subgraph_dir,dataset_name)

            model_path = get_model_path(args, architecture_folder)
            model_folder = os.path.join(architecture_folder, model_path, f'seed{args.seed}')

            results_filename = f'distribution_values_{dataset_name}_all_sizes_KD.txt'
            results_path = os.path.join(model_folder, f'distribution_values_{dataset_name}_all_sizes_KD.txt')

            all_watermarked_sign_betas = []
            for i, k in enumerate(subgraph_dict.keys()):
                student_node_classifier.eval()
                subgraph = subgraph_dict[k]['subgraph']
                y_sub = student_node_classifier(subgraph.x, subgraph.edge_index, 0).clone().exp()
                raw_beta = solve_regression(subgraph.x, y_sub, config.regression_kwargs['lambda'])
                sign_beta = torch.sign(raw_beta)
                all_watermarked_sign_betas.append(raw_beta)       
            watermarked_match_counts_with_zeros = count_matches(torch.sign(torch.vstack(all_watermarked_sign_betas)),ignore_zeros=False)
            watermarked_match_counts_without_zeros = count_matches(torch.sign(torch.vstack(all_watermarked_sign_betas)), ignore_zeros=True)                 

            create_random_subgraphs=False

            for size in [args.subgraph_size_as_fraction]:
                subgraph_dir_path = os.path.join(subgraph_dir, dataset_name, f'random_subgraphs_{dataset_name}_size_{size}.pkl')
                try:
                    subgraphs_ = pickle.load(open(subgraph_dir_path,'rb'))
                except:
                    subgraphs_=None
                    
                if create_random_subgraphs==True or (create_random_subgraphs==False and subgraphs_==None):
                    if (create_random_subgraphs==False and subgraphs_==None):
                        print('Subgraphs dont exist yet -- gathering now')
                    subgraphs_ = gather_random_subgraphs_not_wmk(data, dataset_name,  subgraph_method = args.subgraph_method, size = size, size_type='subgraph_size_as_fraction',seed=0, num_subgraphs=args.num_iters)
                    with open(subgraph_dir_path,'wb') as f:
                        pickle.dump(subgraphs_,f)

                compute_regression_results=True
                get_match_distribution=True

                all_raw_betas=None
                raw_betas_files = [f for f in os.listdir(model_folder) if f'raw_betas_list_size_{size}' in f]

                beta_path_filename = f'raw_betas_list_size_{size}' if args.continuation==False else f'raw_betas_list_size_{size}_continuation_from_{args.starting_epoch}'
                beta_path = os.path.join(model_folder, beta_path_filename)
                if get_match_distribution==True and os.path.exists(beta_path)==False:
                    ret_str = f'Havent yet computed raw betas -- doing that now.'
                    print(ret_str)
                    compute_regression_results=True
                if compute_regression_results==True:
                    num_nodes = int(size*sum(data.train_mask))
                    print(f'Size: {size} ({num_nodes} nodes)')
                    student_node_classifier.eval()
                    all_raw_betas = []
                    p0s = []
                    p1s = []
                    pn1s = []
                    for i, subgraph_dict__ in enumerate(subgraphs_):
                        print(f'subgraph {i}/{len(subgraphs_)}')
                        subgraph_ = subgraph_dict__['subgraph']

                        y_sub = student_node_classifier(subgraph_.x, subgraph_.edge_index, 0).clone().exp()
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

                if get_match_distribution==True:
                    numLayers=args.nLayers
                    hDim=args.hDim
                    dropout=config.node_classifier_kwargs['dropout']
                    skip = config.node_classifier_kwargs['skip_connections']
                    save=False if get_match_distribution==False else True
                    mu_natural, sigma_natural = count_beta_matches(model_folder, args.numSubgraphs, args.subgraph_size_as_fraction, save=True, continuation=args.continuation,starting_epoch=args.starting_epoch)


                print(f"mu_natural, sigma_natural, observed_count (without zeros): {mu_natural:.3f},{sigma_natural:.3f},{watermarked_match_counts_without_zeros}")
                z_score_without_zeros = (watermarked_match_counts_without_zeros - mu_natural)/sigma_natural
                p_value_without_zeros = 1 - stats.norm.cdf(z_score_without_zeros)
                print(f'Signifiance:',p_value_without_zeros)


        config.seed += 1
        args.seed +=1
        seed += 1