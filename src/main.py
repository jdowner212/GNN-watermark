from eaaw_graphlime_utils import *
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


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
    parser.add_argument('--seed',                                             type=int,               default=0,  help='Random seed.')
    args, _ = parser.parse_known_args()
    dataset_name = args.dataset
    seed = args.seed



    if dataset_attributes[dataset_name]['single_or_multi_graph']=='single':
        dataset = prep_data(dataset_name=dataset_name, location='default', batch_size='default', transform_list='default',  train_val_test_split=[0.9,0.05,0.05], seed=seed)
        graph_to_watermark = data = dataset[0]
    elif dataset_attributes[dataset_name]['single_or_multi_graph']=='multi':
        [train_dataset, val_dataset, test_dataset], [train_loader, val_loader, test_loader] = prep_data(dataset_name=dataset_name, location='default', batch_size='default', transform_list='default', train_val_test_split=[0.9,0.05,0.05])
        graph_to_watermark = train_dataset[0]

    get_presets(dataset,dataset_name)
    parser.add_argument('--wmk_size_auto_compute',                            type=str2bool,          default=True,                                                                               help='If True, will automatically compute recommended watermark size.')
    parser.add_argument('--num_iters',                                        type=int,               default=1,                                                                                  help='Number of times to run the experiment, so we can obtain an average.')
    parser.add_argument('--prune',                                            action='store_true',                                                                                                help='Test with pruning.')
    parser.add_argument('--fine_tune',                                        action='store_true',                                                                                                help='Test with fine-tuning.')

    parser.add_argument('--arch',                                             type=str,               default=config.node_classifier_kwargs['arch'],                                              help='GNN architecture (GAT, GCN, GraphConv, SAGE).')
    parser.add_argument('--activation',                                       type=str,               default=config.node_classifier_kwargs['activation'],                                        help='relu or elu.')
    parser.add_argument('--nLayers',                                          type=int,               default=config.node_classifier_kwargs['nLayers'],                                           help='Number of layers in GNN.')
    parser.add_argument('--hDim',                                             type=int,               default=config.node_classifier_kwargs['hDim'],                                              help='Number of hidden dimensions in GNN.')
    parser.add_argument('--dropout',                                          type=float,             default=config.node_classifier_kwargs['dropout'],                                           help='Dropout rate for classification training.')
    parser.add_argument('--dropout_subgraphs',                                type=float,             default=config.node_classifier_kwargs['dropout_subgraphs'],                                 help='Dropout rate for computing forward pass on watermarked subgraphs during training.')
    parser.add_argument('--skip_connections',                                 type=str2bool,          default=config.node_classifier_kwargs['skip_connections'],                                  help='Whether to include skip connections in GNN architecture: True (yes, true, t, y, 1) or False (no, false, f, n, 0).')
    parser.add_argument('--heads_1',                                          type=int,               default=config.node_classifier_kwargs['heads_1'],                                           help='Number of attention heads to use in first layer of GAT architecture.')
    parser.add_argument('--heads_2',                                          type=int,               default=config.node_classifier_kwargs['heads_2'],                                           help='Number of attention heads to use in intermediate layers of GAT architecture.')

    parser.add_argument('--lr',                                           type=float,             default=config.optimization_kwargs['lr'],                                                   help='Learning rate.')
    parser.add_argument('--epochs',                                       type=int,               default=config.optimization_kwargs['epochs'],                                               help='Epochs.')
    parser.add_argument('--penalize_similar_subgraphs',                   type=str2bool,          default=config.optimization_kwargs['penalize_similar_subgraphs'],                           help='Whether to impose penalty on watermarking match to simlar subgraphs during training.')
    parser.add_argument('--similar_subgraph_p_swap',                                       type=float,             default=config.optimization_kwargs['p_swap'],                                               help='If using similar subgraph penalty, the probability of any subgraph nodes getting swapped with another choice.')
    parser.add_argument('--shifted_subgraph_loss_coef',                   type=float,             default=config.optimization_kwargs['shifted_subgraph_loss_coef'],                           help='If using similar subgraph penalty, the coefficient on the corresponding loss term.')
    parser.add_argument('--sacrifice_method',                             type=str,               default=config.optimization_kwargs['sacrifice_kwargs']['method'],                           help='If sacrificing some nodes from training, the method to use.')
    # parser.add_argument('--sacrifice_percentage',                         type=float,             default=config.optimization_kwargs['sacrifice_kwargs']['percentage'],                       help='If sacrificing some nodes from training, the proportion of the chosen subset to use.')
    parser.add_argument('--clf_only',                                     type=str2bool,          default=config.optimization_kwargs['clf_only'],                                             help='Whether to train for classificaiton only (will skip watermarking).')
    parser.add_argument('--coefWmk',                                      type=float,             default=config.optimization_kwargs['coefWmk_kwargs']['coefWmk'],                            help='The coefficient on the watermarking loss term.')
    parser.add_argument('--coefWmk_schedule',                             type=str2bool,          default=config.optimization_kwargs['coefWmk_kwargs']['schedule_coef_wmk'],                  help='Whether to increase coef_wmk in a gradual/scheduled manner.')
    parser.add_argument('--coefWmk_min_scheduled',                        type=float,             default=config.optimization_kwargs['coefWmk_kwargs']['min_coefWmk_scheduled'],              help='If using scheduling for coef_wmk, the lowest value to start with.')
    parser.add_argument('--coefWmk_reach_max_by_epoch',                   type=int,               default=config.optimization_kwargs['coefWmk_kwargs']['reach_max_coef_wmk_by_epoch'],        help='If using scheduling for coef_wmk, the epoch by which the maximum coef_wmk value should be reached.')
    parser.add_argument('--regularization_type',                          type=str,               default=config.optimization_kwargs['regularization_type'],                                  help='The regularization type to use during training.')
    parser.add_argument('--lambda_l2',                                    type=float,             default=config.optimization_kwargs['lambda_l2'],                                            help='If using L2 regularization, the lambda term.')
    parser.add_argument('--use_pcgrad',                                   type=str2bool,          default=config.optimization_kwargs['use_pcgrad'],                                           help='Whether to use PCGrad to help mitigate conflicting gradients from multi-task learning.')
    parser.add_argument('--use_sam',                                      type=str2bool,               default=config.optimization_kwargs['use_sam'],                                              help='Whether or not to use sharpness-aware minimization during training.')
    parser.add_argument('--sam_momentum',                                 type=float,               default=config.optimization_kwargs['sam_momentum'],                                         help='If using sharpness-aware minimization, the momentum value to use.')
    parser.add_argument('--sam_rho',                                      type=float,               default=config.optimization_kwargs['sam_rho'],                                              help='If using sharpness-aware minimization, the rho value to use.')
    parser.add_argument('--separate_forward_passes_per_subgraph',         type=int,               default=config.optimization_kwargs['separate_forward_passes_per_subgraph'],                 help='Whether to use separate forwards pa for subgraphs to obtain outputs that watermarking relies on.')

    parser.add_argument('--pGraphs',                                      type=float,             default=config.watermark_kwargs['pGraphs'],                                                 help='If using a multi-graph dataset, the proportion of graphs to watermark.')
    parser.add_argument('--percent_of_features_to_watermark',             type=float,             default=config.watermark_kwargs['percent_of_features_to_watermark'],                        help='The percentage of node features to watermark, if overriding automatic calculation.')
    parser.add_argument('--watermark_type',                               type=str,               default=config.watermark_kwargs['watermark_type'],                                          help='Watermark type ("unimportant" indices vs "most_represented" indices).')
    parser.add_argument('--unimportant_selection_clf_only_epochs',        type=int,               default=config.watermark_kwargs['unimportant_selection_kwargs']['clf_only_epochs'],         help='If watermarking unimportant indices, the number of classification-only epochs of training prior to beginning watermarking.')
    parser.add_argument('--unimportant_selection_evaluate_individually',  type=str2bool,          default=config.watermark_kwargs['unimportant_selection_kwargs']['evaluate_individually'],   help='If watermarking unimportant indices, whether to use recommendations from each subgraphs betas to choose their own node features to watermark (impractical).')
    parser.add_argument('--unimportant_selection_multi_subg_strategy',    type=str,               default=config.watermark_kwargs['unimportant_selection_kwargs']['multi_subg_strategy'],     help='If watermarking unimportant indices, the way to aggregate the recommendations from all subgraphs betas to choose node features to watermark.')

    parser.add_argument('--subgraph_regenerate',                              type=str2bool,          default=config.subgraph_kwargs['regenerate'],                                               help='Whether to regenerate subgraphs rather than load from local files (recommended if recent changes to code).')
    parser.add_argument('--subgraph_method',                                  type=str,               default=config.subgraph_kwargs['method'],                                                   help='Subgraph method (khop, random, rwr).')
    parser.add_argument('--subgraph_size_as_fraction',                                type=float,             default=config.subgraph_kwargs['subgraph_size_as_fraction'],                        help='Fraction of possible subgraph nodes comprising each watermarked subgraph.')
    parser.add_argument('--numSubgraphs',                                     type=int,               default=config.subgraph_kwargs['numSubgraphs'],                                             help='Number of subgraphs to watermark.')
    parser.add_argument('--khop_autoChooseSubGs',                             type=str2bool,          default=config.subgraph_kwargs['khop_kwargs']['autoChooseSubGs'],                           help='If True, will automatically build the khop subgraphs to use for watermarking.')
    parser.add_argument('--khop_nodeIndices',                                 type=ast.literal_eval,  default=config.subgraph_kwargs['khop_kwargs']['nodeIndices'],                               help='If not using automatic subgraph generation, will use these indices to construct subgraph.')
    parser.add_argument('--khop_numHops',                                     type=int,               default=config.subgraph_kwargs['khop_kwargs']['numHops'],                                   help='Number of hops to use when building khop subgraphs for watermarking.')
    parser.add_argument('--khop_max_degree',                                  type=int,               default=config.subgraph_kwargs['khop_kwargs']['max_degree'],                                help='The maximum number of degrees to consider when choosing central nodes for building khop subgraphs (large value will crash the code, so choose moderate size).')
    parser.add_argument('--subgraph_random_kwargs',                           type=dict,              default=config.subgraph_kwargs['random_kwargs'],                                            help='Empty dict -- no kwargs in current implementation.')
    parser.add_argument('--rwr_restart_prob',                                 type=int,               default=config.subgraph_kwargs['rwr_kwargs']['restart_prob'],                               help='The probability of restart for building subgraphs with random walk with restart.')
    parser.add_argument('--rwr_max_steps',                                    type=int,               default=config.subgraph_kwargs['rwr_kwargs']['max_steps'],                                  help='The maximum number of steps when using random walk with restart for building subgraphs.')

    parser.add_argument('--regression_lambda',                                type=float,             default=config.regression_kwargs['lambda'],                                                 help='The lambda value to use for regresion when getting GraphLIME explanation.')
    
    parser.add_argument('--watermark_loss_epsilon',                           type=float,             default=config.watermark_loss_kwargs['epsilon'],                                            help='Caps the influence of each nod feature index on the watermark loss. Smaller epislon = stricter cap.')

    parser.add_argument('--augment_separate_trainset_from_subgraphs',         type=str2bool,          default=config.augment_kwargs['separate_trainset_from_subgraphs'],                          help='Whether to augment regular training data separately from subgraphs used for watermarking.')
    parser.add_argument('--augment_p',                                        type=float,             default=config.augment_kwargs['p'],                                                         help='The proportion of data to augment.')
    parser.add_argument('--augment_ignore_subgraphs',                         type=str2bool,          default=config.augment_kwargs['ignore_subgraphs'],                                          help='If True, will not augment subgraphs used for watermarking.')
    parser.add_argument('--augment_nodeDrop_use',                             type=str2bool,          default=config.augment_kwargs['nodeDrop']['use'],                                           help='If True, will use nodeDrop augmentation.')
    parser.add_argument('--augment_nodeDrop_p',                               type=float,             default=config.augment_kwargs['nodeDrop']['p'],                                             help='If using nodeDrop augmentation, the probability of dropping a node.')
    parser.add_argument('--augment_nodeMixUp_use',                            type=str2bool,          default=config.augment_kwargs['nodeMixUp']['use'],                                          help='If True, will use nodeMixUp augmentation.')
    parser.add_argument('--augment_nodeMixUp_lambda',                         type=float,             default=config.augment_kwargs['nodeMixUp']['lambda'],                                       help='If using nodeMixUp augmentation, the relative ratio given to each node in the mixup (lambda, 1-lambda).')
    parser.add_argument('--augment_nodeFeatMask_use',                         type=str2bool,          default=config.augment_kwargs['nodeFeatMask']['use'],                                       help='If True, will use nodeFeatMask augmentation.')
    parser.add_argument('--augment_nodeFeatMask_p',                           type=float,             default=config.augment_kwargs['nodeFeatMask']['p'],                                         help='If using nodeFeatMask augmentation, the probability of masking node features.')
    parser.add_argument('--augment_edgeDrop_use',                             type=str2bool,          default=config.augment_kwargs['edgeDrop']['use'],                                           help='If True, will use edgeDrop augmentation.')
    parser.add_argument('--augment_edgeDrop_p',                               type=float,             default=config.augment_kwargs['edgeDrop']['p'],                                             help='If using edgeDrop augmentation, the probability of dropping an edge.')


    args = parser.parse_args()

    config.node_classifier_kwargs['arch']                                               = args.arch
    config.node_classifier_kwargs['activation']                                         = args.activation
    config.node_classifier_kwargs['nLayers']                                            = args.nLayers
    config.node_classifier_kwargs['hDim']                                               = args.hDim
    config.node_classifier_kwargs['dropout']                                            = args.dropout
    config.node_classifier_kwargs['dropout_subgraphs']                                  = args.dropout_subgraphs
    config.node_classifier_kwargs['skip_connections']                                   = args.skip_connections
    config.node_classifier_kwargs['heads_1']                                            = args.heads_1
    config.node_classifier_kwargs['heads_2']                                            = args.heads_2

    config.optimization_kwargs['lr']                                                    = args.lr
    config.optimization_kwargs['epochs']                                                = args.epochs
    config.optimization_kwargs['penalize_similar_subgraphs']                            = args.penalize_similar_subgraphs
    config.optimization_kwargs['p_swap']                                                = args.similar_subgraph_p_swap
    config.optimization_kwargs['shifted_subgraph_loss_coef']                            = args.shifted_subgraph_loss_coef
    config.optimization_kwargs['sacrifice_kwargs']['method']                            = args.sacrifice_method
    # config.optimization_kwargs['sacrifice_kwargs']['percentage']                        = args.sacrifice_percentage
    config.optimization_kwargs['clf_only']                                              = args.clf_only
    config.optimization_kwargs['coefWmk_kwargs']['coefWmk']                             = args.coefWmk
    config.optimization_kwargs['coefWmk_kwargs']['schedule_coef_wmk']                   = args.coefWmk_schedule
    config.optimization_kwargs['coefWmk_kwargs']['min_coefWmk_scheduled']               = args.coefWmk_min_scheduled
    config.optimization_kwargs['coefWmk_kwargs']['reach_max_coef_wmk_by_epoch']         = args.coefWmk_reach_max_by_epoch
    config.optimization_kwargs['regularization_type']                                   = args.regularization_type
    config.optimization_kwargs['lambda_l2']                                             = args.lambda_l2
    config.optimization_kwargs['use_pcgrad']                                            = args.use_pcgrad
    config.optimization_kwargs['use_sam']                                               = args.use_sam
    config.optimization_kwargs['sam_momentum']                                          = args.sam_momentum
    config.optimization_kwargs['sam_rho']                                               = args.sam_rho
    config.optimization_kwargs['separate_forward_passes_per_subgraph']                  = args.separate_forward_passes_per_subgraph

    config.watermark_kwargs['pGraphs']                                                  = args.pGraphs
    config.watermark_kwargs['percent_of_features_to_watermark']                         = args.percent_of_features_to_watermark
    config.watermark_kwargs['watermark_type']                                           = args.watermark_type
    config.watermark_kwargs['unimportant_selection_kwargs']['clf_only_epochs']          = args.unimportant_selection_clf_only_epochs
    config.watermark_kwargs['unimportant_selection_kwargs']['evaluate_individually']    = args.unimportant_selection_evaluate_individually
    config.watermark_kwargs['unimportant_selection_kwargs']['multi_subg_strategy']      = args.unimportant_selection_multi_subg_strategy

    config.subgraph_kwargs['regenerate']                                                = args.subgraph_regenerate
    config.subgraph_kwargs['method']                                                    = args.subgraph_method
    config.subgraph_kwargs['subgraph_size_as_fraction']                                 = args.subgraph_size_as_fraction
    config.subgraph_kwargs['numSubgraphs']                                              = args.numSubgraphs
    config.subgraph_kwargs['khop_kwargs']['autoChooseSubGs']                            = args.khop_autoChooseSubGs
    config.subgraph_kwargs['khop_kwargs']['nodeIndices']                                = args.khop_nodeIndices
    config.subgraph_kwargs['khop_kwargs']['numHops']                                    = args.khop_numHops
    config.subgraph_kwargs['khop_kwargs']['max_degree']                                 = args.khop_max_degree
    config.subgraph_kwargs['random_kwargs']                                             = args.subgraph_random_kwargs
    config.subgraph_kwargs['rwr_kwargs']['restart_prob']                                = args.rwr_restart_prob
    config.subgraph_kwargs['rwr_kwargs']['max_steps']                                   = args.rwr_max_steps

    config.regression_kwargs['lambda']                                                  = args.regression_lambda

    config.watermark_loss_kwargs['epsilon']                                             = args.watermark_loss_epsilon

    config.augment_kwargs['separate_trainset_from_subgraphs']                           = args.augment_separate_trainset_from_subgraphs
    config.augment_kwargs['p']                                                          = args.augment_p
    config.augment_kwargs['ignore_subgraphs']                                           = args.augment_ignore_subgraphs
    config.augment_kwargs['nodeDrop']['use']                                            = args.augment_nodeDrop_use
    config.augment_kwargs['nodeDrop']['p']                                              = args.augment_nodeDrop_p
    config.augment_kwargs['nodeMixUp']['use']                                           = args.augment_nodeMixUp_use
    config.augment_kwargs['nodeMixUp']['lambda']                                        = args.augment_nodeMixUp_lambda
    config.augment_kwargs['nodeFeatMask']['use']                                        = args.augment_nodeFeatMask_use
    config.augment_kwargs['nodeFeatMask']['p']                                          = args.augment_nodeFeatMask_p
    config.augment_kwargs['edgeDrop']['use']                                            = args.augment_edgeDrop_use
    config.augment_kwargs['edgeDrop']['p']                                              = args.augment_edgeDrop_p


    config.seed = args.seed

    if args.wmk_size_auto_compute==True:
        n_features = data.x.shape[1]
        print('n_features:',n_features)
        c = config.subgraph_kwargs['numSubgraphs']
        mu_natural, sigma_natural = get_natural_match_distribution(n_features, c)
        print('mu_natural:',mu_natural)
        print('sigma_natural:',sigma_natural)
        c_LB=0.99
        c_t=0.99
        recommended_watermark_length = find_min_n_uncertain(n_features, mu_natural, sigma_natural, c_LB, c_t, test_effective=True, verbose=True)
        print(f'recommended_watermark_length for confidence={c_t}:',recommended_watermark_length)
        recommended_percent = 100*recommended_watermark_length/n_features
        print('recommended_percent:',recommended_percent)

        config.watermark_kwargs['percent_of_features_to_watermark']=recommended_percent


    title_ = f'Separate Forward Passes -- {config.watermark_kwargs['watermark_type']} feature indices'
    title = f'{title_}.\n{config.subgraph_kwargs['numSubgraphs']} subgraphs.\nWatermarking {config.watermark_kwargs['percent_of_features_to_watermark']}% of node features'
    
    for _ in range(args.num_iters):
        Trainer_ = Trainer(data, dataset_name)

        node_classifier, history, subgraph_dict, all_feature_importances, all_watermark_indices = Trainer_.train(debug_multiple_subgraphs=False, save=True, print_every=1)
        primary_loss_curve, watermark_loss_curve, final_betas, watermarks, percent_matches, percent_match_mean, percent_match_std, primary_acc_curve, watermark_acc_curve, train_acc, val_acc, match_counts, match_count_confidence = get_performance_trends(history, subgraph_dict)

        epoch = config.optimization_kwargs['epochs']-1
        loss_prim = primary_loss_curve[-1]
        loss_watermark = watermark_loss_curve[-1]
        percent_match=percent_matches[-1]
        train_acc=train_acc
        final_performance = f'Seed {seed}\nEpoch {epoch}: primary_loss={loss_prim:.3f}, watermark_loss={loss_watermark:.3f}, train_acc={train_acc:.3f}, wmk_match={percent_match:.3f}, match_count={match_counts}, match_count_confidence={match_count_confidence:.3f}\n'
        print("**")
        results_folder_name = get_results_folder_name(dataset_name)
        results_file_name = 'results.txt' if args.fine_tune==False else 'results_fine_tune.txt'
        model_config_results_filepath = os.path.join(results_folder_name,results_file_name)


        mu_natural, sigma_natural = get_natural_match_distribution(n_features, c)
        if os.path.exists(model_config_results_filepath)==False:
            with open(model_config_results_filepath,'w') as f:
                f.write(f'Natural match distribution: mu={mu_natural:.3f}, sigma={sigma_natural:.3f}\n')
            f.close()        
        with open(model_config_results_filepath,'a') as f:
            f.write(final_performance)
        f.close()

        plot_name = dataset_name
        save_fig=True if args.fine_tune==False else False
        final_plot(history, title, percent_matches, primary_loss_curve, watermark_loss_curve, train_acc, plot_name=plot_name,save=save_fig)
        print(final_performance)
        


        seed += 1
        set_seed(seed)
        




