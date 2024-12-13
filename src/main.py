# from eaaw_graphlime_utils import *
from embed_and_verify import *
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
    parser.add_argument('--create_or_load_data',    type=str,               default='load',  help='Whether to build dataset from scratch or load it -- either "create" or "load".')
    parser.add_argument('--save_data',    action='store_true',               default=False,  help='Whether to build dataset from scratch or load it -- either "create" or "load".')

    args, _ = parser.parse_known_args()
    dataset_name = args.dataset
    seed = args.seed
    assert args.create_or_load_data in ['create','load']
    load_data = True if args.create_or_load_data=="load" else False
    save_data = args.save_data

    parser.add_argument('--train_ratio',      type=float,   default=dataset_attributes[dataset_name]['train_ratio'],  help='Ratio of dataset comprising train set.')
    args, _ = parser.parse_known_args()
    val_ratio = test_ratio = (1-args.train_ratio)/2
    if dataset_attributes[dataset_name]['single_or_multi_graph']=='single':
        dataset = prep_data(dataset_name=dataset_name, location='default', batch_size='default', transform_list='default',  train_val_test_split=[args.train_ratio,val_ratio,test_ratio], seed=seed, save=save_data, load=load_data)
        # graph_to_watermark = 
        data = dataset[0]
    elif dataset_attributes[dataset_name]['single_or_multi_graph']=='multi':
        [train_dataset, val_dataset, test_dataset], [train_loader, val_loader, test_loader] = prep_data(dataset_name=dataset_name, location='default', batch_size='default', transform_list='default', train_val_test_split=[args.train_ratio,val_ratio,test_ratio])
        # graph_to_watermark = train_dataset[0]
    # elif dataset_name=='MUTAG':
# import torch_geometric.datasets as datasets
# pyg_dataset = torch_geometric.datasets.TUDataset(root=f'/tmp/{dataset}', name=dataset,cleaned=cleaned,use_edge_attr=use_edge_attr)

        

    get_presets(dataset,dataset_name)
    parser.add_argument('--num_iters',                                        type=int,               default=1,                                                                                  help='Number of times to run the experiment, so we can obtain an average.')
    parser.add_argument('--prune',                                            action='store_true',                                                                                                help='Test with pruning.')
    parser.add_argument('--fine_tune',                                        action='store_true',                                                                                                help='Test with fine-tuning.')
    parser.add_argument('--confidence', type=float, default=0.999999, help='confidence value for recommending watermark size')
    parser.add_argument('--preserve_edges_between_subsets', action='store_true')

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
    # parser.add_argument('--penalize_similar_subgraphs',                   type=str2bool,          default=config.optimization_kwargs['penalize_similar_subgraphs'],                           help='Whether to impose penalty on watermarking match to simlar subgraphs during training.')
    # parser.add_argument('--similar_subgraph_p_swap',                                       type=float,             default=config.optimization_kwargs['p_swap'],                                               help='If using similar subgraph penalty, the probability of any subgraph nodes getting swapped with another choice.')
    # parser.add_argument('--shifted_subgraph_loss_coef',                   type=float,             default=config.optimization_kwargs['shifted_subgraph_loss_coef'],                           help='If using similar subgraph penalty, the coefficient on the corresponding loss term.')
    parser.add_argument('--sacrifice_method',                             type=str,               default=config.optimization_kwargs['sacrifice_kwargs']['method'],                           help='If sacrificing some nodes from training, the method to use.')
    parser.add_argument('--clf_only',                                     type=str2bool,          default=config.optimization_kwargs['clf_only'],                                             help='Whether to train for classificaiton only (will skip watermarking).')
    parser.add_argument('--coefWmk',                                      type=float,             default=config.optimization_kwargs['coefWmk_kwargs']['coefWmk'],                            help='The coefficient on the watermarking loss term.')
    # parser.add_argument('--coefWmk_schedule',                             type=str2bool,          default=config.optimization_kwargs['coefWmk_kwargs']['schedule_coef_wmk'],                  help='Whether to increase coef_wmk in a gradual/scheduled manner.')
    # parser.add_argument('--coefWmk_min_scheduled',                        type=float,             default=config.optimization_kwargs['coefWmk_kwargs']['min_coefWmk_scheduled'],              help='If using scheduling for coef_wmk, the lowest value to start with.')
    # parser.add_argument('--coefWmk_reach_max_by_epoch',                   type=int,               default=config.optimization_kwargs['coefWmk_kwargs']['reach_max_coef_wmk_by_epoch'],        help='If using scheduling for coef_wmk, the epoch by which the maximum coef_wmk value should be reached.')
    parser.add_argument('--regularization_type',                          type=str,               default=config.optimization_kwargs['regularization_type'],                                  help='The regularization type to use during training.')
    parser.add_argument('--lambda_l2',                                    type=float,             default=config.optimization_kwargs['lambda_l2'],                                            help='If using L2 regularization, the lambda term.')
    parser.add_argument('--use_pcgrad',                                   type=str2bool,          default=config.optimization_kwargs['use_pcgrad'],                                           help='Whether to use PCGrad to help mitigate conflicting gradients from multi-task learning.')
    # parser.add_argument('--use_sam',                                      type=str2bool,               default=config.optimization_kwargs['use_sam'],                                              help='Whether or not to use sharpness-aware minimization during training.')
    # parser.add_argument('--sam_momentum',                                 type=float,               default=config.optimization_kwargs['sam_momentum'],                                         help='If using sharpness-aware minimization, the momentum value to use.')
    # parser.add_argument('--sam_rho',                                      type=float,               default=config.optimization_kwargs['sam_rho'],                                              help='If using sharpness-aware minimization, the rho value to use.')
    parser.add_argument('--separate_forward_passes_per_subgraph',         type=int,               default=config.optimization_kwargs['separate_forward_passes_per_subgraph'],                 help='Whether to use separate forwards pa for subgraphs to obtain outputs that watermarking relies on.')

    parser.add_argument('--pGraphs',                                      type=float,             default=config.watermark_kwargs['pGraphs'],                                                 help='If using a multi-graph dataset, the proportion of graphs to watermark.')
    parser.add_argument('--percent_of_features_to_watermark',             type=float,             default=config.watermark_kwargs['percent_of_features_to_watermark'],                        help='The percentage of node features to watermark, if overriding automatic calculation.')
    parser.add_argument('--watermark_type',                               type=str,               default=config.watermark_kwargs['watermark_type'],                                          help='Watermark type ("unimportant" indices vs "most_represented" indices).')
    # parser.add_argument('--unimportant_selection_clf_only_epochs',        type=int,               default=config.watermark_kwargs['unimportant_selection_kwargs']['clf_only_epochs'],         help='If watermarking unimportant indices, the number of classification-only epochs of training prior to beginning watermarking.')
    # parser.add_argument('--unimportant_selection_evaluate_individually',  type=str2bool,          default=config.watermark_kwargs['unimportant_selection_kwargs']['evaluate_individually'],   help='If watermarking unimportant indices, whether to use recommendations from each subgraphs betas to choose their own node features to watermark (impractical).')
    # parser.add_argument('--unimportant_selection_multi_subg_strategy',    type=str,               default=config.watermark_kwargs['unimportant_selection_kwargs']['multi_subg_strategy'],     help='If watermarking unimportant indices, the way to aggregate the recommendations from all subgraphs betas to choose node features to watermark.')

    parser.add_argument('--subgraph_regenerate',                              type=str2bool,          default=config.subgraph_kwargs['regenerate'],                                               help='Whether to regenerate subgraphs rather than load from local files (recommended if recent changes to code).')
    parser.add_argument('--subgraph_method',                                  type=str,               default=config.subgraph_kwargs['method'],                                                   help='Subgraph method (khop, random, rwr).')
    parser.add_argument('--subgraph_size_as_fraction',                                type=float,             default=config.subgraph_kwargs['subgraph_size_as_fraction'],                        help='Fraction of possible subgraph nodes comprising each watermarked subgraph.')
    parser.add_argument('--numSubgraphs',                                     type=int,               default=config.subgraph_kwargs['numSubgraphs'],                                             help='Number of subgraphs to watermark.')

    # parser.add_argument('--khop_autoChooseSubGs',                             type=str2bool,          default=config.subgraph_kwargs['khop_kwargs']['autoChooseSubGs'],                           help='If True, will automatically build the khop subgraphs to use for watermarking.')
    # parser.add_argument('--khop_nodeIndices',                                 type=ast.literal_eval,  default=config.subgraph_kwargs['khop_kwargs']['nodeIndices'],                               help='If not using automatic subgraph generation, will use these indices to construct subgraph.')
    # parser.add_argument('--khop_numHops',                                     type=int,               default=config.subgraph_kwargs['khop_kwargs']['numHops'],                                   help='Number of hops to use when building khop subgraphs for watermarking.')
    # parser.add_argument('--khop_max_degree',                                  type=int,               default=config.subgraph_kwargs['khop_kwargs']['max_degree'],                                help='The maximum number of degrees to consider when choosing central nodes for building khop subgraphs (large value will crash the code, so choose moderate size).')
    parser.add_argument('--subgraph_random_kwargs',                           type=dict,              default=config.subgraph_kwargs['random_kwargs'],                                            help='Empty dict -- no kwargs in current implementation.')
    # parser.add_argument('--rwr_restart_prob',                                 type=int,               default=config.subgraph_kwargs['rwr_kwargs']['restart_prob'],                               help='The probability of restart for building subgraphs with random walk with restart.')
    # parser.add_argument('--rwr_max_steps',                                    type=int,               default=config.subgraph_kwargs['rwr_kwargs']['max_steps'],                                  help='The maximum number of steps when using random walk with restart for building subgraphs.')

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

    parser.add_argument('--watermark_random_backdoor', action='store_true')
    parser.add_argument('--watermark_random_backdoor_prob_edge', type=float, default=config.watermark_random_backdoor_prob_edge, help='**Hyperparamter for watermarking with backdoor trigger**: probability of edge connection in trigger graph.')
    parser.add_argument('--watermark_random_backdoor_proportion_ones',type=float,  default=config.watermark_random_backdoor_proportion_ones, help='**Hyperparamter for watermarking with backdoor trigger**: proportion of node features that are ones.')
    parser.add_argument('--watermark_random_backdoor_trigger_size_proportion', type=float, default=config.watermark_random_backdoor_trigger_size_proportion,help='**Hyperparamter for watermarking with backdoor trigger**: size of trigger graph as proportion of dataset.')
    parser.add_argument('--watermark_random_backdoor_trigger_alpha', type=float, default=config.watermark_random_backdoor_trigger_alpha, help='**Hyperparamter for watermarking with backdoor trigger**:relative weight for trigger loss')
    parser.add_argument('--watermark_random_backdoor_re_explain',action='store_true')


    parser.add_argument('--watermark_graphlime_backdoor', action='store_true')
    parser.add_argument('--watermark_graphlime_backdoor_target_label',type=int,default=config.watermark_graphlime_backdoor_target_label)
    parser.add_argument('--watermark_graphlime_backdoor_poison_rate',type=float,default=config.watermark_graphlime_backdoor_poison_rate)
    parser.add_argument('--watermark_graphlime_backdoor_size',type=float,default=config.watermark_graphlime_backdoor_size)

    args = parser.parse_args()

    config.preserve_edges_between_subsets = args.preserve_edges_between_subsets
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
    # config.optimization_kwargs['penalize_similar_subgraphs']                            = args.penalize_similar_subgraphs
    # config.optimization_kwargs['p_swap']                                                = args.similar_subgraph_p_swap
    # config.optimization_kwargs['shifted_subgraph_loss_coef']                            = args.shifted_subgraph_loss_coef
    config.optimization_kwargs['sacrifice_kwargs']['method']                            = args.sacrifice_method
    # config.optimization_kwargs['sacrifice_kwargs']['percentage']                        = args.sacrifice_percentage
    config.optimization_kwargs['clf_only']                                              = args.clf_only
    config.optimization_kwargs['coefWmk_kwargs']['coefWmk']                             = args.coefWmk
    # config.optimization_kwargs['coefWmk_kwargs']['schedule_coef_wmk']                   = args.coefWmk_schedule
    # config.optimization_kwargs['coefWmk_kwargs']['min_coefWmk_scheduled']               = args.coefWmk_min_scheduled
    # config.optimization_kwargs['coefWmk_kwargs']['reach_max_coef_wmk_by_epoch']         = args.coefWmk_reach_max_by_epoch
    config.optimization_kwargs['regularization_type']                                   = args.regularization_type
    config.optimization_kwargs['lambda_l2']                                             = args.lambda_l2
    config.optimization_kwargs['use_pcgrad']                                            = args.use_pcgrad
    # config.optimization_kwargs['use_sam']                                               = args.use_sam
    # config.optimization_kwargs['sam_momentum']                                          = args.sam_momentum
    # config.optimization_kwargs['sam_rho']                                               = args.sam_rho
    config.optimization_kwargs['separate_forward_passes_per_subgraph']                  = args.separate_forward_passes_per_subgraph

    config.watermark_kwargs['pGraphs']                                                  = args.pGraphs
    config.watermark_kwargs['percent_of_features_to_watermark']                         = args.percent_of_features_to_watermark
    config.watermark_kwargs['watermark_type']                                           = args.watermark_type
    # config.watermark_kwargs['unimportant_selection_kwargs']['clf_only_epochs']          = args.unimportant_selection_clf_only_epochs
    # config.watermark_kwargs['unimportant_selection_kwargs']['evaluate_individually']    = args.unimportant_selection_evaluate_individually
    # config.watermark_kwargs['unimportant_selection_kwargs']['multi_subg_strategy']      = args.unimportant_selection_multi_subg_strategy

    config.subgraph_kwargs['regenerate']                                                = args.subgraph_regenerate
    config.subgraph_kwargs['numSubgraphs']                                              = args.numSubgraphs
    config.subgraph_kwargs['method']                                                    = args.subgraph_method
    config.subgraph_kwargs['subgraph_size_as_fraction']                                 = args.subgraph_size_as_fraction
    # config.subgraph_kwargs['khop_kwargs']['autoChooseSubGs']                            = args.khop_autoChooseSubGs
    # config.subgraph_kwargs['khop_kwargs']['nodeIndices']                                = args.khop_nodeIndices
    # config.subgraph_kwargs['khop_kwargs']['numHops']                                    = args.khop_numHops
    # config.subgraph_kwargs['khop_kwargs']['max_degree']                                 = args.khop_max_degree
    config.subgraph_kwargs['random_kwargs']                                             = args.subgraph_random_kwargs
    # config.subgraph_kwargs['rwr_kwargs']['restart_prob']                                = args.rwr_restart_prob
    # config.subgraph_kwargs['rwr_kwargs']['max_steps']                                   = args.rwr_max_steps

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

    config.watermark_random_backdoor = args.watermark_random_backdoor
    config.watermark_random_backdoor_prob_edge = args.watermark_random_backdoor_prob_edge
    config.watermark_random_backdoor_proportion_ones = args.watermark_random_backdoor_proportion_ones
    config.watermark_random_backdoor_trigger_size_proportion = args.watermark_random_backdoor_trigger_size_proportion
    config.watermark_random_backdoor_trigger_alpha = args.watermark_random_backdoor_trigger_alpha

    config.watermark_graphlime_backdoor=args.watermark_graphlime_backdoor
    config.watermark_graphlime_backdoor_target_label = args.watermark_graphlime_backdoor_target_label
    config.watermark_graphlime_backdoor_poison_rate = args.watermark_graphlime_backdoor_poison_rate
    config.watermark_graphlime_backdoor_size = args.watermark_graphlime_backdoor_size
    config.watermark_random_backdoor_re_explain=args.watermark_random_backdoor_re_explain

    if args.watermark_random_backdoor==False and args.clf_only==False and args.watermark_graphlime_backdoor==False:
        args.using_our_method=True
        config.using_our_method = True
    else:
        args.using_our_method=False
        config.using_our_method = False

    config.seed = args.seed

    n_features = data.x.shape[1]
    c = config.subgraph_kwargs['numSubgraphs']
    mu_natural, sigma_natural = get_natural_match_distribution(n_features, c)
    c_LB=args.confidence
    c_t=args.confidence
    recommended_watermark_length = find_min_n_uncertain(n_features, mu_natural, sigma_natural, c_LB, c_t, test_effective=True, verbose=True)
    recommended_percent = 100*recommended_watermark_length/n_features
    config.watermark_kwargs['percent_of_features_to_watermark']=recommended_percent


    title_ = f'Separate Forward Passes -- {config.watermark_kwargs['watermark_type']} feature indices'
    title = f'{title_}.\n{config.subgraph_kwargs['numSubgraphs']} subgraphs.\nWatermarking {config.watermark_kwargs['percent_of_features_to_watermark']}% of node features'
    
    z_t=norm.ppf(args.confidence)
    target_number_matches = np.ceil(min(mu_natural +z_t*sigma_natural,data.x.shape[1]))

    data_original = copy.deepcopy(data)


    for _ in range(args.num_iters):
        data = copy.deepcopy(data_original)

        if dataset_name=='PubMed':# 
            if args.clf_only==True or config.watermark_random_backdoor==True or config.watermark_graphlime_backdoor==True:
                config.augment_kwargs['nodeMixUp']['lambda']=0.6#>0.1
                config.augment_kwargs['nodeDrop']['p']=0.1#>0.1
                config.augment_kwargs['nodeFeatMask']['p']=0.1#>0.1
                config.augment_kwargs['edgeDrop']['p']=0.1#>0.1

        Trainer_ = Trainer(data, dataset_name, target_number_matches)


        results = Trainer_.train(debug_multiple_subgraphs=False, save=True, print_every=1)
        results_folder_name = get_results_folder_name(dataset_name)

        # node_classifier = Trainer_.node_classifier
        # node_classifier.eval()
        # log_logits_train = node_classifier(Trainer_.x_train_unaugmented,Trainer_.edge_index_train_unaugmented, 0)
        # acc_train = accuracy(log_logits_train, Trainer_.y_train_unaugmented)

        # node_classifier.eval()
        # log_logits_test = node_classifier(Trainer_.x_test, Trainer_.edge_index_test, 0)
        # acc_test = accuracy(log_logits_test, Trainer_.y_test)

        # node_classifier.eval()
        # log_logits_val = node_classifier(Trainer_.x_val, Trainer_.edge_index_val, 0)
        # acc_val = accuracy(log_logits_val, Trainer_.y_val)
        #with open(os.path.join(results_folder_name,'Trainer'),'wb') as f:
        #    pickle.dump(Trainer_,f)
        

        if config.using_our_method==False:
            results_folder_name = get_results_folder_name(dataset_name)
            node_classifier, history = results
            primary_loss_curve = history['losses_primary']
            train_acc = history['train_accs'][-1]
            val_acc = history['val_accs'][-1]
            test_acc = history['test_accs'][-1]
            epoch = config.optimization_kwargs['epochs']-1
            loss_prim = primary_loss_curve[-1]

            if args.clf_only==True:
                final_performance = f'Seed {seed}\nEpoch: {epoch:3d}, clf_loss = {loss_prim:.3f}, acc (trn/val/test)= {train_acc:.3f}/{val_acc:.3f}/{test_acc:.3f}'
                results_file_name = 'results_clf_only.txt'
            elif config.watermark_random_backdoor==True:
                trigger_acc = history['trigger_accs'][-1]
                final_performance = f'Seed {seed}\nEpoch: {epoch:3d}, clf_loss = {loss_prim:.3f}, acc (trn/val/test)= {train_acc:.3f}/{val_acc:.3f}/{test_acc:.3f}, trigger acc = {trigger_acc:.3f}'
                results_file_name = 'results_watermark_random_backdoor.txt'
            elif config.watermark_graphlime_backdoor==True:
                graphlime_backdoor_acc = history['graphlime_backdoor_accs'][-1]
                final_performance = f'Seed {seed}\nEpoch: {epoch:3d}, clf_loss = {loss_prim:.3f}, acc (trn/val/test)= {train_acc:.3f}/{val_acc:.3f}/{test_acc:.3f}, backdoor nodes acc = {graphlime_backdoor_acc:.3f}'
                results_file_name = 'results_watermark_graphlime_backdoor.txt'
            model_config_results_filepath = os.path.join(results_folder_name,results_file_name)

            with open(model_config_results_filepath,'a') as f:
                f.write(final_performance + '\n')
            f.close()

            print(final_performance)
        if config.using_our_method==True:
            node_classifier, history, subgraph_dict, all_feature_importances, all_watermark_indices  = results

            primary_loss_curve, watermark_loss_curve, final_betas, watermarks, percent_matches, percent_match_mean, percent_match_std, primary_acc_curve, watermark_acc_curve, train_acc, val_acc, test_acc, match_counts_with_zeros, match_counts_without_zeros,match_count_confidence_with_zeros,match_count_confidence_without_zeros = get_performance_trends(history, subgraph_dict, config.optimization_kwargs)

            epoch = config.optimization_kwargs['epochs']-1
            loss_prim = primary_loss_curve[-1]
            loss_watermark = watermark_loss_curve[-1]
            percent_match=percent_matches[-1]
            train_acc=train_acc

            final_performance = f'Seed {seed}\nEpoch: {epoch:3d}, L (clf/wmk) = {loss_prim:.3f}/{loss_watermark:.3f}, acc (trn/val/test)= {train_acc:.3f}/{val_acc:.3f}/{test_acc:.3f}, #_match_WMK w/wout 0s = {match_counts_with_zeros}/{match_counts_without_zeros}, conf w/wout 0s = {match_count_confidence_with_zeros:.3f}/{match_count_confidence_without_zeros:.3f}'
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
        

        del Trainer_
        del results
        del node_classifier
        del history
        
        seed += 1
        set_seed(seed)
        config.seed=seed




