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
    load_data = True

    args, _ = parser.parse_known_args()
    val_ratio = test_ratio = (1-args.train_ratio)/2
    if dataset_attributes[dataset_name]['single_or_multi_graph']=='single':
        dataset = prep_data(dataset_name=dataset_name, location='default', batch_size='default', transform_list='default',  train_val_test_split=[args.train_ratio,val_ratio,test_ratio], seed=seed, save=save_data, load=load_data)
        graph_to_watermark = data = dataset[0]
    elif dataset_attributes[dataset_name]['single_or_multi_graph']=='multi':
        [train_dataset, val_dataset, test_dataset], [train_loader, val_loader, test_loader] = prep_data(dataset_name=dataset_name, location='default', batch_size='default', transform_list='default', train_val_test_split=[args.train_ratio,val_ratio,test_ratio])
        graph_to_watermark = train_dataset[0]

    get_presets(dataset,dataset_name)
    parser.add_argument('--num_iters',                                        type=int,               default=1,                                                                                  help='Number of times to run the experiment, so we can obtain an average.')
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
    parser.add_argument('--regularization_type',                          type=str,               default=config.optimization_kwargs['regularization_type'],                                  help='The regularization type to use during training.')
    parser.add_argument('--lambda_l2',                                    type=float,             default=config.optimization_kwargs['lambda_l2'],                                            help='If using L2 regularization, the lambda term.')
    parser.add_argument('--use_pcgrad',                                   type=str2bool,          default=config.optimization_kwargs['use_pcgrad'],                                           help='Whether to use PCGrad to help mitigate conflicting gradients from multi-task learning.')
    parser.add_argument('--use_sam',                                      type=str2bool,               default=config.optimization_kwargs['use_sam'],                                              help='Whether or not to use sharpness-aware minimization during training.')
    parser.add_argument('--sam_momentum',                                 type=float,               default=config.optimization_kwargs['sam_momentum'],                                         help='If using sharpness-aware minimization, the momentum value to use.')
    parser.add_argument('--sam_rho',                                      type=float,               default=config.optimization_kwargs['sam_rho'],                                              help='If using sharpness-aware minimization, the rho value to use.')


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
    config.optimization_kwargs['regularization_type']                                   = args.regularization_type
    config.optimization_kwargs['lambda_l2']                                             = args.lambda_l2
    config.optimization_kwargs['use_pcgrad']                                            = args.use_pcgrad
    config.optimization_kwargs['use_sam']                                               = args.use_sam
    config.optimization_kwargs['sam_momentum']                                          = args.sam_momentum
    config.optimization_kwargs['sam_rho']                                               = args.sam_rho

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

    config.watermark_graphlime_backdoor=args.watermark_graphlime_backdoor
    config.watermark_graphlime_backdoor_target_label = args.watermark_graphlime_backdoor_target_label
    config.watermark_graphlime_backdoor_poison_rate = args.watermark_graphlime_backdoor_poison_rate
    config.watermark_graphlime_backdoor_size = args.watermark_graphlime_backdoor_size

    if args.watermark_random_backdoor==False and args.clf_only==False and args.watermark_graphlime_backdoor==False:
        args.using_our_method=True
        config.using_our_method = True
    else:
        args.using_our_method=False
        config.using_our_method = False

    config.seed = args.seed