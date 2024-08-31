from eaaw_graphlime_utils import *
from prune_and_fine_tune_utils import *

import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)



# def get_node_classifier_and_optimizer_and_subgraph_dict_for_further_processing(model_path, lr):
#     subgraph_dict   = pickle.load(open(os.path.join(model_path,'subgraph_dict'),'rb'))
#     node_classifier = pickle.load(open(os.path.join(model_path,'node_classifier'),'rb'))
#     params_         = list(node_classifier.parameters())
#     optimizer       = optim.Adam(params_, lr=lr)
#     return node_classifier, optimizer, subgraph_dict


# def run_prune(data, model_path, subgraph_dict, subgraph_dict_not_watermarked, config_dict):
#     watermark_loss_kwargs = config_dict['watermark_loss_kwargs']
#     optimization_kwargs = config_dict['optimization_kwargs']
#     regression_kwargs = config_dict['regression_kwargs']
#     node_classifier_kwargs = config_dict['node_classifier_kwargs']
#     # subgraph_kwargs = config_dict['subgraph_kwargs']
    
#     text_path = os.path.join(model_path, 'results_prune.txt')
#     for amount in np.linspace(0,1,11)[:-1]:
#         node_classifier = pickle.load(open(os.path.join(model_path,'node_classifier'),'rb'))
#         apply_pruning(node_classifier,amount=amount)
#         watermark_match_rates, acc_trn, acc_val, target_matches, \
#             match_count, match_count_conf, \
#                 match_count_not_watermarked, \
#                     match_count_conf_not_watermarked = test_node_classifier(node_classifier, data, subgraph_dict, subgraph_dict_not_watermarked, watermark_loss_kwargs, optimization_kwargs, regression_kwargs, node_classifier_kwargs, task='prune', seed=0)
        
#         mean_watermark_match_rate = np.mean(watermark_match_rates)
#         if mean_watermark_match_rate==0:
#             match_count_conf = np.nan

#         print(f'Prune rate: {np.round(amount, 3)}')
#         print(f'Train acc: {np.round(acc_trn,3)}'.ljust(20) + f'Val acc: {np.round(acc_val,3)}'.ljust(17) + f'Avg wmk match rate: {np.round(mean_watermark_match_rate,3)}'.ljust(30) + f'Target # matches: {target_matches}'.ljust(26) + f'# Matches Wmk Subgraphs: {match_count}'.ljust(32) + f'Confidence: {np.round(match_count_conf,3)}'.ljust(22) + f'# Matches Un-Wmk Subgraphs: {match_count_not_watermarked}'.ljust(32) + f'Confidence: {np.round(match_count_conf_not_watermarked,3)}')
#         with open(text_path,'a') as f:
#             f.write(f'Prune rate: {np.round(amount, 3)}\n')
#             f.write(f'Train acc: {np.round(acc_trn,3)}'.ljust(20) + f'Val acc: {np.round(acc_val,3)}'.ljust(17) + f'Avg wmk match rate: {np.round(mean_watermark_match_rate,3)}'.ljust(30) + f'Target # matches: {target_matches}'.ljust(26) + f'# Matches Wmk Subgraphs: {match_count}'.ljust(32) + f'Confidence: {np.round(match_count_conf,3)}'.ljust(22) + f'# Matches Un-Wmk Subgraphs: {match_count_not_watermarked}'.ljust(32) + f'Confidence: {np.round(match_count_conf_not_watermarked,3)}\n')
#         f.close()

# def run_fine_tune(data, model_path, subgraph_dict, subgraph_dict_not_watermarked):
#     watermark_loss_kwargs = config_dict['watermark_loss_kwargs']
#     optimization_kwargs = config_dict['optimization_kwargs']
#     regression_kwargs = config_dict['regression_kwargs']
#     node_classifier_kwargs = config_dict['node_classifier_kwargs']

#     node_classifier = pickle.load(open(os.path.join(model_path,'node_classifier'),'rb'))
#     params_         = list(node_classifier.parameters())
#     optimizer       = optim.Adam(params_, lr=0.1*config_dict['optimization_kwargs']['lr'])
#     node_aug, edge_aug  = collect_augmentations()
#     train_nodes_to_consider_mask = torch.where(data.test_mask==True)[0]
#     augment_seed=config.seed
#     all_subgraph_indices = []
#     for s in subgraph_dict.keys():
#         all_subgraph_indices += subgraph_dict[s]['nodeIndices']

#     text_path = os.path.join(model_path, 'results_fine_tune.txt')
#     for epoch in tqdm(range(50)):
#         augment_seed=update_seed(augment_seed)      
#         optimizer.zero_grad()
#         edge_index, x, y    = augment_data(data, node_aug, edge_aug, train_nodes_to_consider_mask, all_subgraph_indices, seed=augment_seed)
#         log_logits = node_classifier(x, edge_index, config.node_classifier_kwargs['dropout'])
#         loss   = F.nll_loss(log_logits[train_nodes_to_consider_mask], data.y[train_nodes_to_consider_mask])
#         acc_trn = accuracy(log_logits[data.test_mask], y[data.test_mask],verbose=False)
#         acc_val = accuracy(log_logits[data.val_mask],   y[data.val_mask],verbose=False)
#         loss.backward(retain_graph=False)
#         optimizer.step()

#         _, acc_trn, acc_val, _, \
#             match_count, match_count_conf, \
#                 match_count_not_watermarked, \
#                     match_count_conf_not_watermarked = test_node_classifier(node_classifier, data, subgraph_dict, subgraph_dict_not_watermarked, watermark_loss_kwargs, optimization_kwargs, regression_kwargs, node_classifier_kwargs, task='prune', seed=0)
        
#         message = f'Epoch: {epoch:3d}, loss_primary = {loss:.3f}, train (test) acc = {acc_trn:.3f}, val acc = {acc_val:.3f}, match_count = {match_count}, confidence = {match_count_conf:.3f}, match_count_un_wmk = {match_count_not_watermarked}, confidence_un_wmk = {match_count_conf_not_watermarked:.3f}'
#         print(message)
#         with open(text_path,'a') as f:
#             f.write(message + '\n')
#         f.close()
    
#     with open(os.path.join(model_path,'_fine_tuned'),'wb') as f:
#         pickle.dump(node_classifier, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Test pruning settings')
    parser.add_argument('--dataset', type=str, default='PubMed',help='Dataset Name')
    parser.add_argument('--prune', action='store_true', help='Whether to test effectiveness after pruning.')
    parser.add_argument('--fine_tune', action='store_true', help='Whether to test effectiveness after fine tuning.')
    parser.add_argument('--seed', type=int, default=0, help='random_seed')
    parser.add_argument('--model_path',type=str,default=f'{parent_dir}/training_results/PubMed/archSAGE_elu_nLayers3_hDim256_drop0.1_skipTrue/6.0pctMostRepIndices_random_sub_size_as_fraction0.005_numSubgraphs5_eps0.1_nodeDropP0.1_nodeMixUp5_edgeDrop0.1_lr0.001_epochs200_coefWmk70_sacrifice1subNodes_pcgrad_regressionLambda0.1_seed4',help='Path to model')

    args = parser.parse_args()
    
    dataset_name = args.dataset
    if dataset_attributes[dataset_name]['single_or_multi_graph']=='single':
        dataset = prep_data(dataset_name=dataset_name, location='default', batch_size='default', transform_list='default',  train_val_test_split=[0.9,0.05,0.05], seed=args.seed)
        graph_to_watermark = data = dataset[0]
    elif dataset_attributes[dataset_name]['single_or_multi_graph']=='multi':
        [train_dataset, val_dataset, test_dataset], [train_loader, val_loader, test_loader] = prep_data(dataset_name=dataset_name, location='default', batch_size='default', transform_list='default', train_val_test_split=[0.9,0.05,0.05])
        graph_to_watermark = train_dataset[0]
    get_presets(dataset,dataset_name)

    config_dict = pickle.load(open(os.path.join(args.model_path,'config_dict'),'rb'))

    subgraph_dict = pickle.load(open(os.path.join(args.model_path,'subgraph_dict'),'rb'))
    subgraph_dict_not_watermarked, _ = collect_subgraphs_within_single_graph_for_watermarking(data, dataset_name, use_train_mask=True, subgraph_kwargs=config_dict['subgraph_kwargs'], not_watermarked=True, seed=2575)

    if args.prune==True:
        run_prune(data, args.model_path, subgraph_dict, subgraph_dict_not_watermarked, config_dict, args.seed)
        # text_path = os.path.join(args.model_path, 'results_prune.txt')
        # for amount in np.linspace(0,1,11)[:-1]:
        #     node_classifier = pickle.load(open(os.path.join(args.model_path,'node_classifier'),'rb'))
        #     apply_pruning(node_classifier,amount=amount)
        #     watermark_match_rates, acc_trn, acc_val, target_matches, \
        #         match_count, match_count_conf, \
        #             match_count_not_watermarked, \
        #                 match_count_conf_not_watermarked = test_node_classifier(node_classifier, data, subgraph_dict, subgraph_dict_not_watermarked, watermark_loss_kwargs, optimization_kwargs, regression_kwargs, node_classifier_kwargs, task='prune', seed=0)
            
        #     mean_watermark_match_rate = np.mean(watermark_match_rates)
        #     if mean_watermark_match_rate==0:
        #         match_count_conf = np.nan

        #     print(f'Prune rate: {np.round(amount, 3)}')
        #     print(f'Train acc: {np.round(acc_trn,3)}'.ljust(20) + f'Val acc: {np.round(acc_val,3)}'.ljust(17) + f'Avg wmk match rate: {np.round(mean_watermark_match_rate,3)}'.ljust(30) + f'Target # matches: {target_matches}'.ljust(26) + f'# Matches Wmk Subgraphs: {match_count}'.ljust(32) + f'Confidence: {np.round(match_count_conf,3)}'.ljust(22) + f'# Matches Un-Wmk Subgraphs: {match_count_not_watermarked}'.ljust(32) + f'Confidence: {np.round(match_count_conf_not_watermarked,3)}')
        #     with open(text_path,'a') as f:
        #         f.write(f'Prune rate: {np.round(amount, 3)}\n')
        #         f.write(f'Train acc: {np.round(acc_trn,3)}'.ljust(20) + f'Val acc: {np.round(acc_val,3)}'.ljust(17) + f'Avg wmk match rate: {np.round(mean_watermark_match_rate,3)}'.ljust(30) + f'Target # matches: {target_matches}'.ljust(26) + f'# Matches Wmk Subgraphs: {match_count}'.ljust(32) + f'Confidence: {np.round(match_count_conf,3)}'.ljust(22) + f'# Matches Un-Wmk Subgraphs: {match_count_not_watermarked}'.ljust(32) + f'Confidence: {np.round(match_count_conf_not_watermarked,3)}\n')
        #     f.close()

    if args.fine_tune==True:
        run_fine_tune(data, args.model_path, subgraph_dict, subgraph_dict_not_watermarked, config_dict, args.seed)
        # text_path = os.path.join(args.model_path, 'results_fine_tune.txt')
        # node_classifier = pickle.load(open(os.path.join(args.model_path,'node_classifier'),'rb'))
        # params_         = list(node_classifier.parameters())
        # # config_dict     = pickle.load(open(os.path.join(args.model_path,'config_dict'),'rb'))
        # optimizer       = optim.Adam(params_, lr=0.1*config_dict['optimization_kwargs']['lr'])

        # sig_0 = list(subgraph_dict.keys())[0]
        # watermark = subgraph_dict[sig_0]['watermark']
        # node_aug, edge_aug  = collect_augmentations()
        # train_nodes_to_consider_mask = torch.where(data.test_mask==True)[0]
        # augment_seed=config.seed
        # all_subgraph_indices = []
        # for s in subgraph_dict.keys():
        #     all_subgraph_indices += subgraph_dict[s]['nodeIndices']

        # for epoch in tqdm(range(50)):
        #     augment_seed=update_seed(augment_seed)      
        #     edge_index, x, y    = augment_data(data, node_aug, edge_aug, train_nodes_to_consider_mask, all_subgraph_indices, seed=augment_seed)

        #     optimizer.zero_grad()
        #     log_logits = node_classifier(x, edge_index, config.node_classifier_kwargs['dropout'])
        #     loss   = F.nll_loss(log_logits[train_nodes_to_consider_mask], data.y[train_nodes_to_consider_mask])
        #     acc_trn = accuracy(log_logits[data.test_mask], y[data.test_mask],verbose=False)
        #     acc_val = accuracy(log_logits[data.val_mask],   y[data.val_mask],verbose=False)
        #     loss.backward(retain_graph=False)
        #     optimizer.step()


        #     watermark_match_rates, acc_trn, acc_val, target_matches, \
        #         match_count, match_count_conf, \
        #             match_count_not_watermarked, \
        #                 match_count_conf_not_watermarked = test_node_classifier(node_classifier, data, subgraph_dict, subgraph_dict_not_watermarked, watermark_loss_kwargs, optimization_kwargs, regression_kwargs, node_classifier_kwargs, task='prune', seed=0)
            
        #     message = f'Epoch: {epoch:3d}, loss_primary = {loss:.3f}, train (test) acc = {acc_trn:.3f}, val acc = {acc_val:.3f}, match_count = {match_count}, confidence = {match_count_conf:.3f}, match_count_un_wmk = {match_count_not_watermarked}, confidence_un_wmk = {match_count_conf_not_watermarked:.3f}'
        #     print(message)
        #     with open(text_path,'a') as f:
        #         f.write(message + '\n')
        #     f.close()
        
        # with open(os.path.join(args.model_path,'_fine_tuned'),'wb') as f:
        #     pickle.dump(node_classifier, f)
