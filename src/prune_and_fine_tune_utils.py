import pickle
import os
import numpy as np
import torch.nn.utils.prune as prune
from models import *
from eaaw_graphlime_utils import *
import torch_geometric.nn.dense as dense
from torch.nn.modules.linear import Linear



def calculate_sparsity(model,verbose=False):
    total_params = 0
    total_zero_params = 0
    for name, module in model.named_modules():
        if isinstance(module, SAGEConv):
            weight_tensor = module.lin_l.weight
            total_params += weight_tensor.numel()
            total_zero_params += (weight_tensor == 0).sum().item()
            if hasattr(module, 'lin_r'):
                weight_tensor = module.lin_r.weight
                total_params += weight_tensor.numel()
                total_zero_params += (weight_tensor == 0).sum().item()
        elif isinstance(module, GCNConv):
            if hasattr(module, 'lin'):  # Check if the internal linear layer exists
                weight_tensor = module.lin.weight
                total_params += weight_tensor.numel()
                total_zero_params += (weight_tensor == 0).sum().item()
        else:
            if hasattr(module, 'weight'):
                weight_tensor = module.weight.data
                total_params += weight_tensor.numel()
                total_zero_params += (weight_tensor == 0).sum().item()
            elif hasattr(module, 'weight_orig'):
                weight_tensor = module.weight_orig.data
                total_params += weight_tensor.numel()
                total_zero_params += (module.weight_mask == 0).sum().item()
    sparsity = total_zero_params / total_params
    if verbose==True:
        print(f"Model Sparsity: {sparsity:.2%}")
    return sparsity


# def apply_pruning(node_classifier,amount=0.3):
#     for name, module in node_classifier.named_modules():
#         if isinstance(module, SAGEConv):
#             prune.l1_unstructured(module.lin_l, name='weight', amount=amount)
#             if hasattr(module, 'lin_r'):
#                 prune.l1_unstructured(module.lin_r, name='weight', amount=amount)
#         elif isinstance(module, GCNConv):
#             if hasattr(module, 'lin'):
#                 prune.l1_unstructured(module.lin, name='weight', amount=amount)
#             else:
#                 print(f"{name} does not have an internal 'lin' attribute with 'weight'.")
#         elif isinstance(module, (GATConv, GCNConv, GraphConv)):
#             prune.l1_unstructured(module, name='weight', amount=amount)
#         elif isinstance(module, (GATConv, GCNConv, GraphConv, dense.linear.Linear, Linear)):
#             prune.l1_unstructured(module, name='weight', amount=amount)
# # import torch_geometric.nn.dense.linear.Linear as Linear

#     return node_classifier

def apply_pruning(node_classifier, amount=0.3, structured=True, verbose=False):
    for name, module in node_classifier.named_modules():
        if verbose==True:
            if isinstance(module, SAGEConv):
                print('SAGE')
            elif isinstance(module, GCNConv):
                print("GCNConv")
            elif isinstance(module, dense.linear.Linear):
                print("dense.linear.Linear")
            elif isinstance(module,Linear):
                print("Linear")
            elif isinstance(module,GATConv):
                print("GATConv")
            elif isinstance(module,GraphConv):
                print("GraphConv")
            else:
                print('none of the above:', type(module))
                try:
                    _  = module.lin_l.weight
                    print("-- lin_l.weight: check")
                except:
                    ('-- no lin_l.weight attribute')
                try:
                    _ = module.lin_r.weight
                    print("-- lin_r.weight: check")
                except:
                    ('-- no lin_r.weight attribute')
                try:
                    _ = module.lin.weight
                    print("-- lin.weight: check")
                except:
                    ('-- no lin.weight attribute')
                try:
                    _ = module.weight.data
                    print("-- weight: check")
                except:
                    ('-- no weight attribute')
        if hasattr(module,'weight')==False:
            if verbose==True:
                print("Doesn't  have attribute 'weight")
        else:
            if structured==True:
                prune.ln_structured(module, name='weight', amount=amount, n=1, dim=1)
            elif structured==False:
                prune.l1_unstructured(module, name='weight', amount=amount)
            if verbose==True:
                print('-- Pruning module.weight')
        if hasattr(module, 'lin')==False:
            if verbose==True:
                print("-- Doesn't have attribute 'lin")
        else:
            if structured==True:
                prune.ln_structured(module.lin, name='weight', amount=amount, n=1, dim=0)
            elif structured==False:
                prune.l1_unstructured(module.lin, name='weight', amount=amount)
            if verbose==True:   
                print('-- Pruning module.lin.weight')
        if hasattr(module, 'lin_l')==False:
            if verbose==True:
                print("-- Doesn't have attribute 'lin_l'")
        else:
            if structured==True:
                prune.ln_structured(module.lin_l, name='weight', amount=amount, n=1, dim=0)
            elif structured==False:
                prune.l1_unstructured(module.lin_l, name='weight', amount=amount)
            if verbose==True:
                print('pruning module.lin_l.weight')
        if hasattr(module, 'lin_r')==False:
            if verbose==True:
                print("-- Doesn't have attribute 'lin_r'")
        else:
            if structured==True:
                prune.ln_structured(module.lin_r, name='weight', amount=amount, n=1, dim=0)
            elif structured==False:
                prune.l1_unstructured(module.lin_r, name='weight', amount=amount)
            if verbose==True:
                print('pruning module.lin_r.weight')


# easy_run_node_classifier(Trainer_object, node_classifier, data, mu_natural, sigma_natural, subgraph_dict, subgraph_dict_not_wmk, watermark_loss_kwargs, optimization_kwargs, 
                            #  regression_kwargs, target_confidence=0.99, also_show_un_watermarked_counts=True)
def run_prune(Trainer_object, data, mu_natural, sigma_natural, model_folder, subgraph_dict, subgraph_dict_not_watermarked, config_dict, seed, pruning_type='structured', save=True, target_confidence=0.99, continuation=False, starting_epoch=0, also_show_un_watermarked_counts=True):
    watermark_loss_kwargs = config_dict['watermark_loss_kwargs']
    optimization_kwargs = config_dict['optimization_kwargs']
    regression_kwargs = config_dict['regression_kwargs']
    node_classifier_kwargs = config_dict['node_classifier_kwargs']

    assert pruning_type in ['structured','unstructured']
    structured = True if pruning_type=='structured' else False

    results_prune_filename = f'results_prune_{pruning_type}.txt'# if continuation==False else f'results_prune_{pruning_type}_continuation_from_{starting_epoch}.txt'
    text_path = os.path.join(model_folder, results_prune_filename)


    if pruning_type=='structured':
        prune_rates = np.linspace(0,0.5,26)
    elif pruning_type=='unstructured':
        prune_rates = np.linspace(0,1,21)
    for amount in prune_rates:
        node_classifier_filename = 'node_classifier' if continuation==False else f'node_classifier_continuation_from_{starting_epoch}'
        node_classifier = pickle.load(open(os.path.join(model_folder,node_classifier_filename),'rb'))
        apply_pruning(node_classifier,amount=amount, structured=structured, verbose=False)
        watermark_match_rates, acc_trn, acc_test, acc_val, target_matches, \
            [match_count_wmk_with_zeros, match_count_wmk_without_zeros, \
             confidence_wmk_with_zeros, confidence_wmk_without_zeros], \
            [match_count_not_wmk_with_zeros, match_count_not_wmk_without_zeros, \
             confidence_not_wmk_with_zeros, confidence_not_wmk_without_zeros] = easy_run_node_classifier(Trainer_object, node_classifier, data, mu_natural, sigma_natural, subgraph_dict, subgraph_dict_not_watermarked, watermark_loss_kwargs, optimization_kwargs, regression_kwargs, target_confidence=target_confidence, also_show_un_watermarked_counts=also_show_un_watermarked_counts)



        mean_watermark_match_rate = np.mean(watermark_match_rates)
        if mean_watermark_match_rate==0:
            confidence_wmk_with_zeros, confidence_wmk_without_zeros = np.nan, np.nan

        ret_str = f'Trn acc: {np.round(acc_trn,3)}  ' + \
                  f'Val acc: {np.round(acc_val,3)}  ' + \
                  f'Avg wmk match rate: {np.round(mean_watermark_match_rate,3)}  ' + \
                  f'Target # matches: {target_matches}  ' + \
                  f'# Matches Wmk w/wout 0s: ({match_count_wmk_with_zeros},{match_count_wmk_without_zeros})  ' + \
                  f'Confidence w/wout 0s: ({np.round(confidence_wmk_with_zeros,3)},{np.round(confidence_wmk_without_zeros,3)})  '
        
        if also_show_un_watermarked_counts==True:
                  ret_str +=  f'# Matches Not Wmk w/wout 0s: ({match_count_not_wmk_with_zeros},{match_count_not_wmk_without_zeros})  ' + \
                              f'Confidence w/wout 0s: ({np.round(confidence_not_wmk_with_zeros,3)},{np.round(confidence_not_wmk_without_zeros,3)})'

        print(f'Prune rate: {np.round(amount, 3)}')
        print(ret_str)
        if save==True:
            with open(text_path,'a') as f:
                f.write(f'Prune rate: {np.round(amount, 3)}\n')
                f.write(ret_str)
            f.close()



def run_fine_tune(Trainer_object, dataset_name, data, mu_natural, sigma_natural, model_folder, subgraph_dict, 
                  subgraph_dict_not_watermarked, config_dict, seed, save=True,target_confidence=0.99, 
                  continuation=False, starting_epoch=0, also_show_un_watermarked_counts=True, 
                  lr=None,lr_scale=None, portion_dataset_to_use=None):
    watermark_loss_kwargs = config_dict['watermark_loss_kwargs']
    optimization_kwargs = config_dict['optimization_kwargs']
    regression_kwargs = config_dict['regression_kwargs']

    if lr_scale is not None:
        assert lr is None
        print('scaling lr')
        lr = lr_scale*config_dict['optimization_kwargs']['lr']
    elif lr is not None:
        assert lr_scale is None
    else:
        lr = config_dict['optimization_kwargs']['lr']

    print(f'fine tune -- original lr = {config_dict['optimization_kwargs']["lr"]}, new lr = {lr}')

    node_classifier_filename = 'node_classifier' if continuation==False else f'node_classifier_continuation_from_{starting_epoch}'
    node_classifier = pickle.load(open(os.path.join(model_folder,node_classifier_filename),'rb'))
    params_         = list(node_classifier.parameters())
    optimizer       = optim.Adam(params_, lr=lr)

    mask_to_train_on = data.test_mask.clone()#data.train_mask.clone()
    all_subgraph_indices = []
    for s in subgraph_dict.keys():
        try:
            all_subgraph_indices += subgraph_dict[s]['nodeIndices'].tolist()
        except:
            all_subgraph_indices += subgraph_dict[s]['nodeIndices']
    mask_to_train_on[all_subgraph_indices]=False
    if portion_dataset_to_use==None:
        portion_dataset_to_use = dataset_attributes[dataset_name]['train_ratio']
        mask_to_train_on_indices = torch.where(mask_to_train_on==True)[0]
    else:
        assert 0<=portion_dataset_to_use and portion_dataset_to_use<=1
        mask_to_train_on_indices = torch.where(mask_to_train_on==True)[0]
        print('train_nodes_to_consider_mask before:',mask_to_train_on_indices)
        torch.manual_seed(seed)
        indices_permutation = torch.randperm(len(mask_to_train_on_indices))
        proportion_as_int = int(np.floor(portion_dataset_to_use*len(data.x)))
        indices_to_use = indices_permutation[:proportion_as_int]
        mask_to_train_on_indices = mask_to_train_on_indices[indices_to_use]
        print(f'train_nodes_to_consider_mask after: (len {len(mask_to_train_on_indices)} vs dataset len {len(data.x)})',mask_to_train_on_indices)


    if lr_scale is not None:
        results_filename = f'results_fine_tune_lr_scale={lr_scale}.txt'# if continuation==False else f'results_fine_tune_continuation_from_{starting_epoch}.txt'
    elif lr is not None:
        results_filename = f'results_fine_tune_lr={lr}.txt'# if continuation==False else f'results_fine_tune_continuation_from_{starting_epoch}.txt'
    text_path = os.path.join(model_folder, results_filename)

    x_ = data.x.clone()
    y_ = data.y.clone()
    edge_index_ = data.edge_index.clone()

    if config.preserve_edges_between_subsets==False:
        edge_index_train, _ = subgraph(mask_to_train_on_indices, edge_index_, relabel_nodes=True)
        edge_index_train = edge_index_train.clone()
        x_train = data.x[mask_to_train_on_indices].clone()
        y_train = data.y[mask_to_train_on_indices].clone()

    for epoch in tqdm(range(50)):
        node_classifier.eval()
        _, acc_trn, acc_test, acc_val, _, \
            [match_count_wmk_with_zeros,match_count_wmk_without_zeros, confidence_wmk_with_zeros,confidence_wmk_without_zeros],\
            [match_count_not_wmk_with_zeros,match_count_not_wmk_without_zeros, confidence_not_wmk_with_zeros,confidence_not_wmk_without_zeros] = easy_run_node_classifier(Trainer_object, node_classifier, data, mu_natural, sigma_natural, 
                                                                                                                                                                          subgraph_dict, subgraph_dict_not_watermarked, watermark_loss_kwargs, 
                                                                                                                                                                          optimization_kwargs, regression_kwargs, target_confidence=target_confidence, 
                                                                                                                                                                          also_show_un_watermarked_counts=also_show_un_watermarked_counts)
        node_classifier.train()
        ''' augmentation was too much in some cases, gradients exploded '''
        optimizer.zero_grad()
        if config.preserve_edges_between_subsets==True:
            log_logits = node_classifier(x_, edge_index_, config.node_classifier_kwargs['dropout'])
            log_logits_train = log_logits[mask_to_train_on_indices]
        elif config.preserve_edges_between_subsets==False:
            log_logits_train = node_classifier(x_train, edge_index_train, config.node_classifier_kwargs['dropout'])


        loss   = F.nll_loss(log_logits_train, y_train)
        loss.backward(retain_graph=False)
        optimizer.step()


        fine_tune_train_acc = accuracy(log_logits_train, y_train)

        message = f'Epoch: {epoch:3d}, L_clf = {loss:.3f}, train (fine-tune) acc = {fine_tune_train_acc}, train (og) acc = {acc_trn:.3f}, test acc = {acc_test:.3f}, val acc = {acc_val:.3f}, ' + \
                  f'#_match_wmk w/wout 0s = ({match_count_wmk_with_zeros},{match_count_wmk_without_zeros}), ' + \
                  f'confidence w/wout 0s = ({confidence_wmk_with_zeros:.3f},{confidence_wmk_without_zeros:.3f})'
        if also_show_un_watermarked_counts==True:
                  message += f', #_match_un_wmk w/wout 0s = ({match_count_not_wmk_with_zeros},{match_count_not_wmk_without_zeros}), confidence w/wout 0s = ({confidence_not_wmk_with_zeros:.3f},{confidence_not_wmk_without_zeros:.3f})'

        if save==True:
            print(message)
            action='w' if epoch==0 else 'a'
            with open(text_path,action) as f:
                f.write(message + '\n')
            f.close()
    
    node_classifier_fine_tuned_path = 'node_classifier_fine_tuned'
    if lr_scale is not None:
        node_classifier_fine_tuned_path = f'{node_classifier_fine_tuned_path}_lr_scale={lr_scale}'
    elif lr is not None:
        node_classifier_fine_tuned_path = f'{node_classifier_fine_tuned_path}_lr={lr}'
    if continuation==True:
        node_classifier_fine_tuned_path += f'_continuation_from_{starting_epoch}'
    node_classifier_path = os.path.join(model_folder,node_classifier_fine_tuned_path)
    with open(node_classifier_path,'wb') as f:
        pickle.dump(node_classifier, f)

def get_target_matches_from_dataset(dataset_name,c, confidence):
    n_features = config.dataset_attributes[dataset_name]['num_features']
    mu_natural, sigma_natural = get_natural_match_distribution(n_features, c)
    z_t = norm.ppf(confidence)
    t = np.ceil(min(mu_natural +z_t*sigma_natural,n_features))
    return t, mu_natural, sigma_natural

