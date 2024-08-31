import pickle
import os
import numpy as np
import torch.nn.utils.prune as prune
from models import *
from eaaw_graphlime_utils import *


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


def apply_pruning(node_classifier,amount=0.3):
    for name, module in node_classifier.named_modules():
        if isinstance(module, SAGEConv):
            prune.l1_unstructured(module.lin_l, name='weight', amount=amount)
            if hasattr(module, 'lin_r'):
                prune.l1_unstructured(module.lin_r, name='weight', amount=amount)
        elif isinstance(module, GCNConv):
            if hasattr(module, 'lin'):
                prune.l1_unstructured(module.lin, name='weight', amount=amount)
            else:
                print(f"{name} does not have an internal 'lin' attribute with 'weight'.")
        elif isinstance(module, (GATConv, GCNConv, GraphConv)):
            prune.l1_unstructured(module, name='weight', amount=amount)
    return node_classifier


def run_prune(data, model_path, subgraph_dict, subgraph_dict_not_watermarked, config_dict, seed, save=True):
    watermark_loss_kwargs = config_dict['watermark_loss_kwargs']
    optimization_kwargs = config_dict['optimization_kwargs']
    regression_kwargs = config_dict['regression_kwargs']
    node_classifier_kwargs = config_dict['node_classifier_kwargs']
    
    text_path = os.path.join(model_path, 'results_prune.txt')
    for amount in np.linspace(0,1,11)[:-1]:
        node_classifier = pickle.load(open(os.path.join(model_path,'node_classifier'),'rb'))
        apply_pruning(node_classifier,amount=amount)
        watermark_match_rates, acc_trn, acc_val, target_matches, \
            match_count, match_count_conf, \
                match_count_not_watermarked, \
                    match_count_conf_not_watermarked = easy_run_node_classifier(node_classifier, data, subgraph_dict, subgraph_dict_not_watermarked, watermark_loss_kwargs, optimization_kwargs, regression_kwargs, node_classifier_kwargs, train=False, task='prune')
        
        mean_watermark_match_rate = np.mean(watermark_match_rates)
        if mean_watermark_match_rate==0:
            match_count_conf = np.nan

        print(f'Prune rate: {np.round(amount, 3)}')
        print(f'Train acc: {np.round(acc_trn,3)}'.ljust(20) + f'Val acc: {np.round(acc_val,3)}'.ljust(17) + f'Avg wmk match rate: {np.round(mean_watermark_match_rate,3)}'.ljust(30) + f'Target # matches: {target_matches}'.ljust(26) + f'# Matches Wmk Subgraphs: {match_count}'.ljust(32) + f'Confidence: {np.round(match_count_conf,3)}'.ljust(22) + f'# Matches Un-Wmk Subgraphs: {match_count_not_watermarked}'.ljust(32) + f'Confidence: {np.round(match_count_conf_not_watermarked,3)}')
        if save==True:
            with open(text_path,'a') as f:
                f.write(f'Prune rate: {np.round(amount, 3)}\n')
                f.write(f'Train acc: {np.round(acc_trn,3)}'.ljust(20) + f'Val acc: {np.round(acc_val,3)}'.ljust(17) + f'Avg wmk match rate: {np.round(mean_watermark_match_rate,3)}'.ljust(30) + f'Target # matches: {target_matches}'.ljust(26) + f'# Matches Wmk Subgraphs: {match_count}'.ljust(32) + f'Confidence: {np.round(match_count_conf,3)}'.ljust(22) + f'# Matches Un-Wmk Subgraphs: {match_count_not_watermarked}'.ljust(32) + f'Confidence: {np.round(match_count_conf_not_watermarked,3)}\n')
            f.close()

def run_fine_tune(data, model_path, subgraph_dict, subgraph_dict_not_watermarked, config_dict, seed, save=True):
    watermark_loss_kwargs = config_dict['watermark_loss_kwargs']
    optimization_kwargs = config_dict['optimization_kwargs']
    regression_kwargs = config_dict['regression_kwargs']
    node_classifier_kwargs = config_dict['node_classifier_kwargs']

    node_classifier = pickle.load(open(os.path.join(model_path,'node_classifier'),'rb'))
    params_         = list(node_classifier.parameters())
    optimizer       = optim.Adam(params_, lr=0.1*config_dict['optimization_kwargs']['lr'])
    train_nodes_to_consider_mask = torch.where(data.test_mask==True)[0]
    all_subgraph_indices = []
    for s in subgraph_dict.keys():
        all_subgraph_indices += subgraph_dict[s]['nodeIndices']

    # node_aug, edge_aug  = collect_augmentations()
    # augment_seed=seed
    text_path = os.path.join(model_path, 'results_fine_tune.txt')
    for epoch in tqdm(range(50)):
        ''' augmentation was too much in some cases, gradients exploded '''
        # augment_seed=update_seed(augment_seed)      
        #data    = augment_data(data, node_aug, edge_aug, train_nodes_to_consider_mask, all_subgraph_indices, seed=augment_seed)
        optimizer.zero_grad()
        edge_index, x, y = data.edge_index, data.x, data.y
        log_logits = node_classifier(x, edge_index, config.node_classifier_kwargs['dropout'])
        loss   = F.nll_loss(log_logits[train_nodes_to_consider_mask], data.y[train_nodes_to_consider_mask])
        acc_trn = accuracy(log_logits[data.test_mask], y[data.test_mask],verbose=False)
        acc_val = accuracy(log_logits[data.val_mask],   y[data.val_mask],verbose=False)
        loss.backward(retain_graph=False)
        optimizer.step()

        _, acc_trn, acc_val, _, \
            match_count, match_count_conf, \
                match_count_not_watermarked, \
                    match_count_conf_not_watermarked = easy_run_node_classifier(node_classifier, data, subgraph_dict, subgraph_dict_not_watermarked, watermark_loss_kwargs, optimization_kwargs, regression_kwargs, node_classifier_kwargs, train=True, task='fine_tune')
        
        message = f'Epoch: {epoch:3d}, loss_primary = {loss:.3f}, train (test) acc = {acc_trn:.3f}, val acc = {acc_val:.3f}, match_count = {match_count}, confidence = {match_count_conf:.3f}, match_count_un_wmk = {match_count_not_watermarked}, confidence_un_wmk = {match_count_conf_not_watermarked:.3f}'
        if save==True:
            print(message)
            with open(text_path,'a') as f:
                f.write(message + '\n')
            f.close()
    
    with open(os.path.join(model_path,'_fine_tuned'),'wb') as f:
        pickle.dump(node_classifier, f)

def gather_prune_stats(model_path):
    file = 'results_prune.txt'
    all_train_accs = []
    all_val_accs = []
    all_wmk_matches = []
    all_un_wmk_matches = []
    seeds = [path.split('seed')[1] for path in os.listdir(model_path) if path[0]!='.' and 'png' not in path]
    for seed in seeds:
        full_path = os.path.join(model_path,f'seed{seed}',file)
        with open(full_path,'r') as f:
            lines = f.readlines()
        f.close()
        prune_rate_rows = lines[0::2]
        performance_rows = lines[1::2]
        prune_rates = [float(row.split(' ')[2].split('\n')[0]) for row in prune_rate_rows]
        all_train_accs.append(torch.tensor([float(row.split('Train acc: ')[1].split(' ')[0]) for row in performance_rows]))
        all_val_accs.append(torch.tensor([float(row.split('Val acc: ')[1].split(' ')[0]) for row in performance_rows]))
        all_wmk_matches.append(torch.tensor([int(row.split('Wmk Subgraphs: ')[1].split(' ')[0]) for row in performance_rows]))
        try:
            all_un_wmk_matches.append(torch.tensor([int(row.split('Un-Wmk Subgraphs: ')[1].split(' ')[0]) for row in performance_rows]))
        except:
            all_un_wmk_matches.append(torch.tensor([int(row.split('Un-Wmk Subgraphs: ')[1][:4]) for row in performance_rows]))
            
    train_accs = torch.mean(torch.vstack(all_train_accs),dim=0)
    val_accs = torch.mean(torch.vstack(all_val_accs),dim=0)
    wmk_matches = torch.mean(torch.vstack(all_wmk_matches),dtype=float,dim=0)
    un_wmk_matches =torch.mean(torch.vstack(all_un_wmk_matches),dtype=float,dim=0)

    return prune_rates, train_accs, val_accs, wmk_matches, un_wmk_matches

def gather_fine_tune_stats(model_path):
    file = 'results_fine_tune.txt'
    all_wmk_matches = []
    all_un_wmk_matches = []
    seeds = [path.split('seed')[1] for path in os.listdir(model_path) if path[0]!='.' and 'png' not in path]
    for seed in seeds:
        full_path = os.path.join(model_path,f'seed{seed}',file)
        print('full_path:',full_path)
        with open(full_path,'r') as f:
            lines = f.readlines()
        f.close()
        epochs = [int([r for r in row.split('Epoch:')[1].split(' ') if r != ''][0].split(',')[0]) for row in lines]
        all_wmk_matches.append(torch.tensor([int(row.split('match_count = ')[1].split(',')[0]) for row in lines]))
        all_un_wmk_matches.append(torch.tensor([int(row.split('match_count_un_wmk = ')[1].split(',')[0]) for row in lines]))
    wmk_matches = torch.mean(torch.vstack(all_wmk_matches),dtype=float,dim=0)
    un_wmk_matches =torch.mean(torch.vstack(all_un_wmk_matches),dtype=float,dim=0)
    return epochs, wmk_matches, un_wmk_matches


from scipy.stats import norm
from matplotlib import patches, transforms

def arrow(ax, target_matches):
    x_start = 0.05
    trans = transforms.blended_transform_factory(ax.transAxes,ax.transAxes)
    if target_matches > ax.get_ylim()[1]:
        print('higher')
        y_start  = 0.85#y_ub-0.1
        dy = 0.05
    elif target_matches < ax.get_ylim()[0]: 
        print('lower')
        y_start = 0.15#y_lb+0.1
        dy = -0.05
    arrow = patches.FancyArrow(x_start, y_start, 0, dy, transform=trans,  color='r', width=0.02,  head_width=0.05, head_length=0.06)
    ax.add_patch(arrow)


def prune_plot(dataset_name, prune_rates, train_accs, val_accs, wmk_matches, un_wmk_matches, c, confidence, mu_natural, sigma_natural, target_matches, model_path):
    fig, axs = plt.subplots(2,1,figsize=(7,7))
    axs[0].plot(prune_rates, train_accs, label='Train Acc')
    axs[0].plot(prune_rates, val_accs, label='Val Acc')
    axs[0].set_xlabel('Prune Rate')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlim(-0.05,0.95)
    axs[0].legend(loc='center', bbox_to_anchor=(0.5, -0.24), ncol=2)
    #
    axs[1].plot(prune_rates, wmk_matches, label=f'Matches Across Watermarked Explanations of {c} Target Subgraphs',color='green')
    axs[1].plot(prune_rates, un_wmk_matches, label=f'Matches Across Explanations of {c} Randomly-Selected Subgraphs',color='lightgray')
    axs[1].set_xlabel('Prune Rate')
    x_lb = -0.05
    x_ub = 0.95
    axs[1].set_xlim(x_lb, x_ub)
    y_lb = int(np.floor(torch.min(torch.vstack((wmk_matches,un_wmk_matches))).item()))
    y_ub = int(np.ceil(torch.max(torch.vstack((wmk_matches,un_wmk_matches))).item()))
    axs[1].set_ylim(y_lb,y_ub)
    axs[1].set_ylabel('# Matches Across c Explanations')
    axs[1].legend(loc='center', bbox_to_anchor=(0.5, -0.29), ncol=1)
    if target_matches >= min(min(wmk_matches),min(un_wmk_matches)) and target_matches <= max(max(wmk_matches),max(un_wmk_matches)):
        axs[1].axhline(target_matches, color='red',linestyle=':',linewidth=1)
        axs[1].text(x_lb+0.1*(x_ub-x_lb),target_matches+0.5,f'Target # Matches\nfor Confidence={confidence}')
    else:
        arrow(axs[1], target_matches)
        y_text_pos = y_lb+0.9*(y_ub-y_lb) if target_matches>y_lb else y_lb+0.1*(y_ub-y_lb)
        x_text_pos = x_lb+0.1*(x_ub-x_lb)
        axs[1].text(x_text_pos,y_text_pos, f'Target # Matches = {int(target_matches)}')

    def scale_y1_to_pvalue(y1_value, mu_natural, sigma_natural):
        z_score = (y1_value - mu_natural)/sigma_natural
        p_value = stats.norm.cdf(z_score)
        return p_value
    def forward_convert(y1_value):
        return scale_y1_to_pvalue(y1_value, mu_natural, sigma_natural)
    def inverse_convert(p_value):
        z_score = stats.norm.ppf(p_value)
        y1_value = z_score * sigma_natural + mu_natural
        return y1_value
    
    axs1_r = axs[1].secondary_yaxis('right', functions=(forward_convert, inverse_convert))
    axs1_r.set_ylabel('Confidence',rotation=270)
    fig.suptitle('Effect of Model Pruning on Performance',y=1.01, fontsize=13)
    fig.text(0.5, 0.97, f'("{dataset_name.capitalize()}", {c} Subgraphs)', ha='center', fontsize=11)
    plt.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.15, hspace=0.33)
    plt.savefig(os.path.join(model_path,'pruning_plot.png'))
    plt.show()

def fine_tune_plot(dataset_name, epochs, wmk_matches, un_wmk_matches, c, confidence, mu_natural, sigma_natural, target_matches, model_path):
    fig, ax1 = plt.subplots(1,1,figsize=(7,4))
    ax1.plot(epochs, wmk_matches, label=f'Matches Across Watermarked Explanations of {c} Target Subgraphs',color='green')
    ax1.plot(epochs, un_wmk_matches, label=f'Matches Across Explanations of {c} Randomly-Selected Subgraphs',color='lightgray')
    ax1.set_xlabel('Fine-Tuning Epoch')
    x_lb = 0
    x_ub = epochs[-1]+1
    ax1.set_xlim(x_lb, x_ub)
    y_lb = int(np.floor(torch.min(torch.vstack((wmk_matches,un_wmk_matches))).item()))
    y_ub = int(np.ceil(torch.max(torch.vstack((wmk_matches,un_wmk_matches))).item()))
    ax1.set_ylim(y_lb, y_ub)
    ax1.set_ylabel('# Matches Across c Explanations')
    ax1.legend(loc='center', bbox_to_anchor=(0.5, -0.23), ncol=1)
    def scale_y1_to_pvalue(y1_value, mu_natural, sigma_natural):
        z_score = (y1_value - mu_natural)/sigma_natural
        p_value = stats.norm.cdf(z_score)
        p_value[0] -= 1e-5
        return p_value
    def forward_convert(y1_value):
        ret = scale_y1_to_pvalue(y1_value, mu_natural, sigma_natural)
        return ret
    def inverse_convert(p_value):
        z_score = stats.norm.ppf(p_value)
        y1_value = z_score * sigma_natural + mu_natural
        return y1_value
    axs1_r = ax1.secondary_yaxis('right', functions=(forward_convert, inverse_convert))
    axs1_r.set_ylabel('Confidence', rotation=270)
    if target_matches >= min(min(wmk_matches),min(un_wmk_matches)) and target_matches <= max(max(wmk_matches),max(un_wmk_matches)):
        ax1.axhline(target_matches, color='red',linestyle=':',linewidth=1)
        ax1.text(x_lb+0.1*(x_ub-x_lb),target_matches+0.5,f'Target # Matches\nfor Confidence={confidence}')
    else:
        arrow(ax1, target_matches)
        y_text_pos = y_lb+0.9*(y_ub-y_lb) if target_matches>y_lb else y_lb+0.1*(y_ub-y_lb)
        x_text_pos = x_lb+0.1*(x_ub-x_lb)
        ax1.text(x_text_pos,y_text_pos, f'Target # Matches = {int(target_matches)}')
    fig.suptitle('Effect of Model Fine-Tuning on Performance',y=1.01, fontsize=13)
    fig.text(0.5, 0.9, f'("{dataset_name.capitalize()}", {c} Subgraphs)', ha='center', fontsize=11)
    image_path = os.path.join(model_path,'fine_tuning_plot.png')
    plt.savefig(image_path)
    plt.show()

def get_target_matches_from_dataset(dataset_name,c, confidence):
    n_features = config.dataset_attributes[dataset_name]['num_features']
    mu_natural, sigma_natural = get_natural_match_distribution(n_features, c)
    z_t = norm.ppf(confidence)
    t = np.ceil(min(mu_natural +z_t*sigma_natural,n_features))
    return t, mu_natural, sigma_natural

def prune_plot_from_model_path(model_path):
    dataset_name = model_path.split('training_results/')[1].split('/')[0]
    c = int(model_path.split('numSubgraphs')[1].split('_')[0])
    confidence=0.99
    prune_rates, train_accs, val_accs, wmk_matches, un_wmk_matches = gather_prune_stats(model_path)
    target_matches, mu_natural, sigma_natural = get_target_matches_from_dataset(dataset_name, c, confidence)
    prune_plot(dataset_name, prune_rates, train_accs, val_accs, wmk_matches, un_wmk_matches, c, confidence, mu_natural, sigma_natural, target_matches, model_path)

def fine_tune_plot_from_model_path(model_path):
    dataset_name = model_path.split('training_results/')[1].split('/')[0]
    c = int(model_path.split('numSubgraphs')[1].split('_')[0])
    confidence=0.99
    epochs, wmk_matches, un_wmk_matches =  gather_fine_tune_stats(model_path)
    target_matches, mu_natural, sigma_natural = get_target_matches_from_dataset(dataset_name, c, confidence)
    fine_tune_plot(dataset_name, epochs, wmk_matches, un_wmk_matches, c, confidence, mu_natural, sigma_natural, target_matches, model_path)