from embed_and_verify import *




class Trainer_KD():
    def __init__(self, student_model, teacher_model, data, dataset_name, subgraph_dict, subgraph_dict_not_wmk, alpha, temperature, train_node_indices, test_node_indices, val_node_indices):
        self.temperature = temperature
        self.alpha = alpha
        self.data = data
        self.dataset_name = dataset_name
        self.num_features = data.x.shape[1]
        self.train_mask = data.train_mask
        self.val_mask = data.val_mask
        self.test_mask = data.test_mask
        self.optimization_kwargs = config.KD_student_optimization_kwargs
        self.use_pcgrad        = self.optimization_kwargs['use_pcgrad']
        self.lr                = self.optimization_kwargs['lr']
        self.epochs            = self.optimization_kwargs['epochs']
        self.loss_dict = setup_loss_dict()
        self.node_classifier=student_model
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.instantiate_optimizer()
        self.best_train_acc, self.best_val_acc = 0, 0
        validate_kwargs()

        self.loss = torch.tensor(0.0)
        self.subgraph_dict = subgraph_dict
        self.subgraph_dict_not_wmk = subgraph_dict_not_wmk
        self.num_features = data.x.shape[1]
        self.each_subgraph_feature_importances=None
        self.each_subgraph_watermark_indices=None
        self.subgraph_signatures = list(subgraph_dict.keys())
        all_subgraph_indices = []
        for sig in subgraph_dict.keys():
            nodeIndices = subgraph_dict[sig]['nodeIndices'].tolist()
            all_subgraph_indices.append(nodeIndices)
        self.all_subgraph_indices = torch.tensor(all_subgraph_indices)
        self.subgraph_signatures_not_wmk = list(subgraph_dict_not_wmk.keys())
        self.history, self.betas_dict, self.beta_similarities_dict              = setup_history(subgraph_signatures=self.subgraph_signatures)
        self.beta_weights                                                       = get_beta_weights(self.subgraph_dict, self.num_features)
        self.history_not_wmk, self.betas_dict_not_wmk, self.beta_similarities_dict_not_wmk              = setup_history(subgraph_signatures=self.subgraph_signatures_not_wmk)
        self.beta_weights_not_wmk                                                                       = get_beta_weights(self.subgraph_dict_not_wmk, self.num_features)
        self.train_node_indices = train_node_indices
        if config.kd_subgraphs_only==True:
            config.kd_train_on_subgraphs=True
        if config.kd_train_on_subgraphs==True:
            self.separate_forward_passes_per_subgraph_=True
            if config.kd_subgraphs_only==True:
                self.train_nodes_to_consider = self.all_subgraph_indices.reshape(-1)
                self.train_node_indices =self.train_nodes_to_consider
            else:
                self.train_nodes_to_consider_mask = get_train_nodes_to_consider(self.data, self.all_subgraph_indices, 'subgraph_node_indices', self.data.x.shape[0], train_with_test_set=False)
                self.train_node_indices = self.train_nodes_to_consider_mask.nonzero(as_tuple=True)[0]
                self.train_nodes_to_consider = torch.where(self.train_nodes_to_consider_mask==True)[0]
        else:
            self.separate_forward_passes_per_subgraph_=False
            self.train_nodes_to_consider = train_node_indices

        if config.preserve_edges_between_subsets==False:
            self.edge_index_train, _ = subgraph(self.train_node_indices, self.data.edge_index, relabel_nodes=True)
            self.edge_index_train = self.edge_index_train.clone()
        self.x_train = self.data.x[self.train_node_indices].clone()
        self.y_train = self.data.y[self.train_node_indices].clone()
        ### for evaluating on test set
        self.test_node_indices = test_node_indices
        if config.preserve_edges_between_subsets==False:
            self.edge_index_test, _ = subgraph(self.test_node_indices, self.data.edge_index, relabel_nodes=True)
            self.edge_index_test = self.edge_index_test.clone()
        self.x_test = self.data.x[self.test_node_indices].clone()
        self.y_test = self.data.y[self.test_node_indices].clone()
        
        ### for evaluating on validation set
        self.val_node_indices = val_node_indices
        if config.preserve_edges_between_subsets==False:
            self.edge_index_val, _ = subgraph(val_node_indices, self.data.edge_index, relabel_nodes=True)
            self.edge_index_val = self.edge_index_val.clone()
        self.x_val = self.data.x[self.val_node_indices].clone()
        self.y_val = self.data.y[self.val_node_indices].clone()
        return

    def save_process(self, continuation=False,starting_epoch=None, file_ext=None, verbose=False):
        if file_ext is None or file_ext == '':
            file_ext=''
        else:
            file_ext = "_"+file_ext
        if config.preserve_edges_between_subsets==True:
            file_ext = '_preserve_edges' + file_ext
        if continuation==True:
            Trainer_str = f'Trainer{file_ext}_continuation_from_{starting_epoch}'
        else:
            Trainer_str = f'Trainer{file_ext}'
        with open(os.path.join(get_results_folder_name(self.dataset_name),Trainer_str), 'wb') as f:
            pickle.dump(self, f)
        save_results(self.dataset_name, self.node_classifier, self.history, self.subgraph_dict, 
                        self.each_subgraph_feature_importances, self.each_subgraph_watermark_indices, 
                        verbose=verbose,continuation=continuation, starting_epoch=starting_epoch)

    def instantiate_optimizer(self):
        optimizer = optim.Adam(self.node_classifier.parameters(), lr=self.lr)
        if self.use_pcgrad==True:
            optimizer = PCGrad(optimizer)
        self.optimizer = optimizer


    def train_KD(self, save=True, print_every=10, continuation=False, starting_epoch=0):
        for epoch in tqdm(range(self.epochs)):
            extra_print=''
            epoch += starting_epoch
            self.epoch=epoch
            closure = self.closure_KD
            closure()
            self.optimizer.step()
            self.history = update_history_one_epoch(self.history, self.loss, self.loss_dict, self.acc_trn, self.acc_val, self.acc_test, None, None, None,None, None)
            if epoch%1==0:
                watermarked_subgraph_results, unwatermarked_subgraph_results = self.test_watermark()
                match_count_without_zeros_wmk, _, confidence_without_zeros_wmk = watermarked_subgraph_results[7:10]
                match_count_without_zeros_un_wmk, _, confidence_without_zeros_un_wmk = unwatermarked_subgraph_results[7:10]
                extra_print = f'wmk/unwmk counts (without zeros): {match_count_without_zeros_wmk}/{match_count_without_zeros_un_wmk}, wmk/unwmk confidence: {confidence_without_zeros_wmk:.3f}/{confidence_without_zeros_un_wmk:.3f}'

            if self.epoch%print_every==0:
                print_epoch_status(self.epoch, self.classification_loss, self.acc_trn, self.acc_val, self.acc_test, False, None, None, None,None, None,None, None,None, None,True,self.acc_trn_KD, self.acc_val_KD, self.acc_test_KD, self.distillation_loss_, additional_content=extra_print)
            gc.collect()
            torch.cuda.empty_cache() 
            if save==True:
                self.save_process(continuation,starting_epoch,'KD',verbose=epoch==self.epochs-1)
        self.history = replace_history_Nones(self.history)
        if save==True:
            self.save_process(continuation,starting_epoch,'KD')            
        gc.collect()
        return self.node_classifier, self.history
    
    def distillation_loss(self, student_logits, teacher_logits):
        student_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        return F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)

    def closure_KD(self):
        self.optimizer.zero_grad()
        self.teacher_model.eval()

        self.distillation_loss_=torch.tensor(0,dtype=torch.float)
        self.classification_loss=torch.tensor(0,dtype=torch.float)
        ##

        if config.preserve_edges_between_subsets==True:
            if config.kd_subgraphs_only==False:
                student_logits = self.forward(self.data.x, self.data.edge_index, dropout=config.KD_student_node_classifier_kwargs['dropout'], mode='train')
                student_logits_train = student_logits[self.train_node_indices]
            student_logits_eval  = self.forward(self.data.x, self.data.edge_index, dropout=config.KD_student_node_classifier_kwargs['dropout'], mode='eval')
            student_logits_train_eval = student_logits_eval[self.train_node_indices]
            student_logits_val        = student_logits_eval[self.val_node_indices]
            student_logits_test       = student_logits_eval[self.test_node_indices]

            self.teacher_model.eval()
            teacher_logits = self.teacher_model(self.data.x, self.data.edge_index)
            teacher_logits_train = teacher_logits[self.train_node_indices]
            teacher_logits_val   = teacher_logits[self.val_node_indices]
            teacher_logits_test  = teacher_logits[self.test_node_indices]


        elif config.preserve_edges_between_subsets==False:
            if config.kd_subgraphs_only==False:
                student_logits_train = self.forward(self.x_train, self.edge_index_train, dropout=config.KD_student_node_classifier_kwargs['dropout'], mode='train')
            self.teacher_model.eval()
            teacher_logits_train = self.teacher_model(self.x_train, self.edge_index_train)
            teacher_logits_val   = self.teacher_model(self.x_val, self.edge_index_val)
            teacher_logits_test  = self.teacher_model(self.x_test, self.edge_index_test)
            student_logits_train_eval = self.forward(self.x_train, self.edge_index_train, dropout=0, mode='eval')
            student_logits_val        = self.forward(self.x_val, self.edge_index_val, dropout=0, mode='eval')
            student_logits_test       = self.forward(self.x_test, self.edge_index_test, dropout=0, mode='eval')


        if config.kd_subgraphs_only==False:
            self.distillation_loss_   = self.distillation_loss(student_logits_train, teacher_logits_train)
            self.classification_loss = F.nll_loss(student_logits_train, self.y_train)
        
        if config.kd_train_on_subgraphs==True:
            probas_dict_student = {}
            probas_dict_teacher = {}
            for sig in self.subgraph_dict.keys():
                subgraph = self.subgraph_dict[sig]['subgraph']
                student_logits = self.forward(subgraph.x, subgraph.edge_index, dropout=config.node_classifier_kwargs['dropout_subgraphs'])
                probas_dict_student[sig]= student_logits.clone().exp()
                self.teacher_model.eval()
                teacher_logits = self.teacher_model(subgraph.x, subgraph.edge_index, dropout=config.node_classifier_kwargs['dropout_subgraphs'])
                probas_dict_teacher[sig]= teacher_logits.clone().exp()
                self.distillation_loss_ += self.distillation_loss(student_logits, teacher_logits)
                self.classification_loss += F.nll_loss(student_logits, subgraph.y)
                del subgraph, student_logits, teacher_logits
            self.loss = self.alpha * self.distillation_loss_ + (1 - self.alpha) * self.classification_loss

        self.loss = self.alpha * self.distillation_loss_ + (1 - self.alpha) * self.classification_loss

        self.acc_trn_KD  = accuracy(student_logits_train_eval, teacher_logits_train.argmax(dim=1),verbose=False)
        self.acc_val_KD  = accuracy(student_logits_val, teacher_logits_val.argmax(dim=1),verbose=False)
        self.acc_test_KD  = accuracy(student_logits_test, teacher_logits_test.argmax(dim=1),verbose=False)
        self.acc_trn  = accuracy(student_logits_train_eval, self.y_train,verbose=False)
        self.acc_val  = accuracy(student_logits_val, self.y_val,verbose=False)
        self.acc_test = accuracy(student_logits_test, self.y_test,verbose=False)
        del teacher_logits_train, teacher_logits_val, teacher_logits_test
        del student_logits_train_eval, student_logits_val, student_logits_test
        try:
            del tudent_logits_train
        except:
            pass
        self.backward([self.loss], verbose=False, retain_graph=False)
        return self.loss

    def forward(self, x, edge_index, dropout, mode='train'):
        assert mode in ['train','eval']
        if mode=='train':
            self.node_classifier.train()
            log_logits = self.node_classifier(x, edge_index, dropout)
        elif mode=='eval':
            self.node_classifier.eval()
            log_logits = self.node_classifier(x, edge_index, dropout)
        return log_logits
    
    def backward(self, losses, verbose=False, retain_graph=False):
        self.loss = sum(losses)
        if self.use_pcgrad==True:
            self.optimizer.pc_backward(losses)
            if verbose==True:
                print(f"Epoch {self.epoch}: PCGrad backpropagation for multiple losses")
        elif self.use_pcgrad==False:
            self.loss.backward(retain_graph=retain_graph)
            if verbose==True:
                print(f"Epoch {self.epoch}: Regular backpropagation for multiple losses")


    def get_weighted_losses(self, type_='primary', loss_primary=None, loss_watermark=None):
        self.loss_dict['loss_primary']=loss_primary
        self.loss_dict['loss_watermark']=loss_watermark
        assert type_ in ['primary','combined']
        if type_=='primary':
            assert loss_primary is not None
            self.loss_dict['loss_primary_weighted']=loss_primary
        elif type_=='combined':
            assert loss_watermark is not None
            ##
            assert self.coefWmk is not None
            loss_watermark_weighted = loss_watermark*self.coefWmk 
            self.loss_dict['loss_primary_weighted'] = loss_primary 
            self.loss_dict['loss_watermark_weighted'] = loss_watermark_weighted
        unweighted_total = torch_add_not_None([self.loss_dict[k] for k in ['loss_primary','loss_watermark']])
        weighted_total = torch_add_not_None([self.loss_dict[k] for k in ['loss_primary_weighted','loss_watermark_weighted']])
        return self.loss_dict, unweighted_total, weighted_total



    def apply_watermark_(self):
        watermark_type = config.watermark_kwargs['watermark_type']
        len_watermark = int(config.watermark_kwargs['percent_of_features_to_watermark']*self.num_features/100)
        subgraph_x_concat = torch.concat([self.subgraph_dict[k]['subgraph'].x for k in self.subgraph_dict.keys()])
        self.subgraph_dict, self.each_subgraph_watermark_indices, self.each_subgraph_feature_importances, watermarks = apply_watermark(watermark_type, self.num_features, len_watermark, self.subgraph_dict, subgraph_x_concat,
                                                                                                                                       self.probas_dict, config.watermark_kwargs, seed=config.seed)
        del subgraph_x_concat
        torch.cuda.empty_cache()
        for i, subgraph_sig in enumerate(self.subgraph_dict_not_wmk.keys()):
            self.subgraph_dict_not_wmk[subgraph_sig]['watermark']=watermarks[i]
        del watermarks
        # return subgraph_dict

    def test_watermark(self):    
        is_last_epoch = self.epoch==self.epochs-1
        beta_weights = get_beta_weights(self.subgraph_dict, self.num_features)
        self.probas_dict = separate_forward_passes_per_subgraph(self.subgraph_dict, self.node_classifier, 'eval')
        self.probas_dict_not_wmk = separate_forward_passes_per_subgraph(self.subgraph_dict_not_wmk, self.node_classifier, mode='eval')
        self.apply_watermark_()
        watermarked_subgraph_results = get_watermark_performance(self.probas_dict, 
                                                                 self.subgraph_dict, 
                                                                 {k:[] for k in self.subgraph_dict.keys()}, 
                                                                 {k:None for k in self.subgraph_dict.keys()}, 
                                                                 is_last_epoch,
                                                                 False, beta_weights#, penalize_similar_subgraphs=False, shifted_subgraph_loss_coef=0
                                                                 )


        beta_weights = get_beta_weights(self.subgraph_dict_not_wmk, self.num_features)
        unwatermarked_subgraph_results = get_watermark_performance(self.probas_dict_not_wmk, 
                                                                   self.subgraph_dict_not_wmk, 
                                                                   {k:[] for k in self.subgraph_dict_not_wmk.keys()}, 
                                                                   {k:None for k in self.subgraph_dict_not_wmk.keys()}, 
                                                                   is_last_epoch,
                                                                   False, beta_weights#, penalize_similar_subgraphs=False, shifted_subgraph_loss_coef=0
                                                                   )
        return watermarked_subgraph_results, unwatermarked_subgraph_results

