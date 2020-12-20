# @author : Ramana

import torch
import torch.nn as nn
import torch.optim as optim
import os.path as osp
import torch.nn.functional as F
import numpy as np
import pickle

from torch.nn import Parameter
from torch.autograd import Variable
from layers import AppearancePrecomp, SpatialAppearancePrecomp, LanguageProj



class Net(nn.Module):

    def __init__(self, embed_size=1024, d_hidden=1024,
                 num_layers=2, word_input_dim=300,
                 vis_input_dim=1024,
                 learning_rate=1e-3, weight_decay=0,
                 momentum=0, optimizer='adam',
                 normalize_vis=False,
                 normalize_lang=True,
                 optim_name='adam'):

        """
        The aim of Net method is to control training
        related characterstic, restoration of weights and
        store all information about feature vector dimension.

        Parameters
        ----------
            
        embed_size : int, default=1024
            The common embedding size of visual and lingual features.
        
        d_hidden : int, default=1024
            The hidden state vector's dimension of MLP.

        num_layers : int, default=2
            Number of layers in MLP
        
        word_input_dim : int, default=300
            Input dimension of language feature. 

        vis_input_dim : int, default=1024
            Input dimension of vis feature.
        
        normalize_vis : bool, default=False
            Whether or not to apply L2 normalization
            to subject and object layer.

        normalize_lang : bool, default=True
            Whether to apply L2 norm to language feature?

        """
        super(Net, self).__init__()

        self.embed_size           = embed_size
        self.d_hidden             = embed_size
        self.num_layers           = num_layers
        self.word_input_dim       = word_input_dim
        self.vis_input_dim        = vis_input_dim

        # Optim parameters
        self.learning_rate = learning_rate
        self.weight_decay  = weight_decay
        self.momentum      = momentum
        self.optim_name    = optimizer

        self.normalize_vis    = normalize_vis
        self.normalize_lang   = normalize_lang

        # Nets
        self.modules_name = {}
        self.modules_name['s']   = 'appearanceprecompsubject'
        self.modules_name['o']   = 'appearanceprecompobject'
        self.modules_name['r']   = 'spatialappearanceprecomp'
        self.modules_name['sro'] = 'spatialappearanceprecomp'

        self.get_activated_grams()

        print("Using embed size as : " + str(self.embed_size))
        self.epoch = 0


    def get_activated_grams(self):
        """
        Used for initializing Appearance and Lang feat network
        """
        self.activated_grams = {'s': 0,
         'o': 0,
         'r': 0,
         'sr': 0,
         'ro': 0,
         'sro': 0}
        for gram in self.activated_grams.keys():
            if gram in self.modules_name:
                self.activated_grams[gram] = 1

        if np.array(self.activated_grams.values()).sum() == 0:
            print('Attention we need to activated at least 1 branch of the network')

    def adjust_learning_rate(self, lr, lr_update, epoch):
        """Sets the learning rate to the initial LR
        decayed by 10 every 30 epochs"""
        lr = lr * 0.1 ** (epoch // lr_update)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_optimizer(self):
        if self.optim_name == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optim_name == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            print('Choose optimizer')


    def get_statistics(self, scores, labels):
        """
        Expects scores renormalized [0,1]
        Computes TP, FP and NumPos for AP, Pr.
        """
        activations = (scores > 0.5).float()
        num_pos = labels.data.sum(0).squeeze().cpu()
        tp = (activations * labels.eq(1).float()).sum(0).squeeze().data.cpu()
        fp = (activations * labels.eq(0).float()).sum(0).squeeze().data.cpu()
        return (tp, fp, num_pos)


    def get_module_language(self, num_words=1):

        """
        Create language network
        """

        module = LanguageProj(self.embed_size,
                              num_words=num_words,
                              input_dim=self.word_input_dim,
                              num_layers=self.num_layers,
                              hidden_dimension=None,
                              normalize=self.normalize_lang)

        return module


    def get_module_sub(self):
        """
        Create Visual feat network for subject
        """
        module = AppearancePrecomp(self.embed_size,
                                   'subject',
                                   d_appearance=self.vis_input_dim,
                                   d_hidden=self.d_hidden,
                                   normalize=self.normalize_vis,
                                   num_layers=self.num_layers)
        return module


    def get_module_obj(self):
        """
        Create Visual feat network for Object
        """
        module = AppearancePrecomp(self.embed_size,
                                   'object',
                                   d_appearance=self.vis_input_dim,
                                   d_hidden=self.d_hidden,
                                   normalize=self.normalize_vis,
                                   num_layers=self.num_layers)
        return module

    def get_module_relation(self):

        """
        Create Visual feat network for Spatial Relationship
        """

        module = SpatialAppearancePrecomp(self.embed_size,
                                          d_appearance=self.vis_input_dim,
                                          d_hidden=self.d_hidden,
                                          normalize=self.normalize_vis,
                                          num_layers=self.num_layers)
        return module



    def get_visual_features(self, batch_input, gram):
        """        
        Get visual features for all activated grams:
        vis_feats[gram] : N x d where N is the number of pair of boxes
        """
        vis_feats = self.visual_nets[self.gram_id[gram]](batch_input)

        return vis_feats


    def get_language_features(self, batch_input, gram):
        """
        Pass the word2vec vector into the model.
        """
        word_feat_gram = 'word_vec_' + gram
        language_feats = self.language_nets[self.gram_id[gram]](batch_input[word_feat_gram])
        return language_feats

   
    def compute_similarity(self, vis_feats, language_feats, gram):
        """
        Receives vis_feats[gram] of size (N,d), language_feats[gram] of size (N,M,d)
        When there are too many M queries, split the computation to avoid out-of-memory
        """
        return torch.matmul(vis_feats[:, None, :], language_feats[:, None, :].permute(0, 2, 1))


    def forward(self, batch_input):

        scores = {}
        labels = {}

        for ind, gram in enumerate(self.activated_grams):
            if self.activated_grams[gram]:

                vis_feats             = self.get_visual_features(batch_input, gram) # (N, 1024)
                # queries, labels[gram] = self.get_queries(batch_input, self.sample_negatives, gram)
                labels[gram]          = batch_input['labels_'+gram]
                language_feats        = self.get_language_features(batch_input, gram) # (N, 1024)
                scores[gram]          = self.compute_similarity(vis_feats, language_feats, gram).view(-1, 1)

        return (scores, labels)


    def load_pretrained_weights(self, checkpoint):
        """ Load pretrained network """

        model = self.state_dict()
        for key,_ in model.items():
            if key in checkpoint.keys():
                param = checkpoint[key]
                if isinstance(param, Parameter):
                    param = param.data
                try:
                    model[key].copy_(param)
                except:
                    nn.init.normal_(model[key], mean=0, std=1.0)



class NetIndepEmb(Net):

    def __init__(self, **kwargs):
        super(NetIndepEmb, self).__init__(**kwargs)

        # Compute gram_id to have module <-> gram correspondencies
        self.gram_id = {}
        count = 0
        for gram, is_active in self.activated_grams.items():
            if is_active:
                self.gram_id[gram] = count
                count +=1 

        ######################
        """ Visual network """
        ######################

        self.visual_nets = nn.ModuleList()
        sub_vis = self.get_module_sub()
        obj_vis = self.get_module_obj()
        rel_vis = self.get_module_relation()

        self.visual_nets.append(sub_vis)
        self.visual_nets.append(obj_vis)
        self.visual_nets.append(rel_vis)
        self.visual_nets.append(rel_vis)



        ########################
        """ Language network """
        ########################

        self.language_nets = nn.ModuleList()
        sub_lang = self.get_module_language(num_words=1)
        obj_lang = self.get_module_language(num_words=1)
        rel_lang = self.get_module_language(num_words=1)
        sro_lang = self.get_module_language(num_words=3)

        self.language_nets.append(sub_lang)
        self.language_nets.append(obj_lang)
        self.language_nets.append(rel_lang)
        self.language_nets.append(sro_lang)


        #################
        """ Criterion """
        #################

        self.criterions = {}
        for gram, is_active in self.activated_grams.items():
            if is_active:
                self.criterions[gram] = nn.MultiLabelSoftMarginLoss()


        #################
        """ Optimizer """
        #################

        self.params = [p for p in self.parameters() if p.requires_grad]

        if self.optim_name == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optim_name == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            print('Optimizer not recognized')


    def train_(self, batch_input):

        self.optimizer.zero_grad()
        loss_all = {}
        tp_all = {}
        fp_all = {}
        num_pos_all = {}

        # Gram branches w/o analogy
        scores, labels = self(batch_input)
        for gram, is_active in self.activated_grams.items():
            if is_active:
                loss_all[gram] = self.criterions[gram](scores[gram], labels[gram].float()) 
                activations = F.sigmoid(scores[gram])
                tp_all[gram], fp_all[gram], num_pos_all[gram] = self.get_statistics(activations, labels[gram])


        # Combine losses
        loss = 0
        for _, val in loss_all.items():
            loss += val

        # Gradient step
        loss.backward() 
        self.optimizer.step()
        return (loss_all, tp_all, fp_all, num_pos_all)


    def val_(self, batch_input):
        loss_all = {}
        tp_all = {}
        fp_all = {}
        num_pos_all = {}

        # Gram w/o analogy
        scores, labels = self(batch_input)
        for gram, is_active in self.activated_grams.items():
            if is_active:
                loss_all[gram] = self.criterions[gram](scores[gram], labels[gram].float())
                activations = F.sigmoid(scores[gram])
                tp_all[gram], fp_all[gram], num_pos_all[gram] = self.get_statistics(activations, labels[gram])

        return (loss_all, tp_all, fp_all, num_pos_all)


    def get_scores(self, batch_input):
        """
        Scores produced by independent branches: p(s), p(o), p(r), p(sr), p(ro), p(sro)
        """

        scores_grams, _ = self(batch_input)

        for _, gram in enumerate(self.activated_grams):
            if self.activated_grams[gram]:
                scores_grams[gram] = F.sigmoid(scores_grams[gram])
        return scores_grams


    def test_func(self, batch_input):

        """
        Predict visual visual feature, and compare it
        against existing projections of linguistic feature.

        Return the maximum likelihood index.
        """

        test_grams = ['o', 'r']
        ind_gram = {}

        for gram in test_grams:
            vis_feats             = self.get_visual_features(batch_input, gram)
            language_feats        = self.get_language_features(batch_input, gram)
            scores                = F.sigmoid(self.compute_similarity(vis_feats,
                                                language_feats, gram).view(-1, 1))
            ind_gram[gram]        = torch.argmax(scores).item()

        return ind_gram

