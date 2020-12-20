# embedding_datagen.py
# @author : Ramana

import numpy as np
import torch
import os.path as osp
import json

from glob import glob
from copy import deepcopy
from matplotlib.pyplot import imread
from torch.utils.data import Dataset
from collections import defaultdict
from random import shuffle
from PIL import Image



class EmbeddingDataset(Dataset):

    def __init__(self, base_path, mode='train', n_classes=37,
                 n_neg_samples=3):

        """
        This Class is responsible for providing the model with the required 
        batch input. It basically parses the essential JSON files, which contains
        Visual/Word vectors corresponding to each frame. The `base_path` is also
        expected to have other required JSON files such as,

        1) `objects_inv.json` : which contains information of the frames where object are
                                present
        2) `relationship_inv.json` : that stores information of frames where relationship
                                     are present.

        3) `det_withrel_%.json`%(train, test) : JSON file with various relationships/objects
                                                present in each frame along with bbox info.

        The above three are necessary to perform negative sampling, where we know for certain
        an object/relationship is a negative word indeed. Negative sampling essentially takes in
        word2vec of a word that's not present using this info.

        Due to memory issues, image feature's JSON files are separated into train and test.
        For the same reason, subject/object/relationship are also separated.

        Parameters
        ----------

        base_path : `str`, where all the precomputed vectors are present. 
        n_classes : `int`, default = 37
        n_neg_samples : `int`, default = 3


        Returns
        -------
        A `batch_input` dictionary that contains the following,
        1) Lables for (a) Subject, (b) Object, (c) Relation and (d) SRO,
            represented as `labels_s/r/o/sro`.
        2) Appearance feature vectors of (a) Subject, (b) Object, 
            concatenated into a single tensor, represented as 
            labels['precompappearance'].
        3) Bounding boxes of subject and object horizontally stacked,
            represented as `pair_objects`.
        4) Word2vec embedding of S,R,O and SRO represented as 
            `word_vec_s/r/o/sro`.

        """

        self.mode = mode

        ## Check if Image and Word feature directory exists.
        image_feat_dir = osp.join(base_path, 'ImageFeatures')
        word_feat_dir = osp.join(base_path, 'WordFeatures')
        assert osp.isdir(image_feat_dir), "ImageFeatures not correctly placed."
        assert osp.isdir(word_feat_dir), "Word Features not correctly placed."
        

        ext = '_train.json' if mode == 'train' else '_test.json'
        self.det_dict = json.load(open(osp.join(base_path, 'Annotation', 'det_withrel' + ext), 'r'))
        # Remove the SRO that's not in the train set, from test
        self._clean_dict()

        # Load Word Features stored in JSON
        self.vp_wordvec = json.load(open(osp.join(word_feat_dir, 'visual_phrase.json'), 'r'))
        self.r_wordvec = json.load(open(osp.join(word_feat_dir, 'predicate.json'), 'r'))
        self.o_wordvec = json.load(open(osp.join(word_feat_dir, 'object.json'), 'r'))
        self.s_wordvec = json.load(open(osp.join(word_feat_dir, 'subject.json'), 'r'))

        # Load Image Features stored in JSON
        self.s_imgvec = json.load(open(osp.join(image_feat_dir, 'subject' + ext), 'r'))
        self.o_imgvec = json.load(open(osp.join(image_feat_dir, 'object' + ext), 'r'))

        # Get Inverse dictionary --> Object to frame relationship for negative sampling
        self.cls_inv = json.load(open(osp.join(base_path, 'Annotation', 'objects_inv.json'), 'r'))
        self.rel_inv = json.load(open(osp.join(base_path, 'Annotation', 'relationship_inv.json'), 'r'))


        ############################################################
        ##Get only the intersection frames. 
        ## This is due to the fact some(5) SROs don't exist in Test
        ############################################################

        vp_word_frames = set(list(self.vp_wordvec.keys()))
        p_word_frames  = set(list(self.r_wordvec.keys()))
        o_word_frames  = set(list(self.o_wordvec.keys()))
        s_img_frames   = set(list(self.s_imgvec.keys()))
        o_img_frames   = set(list(self.o_imgvec.keys()))
        # gt_frames
        gt_frames       = set(list(self.det_dict.keys()))
        self.all_frames = list(set.intersection(vp_word_frames, p_word_frames,
                                                o_word_frames,
                                                s_img_frames, o_img_frames,
                                                gt_frames))

        self.n_classes = n_classes
        self.n_neg_samp = n_neg_samples


    def _clean_dict(self):
        """
        Remove frames containing an SRO
        that is not present in training set.
        """
        remove_test=("DVT6L_000305", "FXC28_000585",
                     "G8E71_000460", "F1OMV_000259")
        if self.mode != 'train':
            [self.det_dict.pop(k, None) for k in remove_test]


    def _rectify_keys(self, det_json):
        """
        In `det_json`, change keys' format from 
        {`video.mp4/frame.png` : ...} to {`video_frame`: ...}
        """
        det_dict = json.load(open(det_json, 'r'))
        len_before = len(det_dict)
        for k, v in det_dict.items():
            cur_val = det_dict.pop(k)
            new_key = '_'.join([k.split('.')[0] for k in k.split('/')[-2:]])
            det_dict[new_key] = cur_val

        # Make sure we didn't lose any keys
        assert len(det_dict) == len_before
        return det_dict


    def __len__(self):
        return len(self.all_frames)


    def get_pari_obj(self, cur_det, cur_labels):
        """
        Convert detection to (x,y,w,h) format and then,
        stack subject and object into same tensor.
        """

        obj_cat         = [i for i in cur_labels if i != 1][0]
        cur_det_1       = np.array(cur_det[cur_labels.index(1)])
        cur_det_2       = np.array(cur_det[1-cur_labels.index(1)])
        # Convert to W,H
        cur_det_1[2:4] -= cur_det_1[0:2]
        cur_det_2[2:4] -= cur_det_2[0:2]


        pair_obj = np.vstack((np.r_[cur_det_1, 1, 1],
                              np.r_[cur_det_2, obj_cat, 1]))
        return pair_obj


    def get_neg_wordvec_s(self, n_neg=3):
        """
        Sample negative vectors for subject.
        """
        rand_ind = np.random.randint(0, len(self.all_frames), size=n_neg)
        n_word_vec = np.array([self.o_wordvec[self.all_frames[k]] for k in rand_ind]).reshape(n_neg, 1, 300)
        return n_word_vec


    def get_neg_wordvec_o(self, cur_obj=None, n_neg=3):
        """
        Sample negative vectors for object.
        """

        pos_frames = self.cls_inv[cur_obj]
        neg_frame = list(set(self.all_frames) - set(pos_frames))
        rand_ind = np.random.randint(0, len(neg_frame), size=n_neg)
        n_word_vec = np.array([self.o_wordvec[neg_frame[k]] for k in rand_ind]).reshape(n_neg, 1, 300)
        return n_word_vec


    def get_neg_wordvec_r(self, cur_rel=None, n_neg=3):
        """
        Sample negative vectors for relationship.
        """

        pos_frames = self.rel_inv[cur_rel]
        neg_frame = list(set(self.all_frames) - set(pos_frames))
        rand_ind = np.random.randint(0, len(neg_frame), size=n_neg)
        n_word_vec = np.array([self.r_wordvec[neg_frame[k]] for k in rand_ind]).reshape(n_neg, 1, 300)
        return n_word_vec


    def __getitem__(self, idx):
        # constructing similar to Julia's code
        input_dict = {}

        cur_fr      = self.all_frames[idx]
        cur_gt      = self.det_dict[cur_fr]
        cur_labels  = cur_gt.get('labels')
        cur_det     = cur_gt.get('bbox')
        cur_rel     = cur_gt.get('relation')[0]
        cur_obj     = [i for i in cur_labels if i != 1][0]
        cur_obj_cls = [i for i in cur_gt['class_name'] if i != 'person'][0]

        # Get pair, which is bounding box, for spatial configuration layer
        pair_object = self.get_pari_obj(cur_det, cur_labels)
        input_dict['pair_objects']       = np.repeat(pair_object[np.newaxis, :, :], self.n_neg_samp+1, axis=0)

        """
        First input is positive, rest 3 are negative. Hence, [1] is stacked with [0]s.
        The shuffling happens inside DataLoader and our batch size is 16,
        the stochasticity will be sufficient.
        """
        neg_label                        = np.concatenate(([[1]],
                                                        np.zeros((self.n_neg_samp, 1),
                                                        dtype=np.int64)), axis=0)

        input_dict['labels_s']           = deepcopy(neg_label)
        input_dict['labels_o']           = deepcopy(neg_label)
        input_dict['labels_r']           = deepcopy(neg_label)
        input_dict['labels_sro']         = deepcopy(neg_label)

        # Stack appearance feature
        app_feat                         = np.vstack((self.s_imgvec[cur_fr],
                                                      self.o_imgvec[cur_fr]))
        input_dict['precompappearance']  = np.repeat(app_feat[np.newaxis, :, :],
                                                     self.n_neg_samp+1, axis=0)

        # Stack word features, followed by negative sampels
        p_word_vec_s                     = np.array(self.s_wordvec[cur_fr]).reshape(1, 1, 300)
        p_word_vec_o                     = np.array(self.o_wordvec[cur_fr]).reshape(1, 1, 300)
        p_word_vec_r                     = np.array(self.r_wordvec[cur_fr]).reshape(1, 1, 300)
        p_word_vec_sro                   = np.hstack((p_word_vec_s,
                                                      p_word_vec_o,
                                                      p_word_vec_r))
        n_word_vec_s                     = self.get_neg_wordvec_s(n_neg=self.n_neg_samp)
        n_word_vec_o                     = self.get_neg_wordvec_o(cur_obj_cls, n_neg=2*self.n_neg_samp)
        n_word_vec_r                     = self.get_neg_wordvec_r(cur_rel, n_neg=2*self.n_neg_samp)

        input_dict['word_vec_s']         = np.concatenate((p_word_vec_s,
                                                            n_word_vec_s), axis=0)
        input_dict['word_vec_o']         = np.concatenate((p_word_vec_o,
                                                            n_word_vec_o[self.n_neg_samp:]), axis=0)
        input_dict['word_vec_r']         = np.concatenate((p_word_vec_r,
                                                            n_word_vec_r[self.n_neg_samp:]), axis=0)

        # Take a new pair of O,R for SRO negative sampling.
        neg_o_sro                        = np.concatenate((p_word_vec_o,
                                                            n_word_vec_o[self.n_neg_samp:]), axis=0)
        neg_r_sro                        = np.concatenate((p_word_vec_r,
                                                            n_word_vec_r[self.n_neg_samp:]), axis=0)
        input_dict['word_vec_sro']       = np.concatenate((input_dict['word_vec_s'],
                                                           neg_r_sro,
                                                           neg_o_sro), axis=1)

        
        return input_dict['pair_objects'], input_dict['labels_s'], input_dict['labels_o'], input_dict['labels_r'], \
               input_dict['labels_sro'], input_dict['precompappearance'], input_dict['word_vec_s'], input_dict['word_vec_o'], \
               input_dict['word_vec_r'], input_dict['word_vec_sro']



    def collate_fn(self, data):
        """Creates mini-batch tensors from the list of tuples ().
        """
        pair_objects, labels_s, labels_o, labels_r, labels_sro,\
        precompappearance, word_vec_s, word_vec_o, word_vec_r, word_vec_sro  = zip(*data)
        
        output = {}


        output['pair_objects'] = torch.from_numpy(np.concatenate(pair_objects, axis=0)).float()
        output['labels_s']     = torch.from_numpy(np.concatenate(labels_s, axis=0)).long()
        output['labels_o']     = torch.from_numpy(np.concatenate(labels_o, axis=0)).long()
        output['labels_r']     = torch.from_numpy(np.concatenate(labels_r, axis=0)).long()
        output['labels_sro']   = torch.from_numpy(np.concatenate(labels_sro, axis=0)).long()
        # Word features
        output['word_vec_s']     = torch.from_numpy(np.concatenate(word_vec_s, axis=0)).float()
        output['word_vec_o']     = torch.from_numpy(np.concatenate(word_vec_o, axis=0)).float()
        output['word_vec_r']     = torch.from_numpy(np.concatenate(word_vec_r, axis=0)).float()
        output['word_vec_sro']   = torch.from_numpy(np.concatenate(word_vec_sro, axis=0)).float()

        output['precompappearance'] = torch.from_numpy(np.concatenate(precompappearance, axis=0)).float()


        return output

