# test.py
# @author : Ramana

import torch
import numpy as np
import argparse
import os.path as osp
import os
import warnings
import json
import cv2

from tqdm import tqdm
from glob import glob
from tqdm import tqdm

from net import NetIndepEmb

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Qualitative testing of Action recognition')
# Dataset related arguments
parser.add_argument('--data_path', required=True,
                    type=str,help='Path to detection file')
parser.add_argument('--save_dir', required=True,
                    type=str, help='Path to save Plots')
parser.add_argument('--im_dir', required=True,
                    type=str, help='Directory with images to test on')
parser.add_argument('--pt_path', required=True,
                    type=str,help='Path to PT model')

args = parser.parse_args()



def get_pari_obj(cur_det, cur_labels):
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
    return pair_obj[np.newaxis, :, :]


if __name__ == '__main__':

    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ###########################
    """ Load All feature vec """
    ############################
    in_dict = {}
    obj_word_feat = json.load(open(osp.join(args.data_path,
                                        'WordFeatures', 'obj_word_feat.json')))
    rel_word_feat = json.load(open(osp.join(args.data_path,
                                        'WordFeatures', 'rel_word_feat.json')))
    sub_img_feat  = json.load(open(osp.join(args.data_path,
                                        'ImageFeatures', 'subject_test.json')))
    obj_img_feat  = json.load(open(osp.join(args.data_path,
                                        'ImageFeatures', 'object_test.json')))
    det_dict      = json.load(open(osp.join(args.data_path, 'Annotation',
                                            'det_withrel_test.json'), 'r'))

    obj_names     = list(obj_word_feat.keys())
    rel_names     = list(rel_word_feat.keys())

    in_dict['word_vec_o'] = torch.from_numpy(np.array(list(obj_word_feat.values())).reshape(-1, 1, 300)).float().to(device)
    in_dict['word_vec_r'] = torch.from_numpy(np.array(list(rel_word_feat.values())).reshape(-1, 1, 300)).float().to(device)



    ########################################
    """ Define model and load checkpoint"""
    ########################################
    model = NetIndepEmb()
    if torch.cuda.is_available():
        model.cuda()
    checkpoint = torch.load(args.pt_path)
    model.load_pretrained_weights(checkpoint['model'])

    ########################################
    """ Perform evaluation."""
    ########################################
    model.eval()
    all_img_files = glob(osp.join(args.im_dir, "*.png"))
    if len(all_img_files) == 0:
        raise ValueError("No images provided in the im_dir")

    for im_file in tqdm(all_img_files):
        """
        Try to get the visual feature
        The png file is of format VIDEO_FRAME.png
        extract VIDEO_FRAME info, which is key in precomputed features.
        """

        im_key = im_file.split('/')[-1].split('.')[0]
        if im_key not in sub_img_feat or im_key not in obj_img_feat:
            print("Not testing image " + str(im_key))
            continue

        cur_gt      = det_dict[im_key]
        cur_labels  = cur_gt.get('labels')
        cur_det     = cur_gt.get('bbox')
        pair_object = get_pari_obj(cur_det, cur_labels)

        in_dict['pair_objects']   = torch.from_numpy(pair_object).float().to(device)
        in_dict['precompappearance'] = torch.from_numpy(
                                                np.vstack((sub_img_feat[im_key],
                                                obj_img_feat[im_key])).reshape(-1, 2, 1024)).float().to(device)

        out_ind  = model.test_func(in_dict)
        pred_r   = rel_names[out_ind['r']]
        pred_o   = obj_names[out_ind['o']]
        pred_string = "Person " + str(pred_r) + " " + pred_o

        cur_im = cv2.imread(im_file)
        cv2.putText(cur_im, pred_string, 
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5,
                    (0,0,139),
                    2)

        # Get the GT label for this class and write that at the bottom of image
        gt_obj = [i for i in cur_gt['class_name'] if i != 'person'][0]
        gt_rel = cur_gt.get('relation')[0]
        gt_string = "Person " + str(gt_rel) + " " + gt_obj
        cv2.putText(cur_im, pred_string, 
                    (10, cur_im.shape[0]-50),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5,
                    (0,139,0),
                    2)
        cv2.imwrite(osp.join(args.save_dir, im_key + '.png'), cur_im)



