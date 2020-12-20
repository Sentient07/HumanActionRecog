# evaluate.py
# @author : Ramana

import tensorboard_logger
import torch
import torch.nn as nn
import numpy as np
import time
import argparse
import os.path as osp
import os
import warnings
import matplotlib.pyplot as plt
import itertools


from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import precision_recall_curve
from torch.autograd import Variable

from utils import AverageMeter, Tb_logger
from data import EmbeddingDataset
from net import NetIndepEmb

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Evaluation of embedding')
# Dataset related arguments
parser.add_argument('--data_path', required=True,
                    type=str,help='Path to detection file')
parser.add_argument('--save_dir', required=True,
                    type=str, help='Path to save Plots')

parser.add_argument('--pt_path', required=True,
                    type=str,help='Path to PT model')

parser.add_argument('--num_class', default=37, type=int, help='Classes(O+S+BG)')
parser.add_argument('--embed_size', default=1024, type=int, help='Embedding size')
parser.add_argument('--num_neg', default=3, type=int, help='Num of negative samples.')


parser.add_argument('--num_workers', default=32, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--batch_size', default=8, type=int, help='Batch Size')

args = parser.parse_args()


def evaluate(val_loader, device=torch.device('cuda')):

    model.eval()

    batch_time = 0
    test_loss = {}
    test_recall = {}
    test_precision = {}


    split = 'test'
    start = time.time()

    for batch_idx, batch_input in enumerate(val_loader):

        for key in batch_input.keys():
            batch_input[key] = Variable(batch_input[key].to(device))

        # Eval
        loss, tp_class, fp_class, num_pos_class = model.val_(batch_input)

        batch_time += time.time() - start
        start = time.time()

        # Performance per gram
        for gram in tp_class.keys():

            recall = np.nanmean(tp_class[gram].numpy()/num_pos_class[gram].numpy())
            precision = np.nanmean(tp_class[gram].numpy() / (tp_class[gram].numpy() + fp_class[gram].numpy()))

            if gram not in test_recall.keys():
                test_recall[gram] = AverageMeter()

            if gram not in test_precision.keys():
                test_precision[gram] = AverageMeter()

            if gram not in test_loss.keys():
                test_loss[gram] = AverageMeter()

            test_recall[gram].update(recall, n=batch_input['pair_objects'].size(0))
            test_precision[gram].update(precision, n=batch_input['pair_objects'].size(0))
            test_loss[gram].update(loss[gram].data, n=batch_input['pair_objects'].size(0))

    # Save total loss on test
    total_test_loss = 0

    for _, val in test_loss.items():
        total_test_loss += val.avg

    tb_logger[split].log_value('loss', total_test_loss, 1)
    tb_logger[split].log_value('batch_time', batch_time/len(val_loader), 1)

    # Total performance per gram
    recall_gram = {}
    loss_gram = {}
    precision_gram = {}
    recall_gram = {}

    for gram in tp_class.keys():

        tb_logger[split].log_value(gram+'_loss', test_loss[gram].avg, 1)
        tb_logger[split].log_value(gram+'_mean_recall', 100.*test_recall[gram].avg, 1)
        tb_logger[split].log_value(gram+'_mean_precision', 100.*test_precision[gram].avg, 1)
        recall_gram[gram]    = test_recall[gram]
        precision_gram[gram] = test_precision[gram]
        loss_gram[gram]      = test_loss[gram].avg

    print("Average Precision : " + str(np.nan_to_num(np.nanmean([i.avg for i in precision_gram.values()]))))
    # print('{} set: Average loss: {:.4f}, Recall: ({:.0f}%)'.format(split, sum(loss_gram.values()), \
    #                                 100. * np.mean([i.avg for i in test_recall.values()])))


    for gram in tp_class.keys():
        test_loss[gram].reset()


    return loss_gram, precision_gram, recall_gram


def plot_pr_curve(model, loader, device=torch.device('cuda')):

    all_label, all_score = defaultdict(list), defaultdict(list)
    for batch_idx, batch_input in enumerate(loader):
        for key in batch_input.keys():
            batch_input[key] = Variable(batch_input[key].to(device))
        batch_scores = model.get_scores(batch_input)
        for gram in batch_scores.keys():
            cur_label = batch_input['labels_'+gram].cpu().detach().numpy()
            all_label[gram].append(np.squeeze(cur_label).tolist())
            if isinstance(batch_scores[gram], torch.Tensor):
                all_score[gram].append(np.squeeze(batch_scores[gram].cpu().detach().numpy()).tolist())

    for gram in ['s', 'o', 'r', 'sro']:
        precision, recall, thresholds = precision_recall_curve(list(itertools.chain(*all_label[gram])),
                                                               list(itertools.chain(*all_score[gram])))
        fig, ax = plt.subplots()
        ax.plot(recall, precision, label='Test Set')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(loc='center left')
        fig_name = 'pr_curve_' + gram + '_test.png'
        plot_save_dir = osp.join(args.save_dir, 'Plots')
        if not osp.exists(plot_save_dir):
            os.makedirs(plot_save_dir, exist_ok=True)
        plt.savefig(osp.join(plot_save_dir, fig_name))


if __name__ == '__main__':

    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    ### Initialize logger to view perf curves ####
    log = Tb_logger()
    logger_path = osp.join('./Logs', 'Test')
    tb_logger = log.init_logger(logger_path, ['test'])

    ####################
    """ Data loader """
    ####################
    # Test split
    test_dataset = EmbeddingDataset(args.data_path,
                                    mode='test',
                                    n_classes  = args.num_class,
                                    n_neg_samples = args.num_neg)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size = args.batch_size, \
                                              shuffle = True, \
                                              num_workers = args.num_workers, \
                                              collate_fn = test_dataset.collate_fn)
    ########################################
    """ Define model and load checkpoint"""
    ########################################
    model = NetIndepEmb(embed_size=args.embed_size)
    if torch.cuda.is_available():
        model.cuda()
    checkpoint = torch.load(args.pt_path)
    model.load_pretrained_weights(checkpoint['model'])

    ########################################
    """ Perform evaluation."""
    ########################################
    model.eval()
    loss_test, precision_test, recall_test = evaluate(test_loader)
    loss_test = torch.mean(torch.Tensor(list(loss_test.values()))).item()
    precision_test = np.nan_to_num(np.nanmean([i.avg for i in precision_test.values()]))
    recall_test = np.nan_to_num(np.nanmean([i.avg for i in recall_test.values()]))
    plot_pr_curve(model, test_loader)