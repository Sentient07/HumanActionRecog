# train_embedding.py
# @author : Ramana

import tensorboard_logger
import torch
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


"""
Parsing options
"""

parser = argparse.ArgumentParser(description='Training of embedding')
# Dataset related arguments
parser.add_argument('--data_path', required=True,
                    type=str,help='Path to detection file')
parser.add_argument('--save_dir', required=True,
                    type=str, help='Path to save weights')
parser.add_argument('--exp_name', required=True,
                    type=str, help='Name for experiment')

parser.add_argument('--pt_path',
                    type=str,help='Path to PT model')

parser.add_argument('--num_class', default=37, type=int, help='Classes(O+S+BG)')

parser.add_argument('--embed_size', default=1024, type=int, help='Embedding size')
parser.add_argument('--num_neg', default=3, type=int, help='Num of negative samples')

parser.add_argument("--plot_pr", action='store_true', help='Whether or not to plot PR curve')


parser.add_argument('--learning_rate', default=1e-3, type=float,
                    help='Learning rate')
parser.add_argument('--lr_update', default=5, type=int,
                    help='Number of epoch before decreasing learning rate')
parser.add_argument('--start_epoch', default=1, type=int,
                    help='epochs for starting')
parser.add_argument('--num_workers', default=32, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--batch_size', default=16, type=int, help='Batch Size')
parser.add_argument('--nepoch', default=10, type=int,
                    help='Number of epochs for training')

args = parser.parse_args()


"""
Train / val
"""

def train(epoch, train_loader, device=torch.device('cuda')):

    batch_time = 0
    train_loss = {}
    train_recall = {}
    train_precision = {}


    model.train()
    split = 'train'

    start_time = time.time()
    start = time.time()

    for batch_idx, batch_input in enumerate(train_loader):

        for key in batch_input.keys():
            batch_input[key] = Variable(batch_input[key].to(device))

        # Train
        loss, tp_class, fp_class, num_pos_class = model.train_(batch_input)


        batch_time += time.time() - start
        start = time.time()

        # True pos/false pos per branch
        for gram in tp_class.keys():

            recall = np.nanmean(tp_class[gram].numpy()/num_pos_class[gram].numpy())
            precision = np.nanmean(tp_class[gram].numpy() / (tp_class[gram].numpy() + fp_class[gram].numpy()))

            if gram not in train_recall.keys():
                train_recall[gram] = AverageMeter()

            if gram not in train_precision.keys():
                train_precision[gram] = AverageMeter()

            if gram not in train_loss.keys():
                train_loss[gram] = AverageMeter()


            train_recall[gram].update(recall, n=batch_input['pair_objects'].size(0))
            train_precision[gram].update(precision, n=batch_input['pair_objects'].size(0))
            train_loss[gram].update(loss[gram].data, n=batch_input['pair_objects'].size(0))


        learning_rate = model.optimizer.param_groups[0]['lr']

        if batch_idx % 100 ==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tDone in: {:.2f} sec'.format(epoch, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), sum(loss.values()).data, (time.time()-start_time)))
            start_time = time.time()
        
        
        # Record logs in tensorboard
        if model.epoch % 500 ==0:

            batch_time /= 500
           
            total_train_loss = 0

            for _, val in train_loss.items():
                total_train_loss += val.avg

            # Register in logger 
            tb_logger[split].log_value('epoch', epoch, model.epoch)
            tb_logger[split].log_value('loss', total_train_loss, model.epoch)
            tb_logger[split].log_value('batch_time', batch_time, model.epoch)
            tb_logger[split].log_value('learning_rate', learning_rate, model.epoch)
            tb_logger[split].log_value('weight_decay', 0, model.epoch)

            for gram in tp_class.keys():
                tb_logger[split].log_value(gram+'_loss', train_loss[gram].avg, model.epoch)
                tb_logger[split].log_value(gram+'_mean_recall', 100.*train_recall[gram].avg, model.epoch)
                tb_logger[split].log_value(gram+'_mean_precision', 100.*train_precision[gram].avg, model.epoch)

            batch_time = 0

        model.epoch += 1
        
    for gram in tp_class.keys():
        train_loss[gram].reset()



def evaluate(epoch, val_loader, device=torch.device('cuda')):

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

    tb_logger[split].log_value('epoch', epoch, model.epoch)
    tb_logger[split].log_value('loss', total_test_loss, model.epoch)
    tb_logger[split].log_value('batch_time', batch_time/len(val_loader), model.epoch)

    # Total performance per gram
    recall_gram = {}
    loss_gram = {}
    precision_gram = {}
    recall_gram = {}

    for gram in tp_class.keys():

        tb_logger[split].log_value(gram+'_loss', test_loss[gram].avg, model.epoch)
        tb_logger[split].log_value(gram+'_mean_recall', 100.*test_recall[gram].avg, model.epoch)
        tb_logger[split].log_value(gram+'_mean_precision', 100.*test_precision[gram].avg, model.epoch)
        recall_gram[gram]    = test_recall[gram]
        precision_gram[gram] = test_precision[gram]
        loss_gram[gram]      = test_loss[gram].avg

    print("Average Precision : " + str(np.nan_to_num(np.nanmean([i.avg for i in precision_gram.values()]))))
    # print('{} set: Average loss: {:.4f}, Recall: ({:.0f}%)'.format(split, sum(loss_gram.values()), \
    #                                 100. * np.mean([i.avg for i in test_recall.values()])))


    for gram in tp_class.keys():
        test_loss[gram].reset()


    return loss_gram, precision_gram, recall_gram


def plot_pr_curve(model, loader, epoch, device=torch.device('cuda')):

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
        fig_name = 'pr_curve_' + gram + '_' + str(epoch) + '.png'
        plot_save_dir = osp.join(args.save_dir, 'Plots')
        if not osp.exists(plot_save_dir):
            os.makedirs(plot_save_dir, exist_ok=True)
        plt.savefig(osp.join(plot_save_dir, fig_name))



if __name__ == '__main__':
    # Init logger
    log = Tb_logger()
    logger_path = osp.join('./Logs', args.exp_name)


    tb_logger = log.init_logger(logger_path, ['train', 'test'])

    ####################
    """ Data loaders """
    ####################


    # Train split
    train_dataset = EmbeddingDataset(args.data_path,
                                     mode='train',
                                     n_classes  = args.num_class,
                                     n_neg_samples = args.num_neg)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size, \
                                               shuffle = True, \
                                               num_workers = args.num_workers, \
                                               collate_fn = train_dataset.collate_fn)

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

    ####################
    """ Define model """
    ####################
    model = NetIndepEmb(embed_size=args.embed_size)

    if torch.cuda.is_available():
        model.cuda()

    # Load pre-trained model
    if args.pt_path:
        assert args.start_epoch, 'Indicate epoch you start from'
        if args.start_epoch:
            checkpoint = torch.load(args.pt_path, map_location=lambda storage, loc: storage)
            model.load_pretrained_weights(checkpoint['model']) 



    print('Train classifier')
    best_precision = 0
    for epoch in tqdm(range(args.nepoch)):
        epoch_effective = epoch + args.start_epoch + 1

        # Train
        model.adjust_learning_rate(args.learning_rate, args.lr_update, epoch)
        train(epoch, train_loader)

        # Val
        loss_test, precision_test, recall_test = evaluate(epoch, test_loader)

        loss_test = torch.mean(torch.Tensor(list(loss_test.values()))).item()
        precision_test = np.nan_to_num(np.nanmean([i.avg for i in precision_test.values()]))
        recall_test = np.nan_to_num(np.nanmean([i.avg for i in recall_test.values()]))

        if args.plot_pr:
            plot_pr_curve(model, test_loader, epoch)


        state = {\
                'epoch':epoch_effective,
                'model':model.state_dict(),
                'loss':loss_test,
                'precision':precision_test,
                'recall':recall_test,
                }

        # Save every epoch.
        save_dir = osp.join(args.save_dir, args.exp_name)
        if not osp.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        torch.save(state, osp.join(save_dir, 'model_' + 'epoch' + str(epoch_effective) + '.pth'))

        # Save with a different name for model with higher recall
        if precision_test > best_precision:
            state = {
                    'epoch':epoch_effective,
                    'model':model.state_dict(),
                    'min_loss':loss_test,
                    'precision':precision_test,
                    'recall':recall_test,
                    }
            torch.save(state, osp.join(save_dir, 'model_best.pth'))
            best_recall = recall_test
        
