import os
import numpy as np
import matplotlib.pyplot as plt

import models_mae_finetune
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils import de_norm
import utils
import torch

# !/usr/bin/env bash


from argparse import ArgumentParser


class CDEvaluator():

    def __init__(self, args, dataloader):

        self.dataloader = dataloader

        self.n_class = args.n_class
        # define G
        self.net_G = models_mae_finetune.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids) > 0
                                   else "cpu")
        print(self.device)

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        # define logger file
        logger_path = './log_test.txt'
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)

        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)

        self.G_pred = None
        self.pred_vis = None
        self.label=None
        self.img_in1=None
        self.img_in2=None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = '/home/yanghualin/zhangwen/mae-main_1/vis'

    def _load_checkpoint(self, checkpoint_dir='/home/yanghualin/zhangwen/mae-main_1/output_dir/checkpoint-99.pth'):

        if os.path.exists(checkpoint_dir):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(checkpoint_dir, map_location='cpu')

            self.net_G.load_state_dict(checkpoint['model'])

            self.net_G.to(self.device)

    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _update_metric(self):
        """
        update metric
        """
        target = self.label.detach()
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' % \
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

        if np.mod(self.batch_id, 100) == 1:
            vis_input = utils.make_numpy_grid(de_norm(self.img_in1))
            vis_input2 = utils.make_numpy_grid(de_norm(self.img_in2))

            vis_pred = utils.make_numpy_grid(self._visualize_pred())

            vis_gt = utils.make_numpy_grid(self.label)
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(
                self.vis_dir, 'eval_' + str(self.batch_id) + '.jpg')
            plt.imsave(file_name, vis)

    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        np.save(os.path.join('/home/yanghualin/zhangwen/mae-main_1/finetune_output', 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['mf1']

        with open(os.path.join('/home/yanghualin/zhangwen/mae-main_1/finetune_output', '%s.txt' % (self.epoch_acc)),
                  mode='a') as file:
            pass

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        img_in1,img_in2,label=batch

        img_in1 = img_in1.to(self.device)
        img_in2 = img_in2.to(self.device)
        label = label.to(self.device)
        self.label=label
        self.img_in1=img_in1
        self.img_in2=img_in2
        self.G_pred, _ = self.net_G(img_in1, img_in2, label)

    def eval_models(self, checkpoint_dir='best_ckpt.pt'):

        self._load_checkpoint(checkpoint_dir)

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states()
        self._collect_epoch_states()


def get_device(args):
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])


"""
eval the CD model
"""


def main():
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='4', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--split', default="test", type=str)
    parser.add_argument('--checkpoint_dir', default='/home/yanghualin/zhangwen/mae-main_1/output_dir/checkpoint-99.pth',
                        type=str)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--data_name', default='LEVIR', type=str)
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    args = parser.parse_args()
    get_device(args)
    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split=args.split)
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models(checkpoint_dir=args.checkpoint_dir)


if __name__ == '__main__':
    main()

# Decide which device we want to run on
# torch.cuda.current_device()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


