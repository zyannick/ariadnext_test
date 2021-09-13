import os

from numpy.lib.npyio import save

# dirpath = os.pardir
import sys

# sys.path.append(dirpath)
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.optim import lr_scheduler, optimizer

from pytorch_balanced_sampler.sampler import SamplerFactory

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import numpy as np

import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
from math import ceil, gamma
from miscellaneous.utils import *
from data_factory.datareader import *
import os
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from termcolor import colored, COLORS
import torch.cuda as cuda
import multiprocessing as mp
import copy
import pickle as pickle
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from time import time
from models.plotting import plot
from torch.nn.modules.distance import PairwiseDistance
from models.plotting import plot_roc_lfw, plot_accuracy_lfw
from backbones.margins import *
from miscellaneous.utils import Visualizer


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


'''
Class containing:
- training functions
- inference
'''
class ImageModel(object):
    def __init__(self,
                 flags,
                 backbone,
                 nb_classes_train,
                 nb_classes_val,
                 datafiles=None,
                 checkpoint_path=None,
                 train_mode=''):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.init_lr = flags.init_lr
        self.network = backbone
        self.datafiles = datafiles
        self.nb_classes_train = nb_classes_train
        self.nb_classes_test = nb_classes_val
        self.verbose = flags.verbose
        self.checkpoint_path = checkpoint_path
        self.save_every = flags.save_every
        self.flags = flags
        self.launch_mode = train_mode
        self.l1_lambda = flags.l1_lambda
        self.num_devices = cuda.device_count()
        self.cur_epoch = 0
        self.current_lr = None
        self.nb_step_without_improving = 0
        self.is_optimised = False
        self.save_epoch_fmt_task = os.path.join(self.checkpoint_path, 'checkpoint_{}ep.pt')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.setup(flags)
        self.setup_path(flags)
        self.configure(flags)
        self.cuda_mode = flags.cuda
        self.history = {'loss': [], 'accuracy': []}
        self.start_training = datetime.now()
        self.end_training = None
        self.train_num_iter = 0
        self.embedding_dimension = flags.embedding_dimension
        self.num_human_identities_per_batch = flags.num_human_identities_per_batch
        self.batch_size = flags.batch_size
        self.lfw_batch_size = flags.lfw_batch_size
        self.resume_path = flags.resume_path
        self.num_workers = flags.num_workers
        self.optimizer = flags.optimizer
        self.margin = flags.margin
        self.use_semihard_negatives = flags.use_semihard_negatives
        self.training_triplets_path = flags.training_triplets_path
        if flags.phase == 'train':
            self.writer = SummaryWriter(log_dir=os.path.join('runs', self.flags.all_parameters + '_' + str(datetime.now())))
        else:
            self.writer = None

    def setup(self, flags):
        torch.backends.cudnn.deterministic = flags.deterministic
        print('torch.backends.cudnn.deterministic:',
              torch.backends.cudnn.deterministic)
        fix_all_seed(flags.launch_seed)

        if flags.freeze:
            self.freeze()

        self.network = self.network.cuda()
        if flags.data_parallel:
            if cuda.device_count() > 1:
                self.network = nn.DataParallel(self.network)

    def setup_path(self, flags):

        image_size = flags.image_size
        self.train_data = self.datafiles['train']
        if self.flags.validate:
            self.val_data = self.datafiles['val']
        else:
            self.val_data = []

        self.test_data = self.datafiles['test']

        train_dataset = Image_Reader(
            args=self.flags, filelist=self.train_data)
        val_dataset = Image_Reader(args=self.flags,
                                   filelist=self.val_data,
                                   is_train=False)
        test_dataset = Image_Reader(args=self.flags,
                                    filelist=self.test_data,
                                    is_train=False)

        self.datasets = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }

        if flags.phase == 'vizualize':
            flags.batch_size = 1

        self.train_len = train_dataset.__len__()
        self.test_len = test_dataset.__len__()
        self.val_len = val_dataset.__len__()

        print('longeur train {}'.format(train_dataset.__len__()))
        print('longeur val {}'.format(val_dataset.__len__()))
        print('longeur test {}'.format(test_dataset.__len__()))

        print('number of workers = {}'.format(mp.cpu_count()))

        image_datasets = {'train': train_dataset,
                          'val': val_dataset, 'test': test_dataset}

        self.dataset_sizes = {
            x: len(image_datasets[x])
            for x in ['train', 'val', 'test']
        }

        val_dataloader = None
        test_dataloader = None

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=flags.batch_size * self.num_devices,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            worker_init_fn=worker_init_fn)

        if self.dataset_sizes['val'] > 0:
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=flags.batch_size * self.num_devices,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                worker_init_fn=worker_init_fn)

        if self.dataset_sizes['test'] > 0:
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=flags.batch_size * self.num_devices,
                shuffle=(flags.phase != 'timeline_prediction'),
                num_workers=0,
                pin_memory=True,
                worker_init_fn=worker_init_fn)

        self.dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}


        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

    def freeze(self):
        """Freeze model except the last layer"""
        for name, param in self.network.named_parameters():
            # or 'layer3' in name
            if 'fc' in name or 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def configure(self, flags):

        if flags.load_checkpoint != -7:
            self.load_checkpoint(flags.load_checkpoint)

        for name, para in self.network.named_parameters():
            print(name, para.size())

        total_params = 0

        named_params_to_update = {}

        if self.current_lr is None:
            self.current_lr = flags.init_lr

        for name, param in self.network.named_parameters():
            total_params += 1
            if param.requires_grad:
                named_params_to_update[name] = param

        if flags.metric == 'add_margin':
            self.metric_fc = AddMarginProduct(512, flags.num_classes, s=30, m=0.35)
        elif flags.metric == 'arc_margin':
            self.metric_fc = ArcMarginProduct(512, flags.num_classes, s=30, m=0.5, easy_margin=flags.easy_margin)
        elif flags.metric == 'sphere':
            self.metric_fc = SphereProduct(512, flags.num_classes, m=4)
        else:
            self.metric_fc = nn.Linear(512, flags.num_classes)

        metrics_parameters_to_update = {}

        for name, param in self.metric_fc.named_parameters():
            total_params += 1
            if param.requires_grad:
                metrics_parameters_to_update[name] = param

        print("Params to learn:")
        if len(named_params_to_update) +  len(metrics_parameters_to_update) == total_params:
            print("\tfull network")
        else:
            for name in named_params_to_update:
                print("\t{}".format(name))

        parameters = [{'params': list(named_params_to_update.values())}, {'params': list(metrics_parameters_to_update.values())}]

        if flags.optimizer == "sgd":
            print(colored(flags.optimizer, 'red'))
            self.network_optimizer = optim.SGD(
                params=parameters,
                lr=self.init_lr,
                momentum=0.9,
                dampening=0,
                nesterov=False,
                weight_decay=1e-5
            )

        elif flags.optimizer == "adagrad":
            print(colored(flags.optimizer, 'red'))
            self.network_optimizer = optim.Adagrad(
                params=parameters,
                lr=self.init_lr,
                lr_decay=0,
                initial_accumulator_value=0.1,
                eps=1e-10,
                weight_decay=1e-5
            )

        elif flags.optimizer == "rmsprop":
            print(colored(flags.optimizer, 'red'))
            self.network_optimizer = optim.RMSprop(
                params=parameters,
                lr=self.init_lr,
                alpha=0.99,
                eps=1e-08,
                momentum=0,
                centered=False,
                weight_decay=1e-5
            )

        elif flags.optimizer == "adam":
            print(colored(flags.optimizer, 'red'))
            self.network_optimizer = optim.Adam(
                params=parameters,
                lr=self.init_lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=False,
                weight_decay=1e-5
            )

        self.device = next(self.network.parameters()).device

        self.loss_fn = torch.nn.CrossEntropyLoss()

        if self.flags.sheduler_type == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.network_optimizer, mode='min', patience=flags.patience)
        elif self.flags.sheduler_type == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.network_optimizer, step_size=self.flags.step_size)
        elif self.flags.sheduler_type == 'CyclicLR':
            self.scheduler = optim.lr_scheduler.CyclicLR(
                self.network_optimizer, base_lr=self.init_lr, max_lr=1e-2, step_size_up=self.flags.step_size,
                cycle_momentum=(flags.optimizer != 'Adam'))
        else:
            self.scheduler = None

        self.loss_function = torch.nn.CrossEntropyLoss()

    def adjust_learning_rate(self, flags, epoch=1, every_n=30):
        """Sets the learning rate to the initial LR decayed by 10 every n epoch epochs"""
        every_n_epoch = every_n  # n_epoch/n_step
        lr = flags.init_lr * (0.1 ** (epoch // flags.step_size))
        for param_group in self.network_optimizer.param_groups:
            param_group['lr'] = lr

        self.current_lr = lr

    def checkpointing(self, save_curr=True):
        # if self.verbose > 0:
        # print(' ')
        #    print('Checkpointing...')

        ckpt = {
            'model': self.network.state_dict(),
            'history': self.history,
            'cur_epoch': self.cur_epoch,
            'learning_rate': self.current_lr,
            'optimizer': self.optimizer
        }
        if save_curr:
            torch.save(ckpt, self.save_epoch_fmt_task.format(self.cur_epoch))

        if self.cur_epoch == self.best_epoch:
            torch.save(ckpt, os.path.join(self.checkpoint_path,
                                          'best_model.pt'))

    def load_checkpoint(self, epoch):

        def load_keys(ckpt):
            if os.path.isfile(ckpt):
                ckpt = torch.load(ckpt)
                # Load model state
                self.network.load_state_dict(ckpt['model'])
                # Load history
                self.history = ckpt['history']
                self.cur_epoch = ckpt['cur_epoch']
                self.current_lr = ckpt['learning_rate']
                self.l1_lambda = ckpt['learning_rate']
                #self.optimizer = ckpt['optimizer']
                return True

            return False

        self.current_lr = None
        ckpt = self.save_epoch_fmt_task.format(epoch)

        if load_keys(ckpt):
            print('Checkpoint number {} loaded  ---> {}'.format(epoch, ckpt))
        else:

            ckpt_best = os.path.join(self.checkpoint_path, 'best_model.pt')

            if load_keys(ckpt_best):
                print('Checkpoint number {} loaded  ---> {}'.format(self.cur_epoch, ckpt_best))
            else:
                print(colored('No checkpoint found at: {}'.format(ckpt), 'red'))
                if self.flags.phase != 'train':
                    raise ValueError('----------Unable to load checkpoint  {}. The program will exit now----------\n\n'.format(ckpt))

            return False

    def printing_train(self, tot_loss, current_lr):
        aff = 'Train results : ite:{}, Loss:{:.4f}, current_lr: {}'.format(self.cur_epoch, tot_loss, current_lr)
        return aff

    def forward_pass(self, imgs, model, batch_size):
        imgs = imgs.cuda()
        embeddings = model(imgs)

        # Split the embeddings into Anchor, Positive, and Negative embeddings
        anc_embeddings = embeddings[:batch_size]
        pos_embeddings = embeddings[batch_size: batch_size * 2]
        neg_embeddings = embeddings[batch_size * 2:]

        return anc_embeddings, pos_embeddings, neg_embeddings, model

    def train_am(self, flags):

        loader = self.dataloaders['train']

        iters = 0

        start = time()

        while self.cur_epoch < self.flags.num_epochs:

            self.network.train()
            self.metric_fc.train()
            data_iter = tqdm(enumerate(loader), disable=False)
            total_loss = 0.0


            for step, data in data_iter:
                inputs, labels = data

                if (self.cur_epoch == 0) and (step == 0) and (self.writer is not None):
                    grid = torchvision.utils.make_grid(inputs)
                    self.writer.add_image("images", grid)

                inputs = Variable(inputs.cuda())

                labels = Variable(labels.cuda())

                self.network_optimizer.zero_grad()
                print(inputs.shape)
                embeddings = self.network(inputs)
                print(embeddings.shape)
                outputs = self.metric_fc(embeddings, labels)
                loss = self.loss_fn(outputs, labels)

                self.network_optimizer.zero_grad()
                loss.backward()
                self.network_optimizer.step()

                iters += 1


                if iters % flags.print_freq == 0:

                    outputs = outputs.data.cpu().numpy()
                    outputs = np.argmax(outputs, axis=1)

                    labels = labels.data.cpu().numpy()
                    # print(output)
                    # print(label)
                    acc = np.mean((outputs == labels).astype(int))
                    speed = flags.print_freq / (time.time() - start)
                    time_str = time.asctime(time.localtime(time.time()))
                    print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, self.cur_epoch, iters, speed,
                                                                                       loss.item(), acc))


                    start = time.time()

            self.cur_epoch += 1




    def bn_process(self, flags):
        if flags.bn_eval:
            self.network.bn_eval()
