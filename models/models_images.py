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
from losses.triplet_loss import TripletLoss
from data_factory.TripletLossDataset import TripletFaceDataset
from models.plotting import plot_roc_lfw, plot_accuracy_lfw
from backbones.margins import *


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
                 datafiles=None,
                 checkpoint_path=None,
                 train_mode='',
                 data_weitghs=None):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.init_lr = flags.init_lr
        self.network = backbone
        self.datafiles = datafiles
        self.verbose = flags.verbose
        self.checkpoint_path = checkpoint_path
        self.save_every = flags.save_every
        self.flags = flags
        self.launch_mode = train_mode
        self.l1_lambda = flags.l1_lambda
        self.num_devices = cuda.device_count()
        self.train_data_weitghs, _ = data_weitghs if data_weitghs is not None else (None, None)
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
        use_semihard_negatives = flags.use_semihard_negatives
        training_triplets_path = flags.training_triplets_path
        flag_training_triplets_path = False


        if training_triplets_path is not None:
            flag_training_triplets_path = True  # Load triplets file for the first training epoch


        self.data_transforms = transforms.Compose([
            transforms.Resize(size=image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.6071, 0.4609, 0.3944],
                std=[0.2457, 0.2175, 0.2129]
            )
        ])

        lfw_transforms = transforms.Compose([
            transforms.Resize(size=image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.6071, 0.4609, 0.3944],
                std=[0.2457, 0.2175, 0.2129]
            )
        ])


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

        print("Params to learn:")
        if len(named_params_to_update) == total_params:
            print("\tfull network")
        else:
            for name in named_params_to_update:
                print("\t{}".format(name))


        if flags.optimizer == "sgd":
            print(colored(flags.optimizer, 'red'))
            self.network_optimizer = optim.SGD(
                params=list(named_params_to_update.values()),
                lr=self.init_lr,
                momentum=0.9,
                dampening=0,
                nesterov=False,
                weight_decay=1e-5
            )

        elif flags.optimizer == "adagrad":
            print(colored(flags.optimizer, 'red'))
            self.network_optimizer = optim.Adagrad(
                params=list(named_params_to_update.values()),
                lr=self.init_lr,
                lr_decay=0,
                initial_accumulator_value=0.1,
                eps=1e-10,
                weight_decay=1e-5
            )

        elif flags.optimizer == "rmsprop":
            print(colored(flags.optimizer, 'red'))
            self.network_optimizer = optim.RMSprop(
                params=list(named_params_to_update.values()),
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
                params=list(named_params_to_update.values()),
                lr=self.init_lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=False,
                weight_decay=1e-5
            )

        if flags.metric is not None:
            if flags.metric == 'add_margin':
                metric_fc = AddMarginProduct(512, flags.num_classes, s=30, m=0.35)
            elif flags.metric == 'arc_margin':
                metric_fc = ArcMarginProduct(512, flags.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
            elif flags.metric == 'sphere':
                metric_fc = SphereProduct(512, flags.num_classes, m=4)
            else:
                metric_fc = nn.Linear(512, flags.num_classes)

        self.device = next(self.network.parameters()).device


        if self.flags.sheduler_type == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=flags.patience)
        elif self.flags.sheduler_type == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.flags.step_size)
        elif self.flags.sheduler_type == 'CyclicLR':
            self.scheduler = optim.lr_scheduler.CyclicLR(
                self.optimizer, base_lr=self.init_lr, max_lr=1e-2, step_size_up=self.flags.step_size,
                cycle_momentum=(flags.optimizer != 'Adam'))
        else:
            self.scheduler = None

        self.loss_function = torch.nn.CrossEntropyLoss()

    def adjust_learning_rate(self, flags, epoch=1, every_n=30):
        """Sets the learning rate to the initial LR decayed by 10 every n epoch epochs"""
        every_n_epoch = every_n  # n_epoch/n_step
        lr = flags.init_lr * (0.1 ** (epoch // flags.step_size))
        for param_group in self.optimizer.param_groups:
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

    def train_am(self, train_loader, loss_type):
        total_step = len(train_loader)
        for epoch in tqdm(range(self.flags.num_epochs)): 
            for i, (feats, labels) in enumerate(tqdm(train_loader)):
                #print(labels)
                feats = feats.cuda()
                labels = labels.cuda()
                loss = self.network(feats, labels=labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i+1) % 100 == 0:
                    print('{}: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(loss_type, epoch+1, args.num_epochs, i+1, total_step, loss.item()))

            if((epoch+1) % 8 == 0):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr']/4
            
        return model.cpu()

    def train_triplet(self, flags):

        flag_training_triplets_path = False

        if flags.use_semihard_negatives:
            print("Using Semi-Hard negative triplet selection!")
        else:
            print("Using Hard negative triplet selection!")

        start_epoch = 0

        print("Training using triplet loss starting for {} epochs:\n".format(flags.num_epochs - start_epoch))

        for epoch in range(start_epoch, flags.num_epochs):
            num_valid_training_triplets = 0
            l2_distance = PairwiseDistance(p=2)
            _training_triplets_path = None

            if flag_training_triplets_path:
                _training_triplets_path = flags.training_triplets_path
                flag_training_triplets_path = False  # Only load triplets file for the first epoch

            # Re-instantiate training dataloader to generate a triplet list for this training epoch
            train_dataloader = torch.utils.data.DataLoader(
                dataset=TripletFaceDataset(
                    root_dir=flags.dataroot,
                    training_dataset_csv_path=flags.training_dataset_csv_path,
                    num_triplets=flags.iterations_per_epoch * flags.batch_size,
                    num_human_identities_per_batch=flags.num_human_identities_per_batch,
                    triplet_batch_size=flags.batch_size,
                    epoch=epoch,
                    training_triplets_path=_training_triplets_path,
                    transform=self.data_transforms
                ),
                batch_size=flags.batch_size,
                num_workers=flags.num_workers,
                shuffle=False  # Shuffling for triplets with set amount of human identities per batch is not required
            )

            # Training pass
            self.network.train()
            progress_bar = enumerate(tqdm(train_dataloader))

            total_loss = 0.0

            for batch_idx, (batch_sample) in progress_bar:

                # Forward pass - compute embeddings
                anc_imgs = batch_sample['anc_img']
                pos_imgs = batch_sample['pos_img']
                neg_imgs = batch_sample['neg_img']

                # Concatenate the input images into one tensor because doing multiple forward passes would create
                #  weird GPU memory allocation behaviours later on during training which would cause GPU Out of Memory
                #  issues
                all_imgs = torch.cat((anc_imgs, pos_imgs, neg_imgs))  # Must be a tuple of Torch Tensors

                anc_embeddings, pos_embeddings, neg_embeddings, self.network = self.forward_pass(
                    imgs=all_imgs,
                    model=self.network,
                    batch_size=flags.batch_size
                )

                pos_dists = l2_distance.forward(anc_embeddings, pos_embeddings)
                neg_dists = l2_distance.forward(anc_embeddings, neg_embeddings)

                if flags.use_semihard_negatives:
                    # Semi-Hard Negative triplet selection
                    #  (negative_distance - positive_distance < margin) AND (positive_distance < negative_distance)
                    #   Based on: https://github.com/davidsandberg/facenet/blob/master/src/train_tripletloss.py#L295
                    first_condition = (neg_dists - pos_dists < flags.margin).cpu().numpy().flatten()
                    second_condition = (pos_dists < neg_dists).cpu().numpy().flatten()
                    all = (np.logical_and(first_condition, second_condition))
                    valid_triplets = np.where(all == 1)
                else:
                    # Hard Negative triplet selection
                    #  (negative_distance - positive_distance < margin)
                    #   Based on: https://github.com/davidsandberg/facenet/blob/master/src/train_tripletloss.py#L296
                    all = (neg_dists - pos_dists < flags.margin).cpu().numpy().flatten()
                    valid_triplets = np.where(all == 1)

                anc_valid_embeddings = anc_embeddings[valid_triplets]
                pos_valid_embeddings = pos_embeddings[valid_triplets]
                neg_valid_embeddings = neg_embeddings[valid_triplets]

                

                # Calculate triplet loss
                triplet_loss = TripletLoss(margin=flags.margin).forward(
                    anchor=anc_valid_embeddings,
                    positive=pos_valid_embeddings,
                    negative=neg_valid_embeddings
                )

                total_loss = triplet_loss.cpu().item()

                # Calculating number of triplets that met the triplet selection method during the epoch
                num_valid_training_triplets += len(anc_valid_embeddings)

                # Backward pass
                self.network_optimizer.zero_grad()
                triplet_loss.backward()
                self.network_optimizer.step()

            # Print training statistics for epoch and add to log
            print('Epoch {}:\t loss{} and Number of valid training triplets in epoch: {}'.format(
                    epoch,
                    total_loss,
                    num_valid_training_triplets
                )
            )

            with open('logs/{}_log_triplet.txt'.format(flags.backbone), 'a') as f:
                val_list = [
                    epoch,
                    num_valid_training_triplets
                ]
                log = '\t'.join(str(value) for value in val_list)
                f.writelines(log + '\n')

            # Evaluation pass on LFW dataset
            # best_distances = validate_lfw(
            #     model=model,
            #     lfw_dataloader=lfw_dataloader,
            #     model_architecture=model_architecture,
            #     epoch=epoch
            # )



    def bn_process(self, flags):
        if flags.bn_eval:
            self.network.bn_eval()
