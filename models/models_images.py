import os

from numpy.lib.npyio import save

# dirpath = os.pardir
import sys

# sys.path.append(dirpath)
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.optim import lr_scheduler, optimizer

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
from models.plotting import plot
from backbones.margins import *
from .test_utils import *
from .tsne_utils import *
import copy
from itertools import combinations


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
        self.embedding_dimension = flags.embedding_dimension
        self.batch_size = flags.batch_size
        self.num_workers = flags.num_workers
        self.opencv_net = None
        if flags.phase == 'train':
            self.writer = SummaryWriter(
                log_dir=os.path.join('runs', self.flags.all_parameters + '_' + str(datetime.now())))
        else:
            self.writer = None

    def setup(self, flags):
        torch.backends.cudnn.deterministic = flags.deterministic
        print('torch.backends.cudnn.deterministic:',
              torch.backends.cudnn.deterministic)
        fix_all_seed(flags.launch_seed)

        if flags.freeze:
            self.freeze()

        if flags.metric == 'add_margin':
            self.metric_fc = AddMarginProduct(512, flags.num_classes, s=30, m=0.35)
        elif flags.metric == 'arc_margin':
            self.metric_fc = ArcMarginProduct(512, flags.num_classes, s=30, m=0.5, easy_margin=flags.easy_margin)
        elif flags.metric == 'sphere':
            self.metric_fc = SphereProduct(512, flags.num_classes, m=4)
        else:
            self.metric_fc = nn.Linear(512, flags.num_classes)

        self.network = self.network.cuda()
        self.metric_fc = self.metric_fc.cuda()
        if flags.data_parallel:
            if cuda.device_count() > 1:
                self.network = nn.DataParallel(self.network)
                self.metric_fc = nn.DataParallel(self.metric_fc)

    def setup_path(self, flags):

        image_size = flags.image_size
        self.train_data = self.datafiles['train']

        train_dataset = Image_Reader(
            args=self.flags, filelist=self.train_data)

        self.datasets = {
            'train': train_dataset
        }

        if flags.phase == 'vizualize':
            flags.batch_size = 1

        self.train_len = train_dataset.__len__()

        print('longeur train {}'.format(train_dataset.__len__()))

        print('number of workers = {}'.format(mp.cpu_count()))

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=flags.batch_size * self.num_devices,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            worker_init_fn=worker_init_fn)

        self.dataloaders = {'train': train_dataloader}

        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

    def freeze(self):
        """Freeze model except the last layer"""
        for name, param in self.network.named_parameters():
            if 'fc' in name or 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def configure(self, flags):

        if flags.load_checkpoint != -7:
            self.load_checkpoint(flags.load_checkpoint)

        # for name, para in self.network.named_parameters():
        #     print(name, para.size())

        total_params = 0

        if self.current_lr is None:
            self.current_lr = flags.init_lr

        named_params_to_update = {}
        for name, param in self.network.named_parameters():
            total_params += 1
            if param.requires_grad:
                named_params_to_update[name] = param

        metrics_parameters_to_update = {}

        for name, param in self.metric_fc.named_parameters():
            total_params += 1
            if param.requires_grad:
                metrics_parameters_to_update[name] = param

        if flags.phase == 'train':
            print("Params to learn:")
            if len(named_params_to_update) + len(metrics_parameters_to_update) == total_params:
                print("\tfull network")
            else:
                for name in named_params_to_update:
                    print("\t{}".format(name))

        parameters = [{'params': list(named_params_to_update.values())},
                      {'params': list(metrics_parameters_to_update.values())}]

        if flags.optimizer == "sgd":
            self.network_optimizer = optim.SGD(
                params=parameters,
                lr=self.init_lr,
                momentum=0.9,
                dampening=0,
                nesterov=False,
                weight_decay=1e-5
            )

        elif flags.optimizer == "adagrad":
            self.network_optimizer = optim.Adagrad(
                params=parameters,
                lr=self.init_lr,
                lr_decay=0,
                initial_accumulator_value=0.1,
                eps=1e-10,
                weight_decay=1e-5
            )

        elif flags.optimizer == "rmsprop":
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

        self.scheduler = optim.lr_scheduler.CyclicLR(
            self.network_optimizer, base_lr=self.init_lr, max_lr=1e-2, step_size_up=self.flags.step_size,
            cycle_momentum=(flags.optimizer != 'adam' and flags.optimizer != 'adagrad'))

        self.loss_function = torch.nn.CrossEntropyLoss()

    def adjust_learning_rate(self, flags, epoch=1, every_n=30):
        """Sets the learning rate to the initial LR decayed by 10 every n epoch epochs"""
        every_n_epoch = every_n  # n_epoch/n_step
        lr = flags.init_lr * (0.1 ** (epoch // flags.step_size))
        for param_group in self.network_optimizer.param_groups:
            param_group['lr'] = lr

        self.current_lr = lr

    def checkpointing(self):
        # if self.verbose > 0:
        # print(' ')
        #    print('Checkpointing...')

        ckpt = {
            'model': self.network.state_dict(),
            'metric': self.metric_fc.state_dict(),
            'history': self.history,
            'cur_epoch': self.cur_epoch,
            'learning_rate': self.current_lr,
            'optimizer': self.optimizer
        }

        torch.save(ckpt, self.save_epoch_fmt_task.format(self.cur_epoch))

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
                # self.optimizer = ckpt['optimizer']
                return True

            return False

        self.current_lr = None
        ckpt = self.save_epoch_fmt_task.format(epoch)

        if load_keys(ckpt):
            print('Checkpoint number {} loaded  ---> {}'.format(epoch, ckpt))
        else:
            print(colored('No checkpoint found at: {}'.format(ckpt), 'red'))
            if self.flags.phase != 'train':
                raise ValueError(
                    '----------Unable to load checkpoint  {}. The program will exit now----------\n\n'.format(ckpt))

        return False

    def printing_train(self, tot_loss, current_lr):
        aff = 'Train results : ite:{}, Loss:{:.4f}, current_lr: {}'.format(self.cur_epoch, tot_loss, current_lr)
        return aff

    def train_am(self, flags):

        loader = self.dataloaders['train']
        iters = 0
        start = time.time()

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
                # print(inputs.shape)

                self.network_optimizer.zero_grad()
                # print(inputs.shape)
                embeddings = self.network(inputs)
                # print(embeddings.shape)
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
                    # print(outputs)
                    # print(labels)
                    acc = np.mean((outputs == labels).astype(int))
                    speed = flags.print_freq / (time.time() - start)
                    time_str = time.asctime(time.localtime(time.time()))
                    print('Train epoch {} iter {} {} iters/s loss {:.4f} acc {}'.format(self.cur_epoch, iters, speed,
                                                                                        loss.item(), acc))

            acc = self.validation()
            # print(colored('Testing accuracy {}'.format(acc), 'blue'))
            start = time.time()

            self.scheduler.step()
            self.current_lr = self.scheduler.get_last_lr()

            self.cur_epoch += 1

            if (self.cur_epoch +
                1) % self.save_every == 0 and self.cur_epoch != 0:
                self.checkpointing()

    def test(self, flags):
        self.load_checkpoint(self.flags.load_checkpoint)
        self.cur_epoch = self.flags.load_checkpoint
        self.network.eval()
        self.validation()

    def convert_to_onnx(self, flags):
        self.load_checkpoint(self.flags.load_checkpoint)
        onnx_model_name = flags.backbone + ".onnx"
        # create directory for further converted model
        # get full path to the converted model
        generated_input = Variable(
            torch.randn(1, 1, 128, 128)
        )
        full_model_path = os.path.join(self.checkpoint_path, onnx_model_name)
        torch.onnx.export(
            self.network,
            generated_input,
            full_model_path,
            verbose=False,
            input_names=["input"],
            output_names=["output"],
            opset_version=11
        )
        print('Model coonvert to onnx')
        self.opencv_net = cv2.dnn.readNetFromONNX(full_model_path)

    def evaluate_pt_vs_cv(self, flags):
        if self.opencv_net is None:
            self.convert_to_onnx(flags)

        print(colored('With pytorch', 'red'))
        self.validation(onnx=False)
        print(colored('With opencv', 'red'))
        self.validation(onnx=False)

        return

    def demo(self, flags):

        link_video = flags.video
        list_imgs = flags.list_imgs
        img_paths = [os.path.join('demo_faces', each) for each in list_imgs]


        searches_faces = []
        for img_path in img_paths:
            img = cv2.imread(img_path, 0)
            if img:
                img = cv2.resize(img, dsize=(128,128))
                img = np.expand_dims(img, axis=2)
                searches_faces.append(img)

        # to find path of xml file containing haarCascade file
        cfp = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
        # load the harcaascade in the cascade classifier
        face_cascade = cv2.CascadeClassifier(cfp)

        cap = cv2.VideoCapture(os.path.join('demo_videos', link_video))

        # Read until video is completed
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()

            faces = face_cascade.detectMultiScale(frame)

            list_faces_images = []
            ref_values = []

            for i, (x, y, w, h) in enumerate(faces):
                img = copy.copy(frame[y:y + h, x:x + w])
                img = cv2.resize(img, dsize=(128, 128))
                img = rgb2gray(img)
                img = np.expand_dims(img, axis=2)
                list_faces_images.append(img)
                ref_values.append(
                    (i, (x, y, w, h))
                )
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            #video_features = self.get_features_imgs(list_faces_images, onnx=True)
            #searches_features = self.get_features_imgs()


            if ret == True:

                # Display the resulting frame
                cv2.imshow('Demo', frame)

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            # Break the loop
            else:
                break

        # When everything done, release the video capture object
        cap.release()

        # Closes all the frames
        cv2.destroyAllWindows()

        return

    def validation(self, onnx=False):
        flags = self.flags
        s = time.time()
        list_labels = []
        identity_list = get_lfw_list(self.datafiles['val'][0])
        img_paths = [os.path.join(each) for each in identity_list]
        for each in identity_list:
            list_labels.append(
                self.datafiles['val'][1][each]
            )

        features, cnt = self.get_features(img_paths, onnx)

        t = time.time() - s

        print('total time is {}, average time is {}'.format(t, t / cnt))
        fe_dict = get_feature_dict(identity_list, features, )
        acc, th = test_performance(fe_dict, self.datafiles['val'][0])
        plot(features, np.asarray(list_labels),
             fig_path='./figs/{}_{}_{}.png'.format(flags.loss_type, self.cur_epoch, onnx))
        print('lfw face verification accuracy: ', acc, 'threshold: ', th)

        # plot_with_pca(self.checkpoint_path, features, img_paths, list_labels)

        return acc

    def get_features_imgs(self, list_imgs, onnx=False):

        self.network.eval()

        batch_size = 1

        images = None
        features = []
        cnt = 0
        for i, image in enumerate(list_imgs):

            if images is None:
                images = image
            else:
                images = np.concatenate((images, image), axis=0)

            if images.shape[0] % batch_size == 0 or i == len(list_imgs) - 1:
                cnt += 1
                if not onnx:
                    data = torch.from_numpy(images)
                    data = data.cuda()
                    # print(data.shape)
                    output = self.network(data)
                    output = output.data.cpu().numpy()
                else:
                    input_blob = cv2.dnn.blobFromImage(
                        image=images,
                        scalefactor=1
                    )
                    output = self.opencv_net(input_blob)

                fe_1 = output[::2]
                fe_2 = output[1::2]
                feature = np.hstack((fe_1, fe_2))
                # print(feature.shape)

                features.append(feature)

                images = None

        return features, cnt

    def get_features(self, test_list, onnx=False):

        self.network.eval()

        batch_size = 1

        images = None
        features = None
        cnt = 0
        for i, img_path in enumerate(test_list):
            image = load_image(img_path)
            if image is None:
                print('read {} error'.format(img_path))

            if images is None:
                images = image
            else:
                images = np.concatenate((images, image), axis=0)

            if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
                cnt += 1
                if not onnx:
                    data = torch.from_numpy(images)
                    data = data.cuda()
                    # print(data.shape)
                    output = self.network(data)
                    output = output.data.cpu().numpy()
                else:
                    input_blob = cv2.dnn.blobFromImage(
                        image=images,
                        scalefactor=1
                    )
                    output = self.opencv_net(input_blob)

                fe_1 = output[::2]
                fe_2 = output[1::2]
                feature = np.hstack((fe_1, fe_2))
                # print(feature.shape)

                if features is None:
                    features = feature
                else:
                    features = np.vstack((features, feature))

                images = None

        return features, cnt

    def bn_process(self, flags):
        if flags.bn_eval:
            self.network.bn_eval()
