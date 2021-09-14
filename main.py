from models.models_images import ImageModel
from data_factory.datacreator import get_data
from backbones.load_models import get_backbone_model

import glob
import random
import os
import sys
import argparse
import setproctitle
import json

import torch

from termcolor import colored
from viz_window import *

import torchvision
from time import time
from datetime import datetime


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def string_to_int_list(s):
    if s is None or len(s) == 0:
        raise ValueError('Empty string')
    s = s.split(',')
    results = []
    for value in s:
        value = int(value)
        results.append(value)
    return results

def string_to_string_list(s):
    if s is None or len(s) == 0:
        raise ValueError('Empty string')
    s = s.split(',')
    results = []
    for value in s:
        results.append(value)
    return results

parser = argparse.ArgumentParser()

parser.add_argument('-mode', type=str, default='train_model', help='train_model, simulate_data, display_data')
parser.add_argument('-backbone', type=str, default='resnet18', help='the model')
parser.add_argument('-batch_size', type=int, default=16)
parser.add_argument('-seed', type=string_to_int_list, default='0')
parser.add_argument('-dataset', type=str, default='lfw_cropped')
parser.add_argument('-phase', type=str, default='demo', help='train, extract_features, vizualize, timeline_prediction')
parser.add_argument('-cuda', type=boolean_string, default=True)
parser.add_argument('-early_stopping', type=boolean_string, default=True)
parser.add_argument('-nb_max_step_without_improvements', type=int, default=5)
parser.add_argument('-data_parallel', type=boolean_string, default=False)
parser.add_argument('-freeze', type=boolean_string, default=False)
parser.add_argument('-checkpoint_path', type=str, default='checkpoint')
parser.add_argument("-init_lr", type=float, default=0.0001, help='')
parser.add_argument("-dropout", type=float, default=0, help='')
parser.add_argument("-l1_lambda", type=float, default=0, help='')
parser.add_argument("-grad_steps", type=float, default=2, help='')
parser.add_argument("-weight_decay", type=float, default=0.00005, help='0.00005')
parser.add_argument("-patience", type=int, default=10, help='')
parser.add_argument("-lr_decay_factor", type=float, default=0.001, help='')
parser.add_argument("-lr_decay_steps", type=float, default=None, help='')
parser.add_argument("-momentum", type=float, default=0.95, help='')
parser.add_argument("-warmup_pct", type=float, default=0.3, help='')
parser.add_argument("-deterministic", default=True, type=boolean_string, help='')
parser.add_argument("-test_every", type=int, default=1, help="")
parser.add_argument("-print_freq", type=int, default=100, help="")
parser.add_argument("-save_every", type=int, default=1, help="")
parser.add_argument("-step_size", type=int, default=7, help="")
parser.add_argument("-num_epochs", type=int, default=100, help="")
parser.add_argument("-logs", type=str, default='logs/', help='')
parser.add_argument('-verbose', type=boolean_string, default=True)
parser.add_argument('-use_se', type=boolean_string, default=False)
parser.add_argument('-augment', type=boolean_string, default=False)
parser.add_argument('-sheduler_type', type=str, default='CyclicLR', help='StepLR, CyclicLR, ReduceLROnPlateau,None')
parser.add_argument('-load_checkpoint', type=int, default=20, metavar='N',
                    help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('-optimizer', type=str, default="adagrad", choices=["sgd", "adagrad", "rmsprop", "adam"],
                    help="Required optimizer for training the model: ('sgd','adagrad','rmsprop','adam'), (default: 'adagrad')")

parser.add_argument('-dataroot', '-d', type=str, default='Datasets',
                    help="Absolute path to the training dataset folder")
parser.add_argument('-video', '-v', type=str, default='the_office.mp4',
                    help="")
parser.add_argument('-list_imgs', type=string_to_string_list, default='mickael_scott.jpg,dwight.jpg')
parser.add_argument('-pretrained', default=False, type=bool,
                    help="Download a model pretrained on the ImageNet dataset (Default: False)")
parser.add_argument('-embedding_dimension', default=512, type=int,
                    help="Dimension of the embedding vector (default: 512)")
parser.add_argument('-num_workers', default=0, type=int, help="Number of workers for data loaders (default: 4)")
parser.add_argument('-loss_type', type=str, default='cosface', help='cosface, sphereface, arcface')
parser.add_argument('-writer_dir', type=str, default='runs', help='Where to save metrics')
parser.add_argument('-gpus', type=str, default='0')
parser.add_argument('-metric', type=str, default="add_margin", choices=["add_margin", "arc_margin", "sphere"])


flags = parser.parse_args()
argparse_dict = vars(flags)


def system_info():
    import torch.cuda as cuda
    print(sys.version, "\n")
    print("PyTorch {}".format(torch.__version__), "\n")
    print("Torch-vision {}".format(torchvision.__version__), "\n")
    print("Available devices:")
    if cuda.is_available():
        for i in range(cuda.device_count()):
            print("{}: {}".format(i, cuda.get_device_name(i)))
    else:
        print("CPUs")




def running(run):
    if flags.seed is None:
        launch_seed = 10 * run
    else:
        launch_seed = flags.seed[run]

    flags.launch_seed = launch_seed

    torch.backends.cudnn.benchmark = False

    train_mode = flags.backbone + '_' + flags.dataset + '_' + flags.loss_type + '_' + flags.optimizer

    if flags.phase == 'train':
        flags.nb_max_step_without_improvements = 15

    flags.launch_mode = train_mode
    flags.all_parameters = train_mode + '_seed' + str(launch_seed)

    flags.image_size = 128

    setproctitle.setproctitle(flags.backbone)

    random.seed(launch_seed)
    torch.manual_seed(launch_seed)
    if flags.cuda:
        torch.cuda.manual_seed(launch_seed)
    checkpoint_path = os.path.join(flags.checkpoint_path, train_mode, '_seed' + str(launch_seed))

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    flags.model_path = checkpoint_path

    if flags.phase == 'train':
        if not os.path.exists(os.path.join(checkpoint_path, 'parameters.json')):
            with open(os.path.join(checkpoint_path, 'parameters.json'), 'w') as fp:
                json.dump(argparse_dict, fp, indent=4)
        else:
            with open(os.path.join(checkpoint_path, 'parameters.json')) as f:
                saved_dict = json.load(f)
            assert (argparse_dict['all_parameters'] == saved_dict['all_parameters'])

    print(colored('Loading data', 'blue'))

    # get_lfw_list('lfw_test_pair.txt')

    datafiles, taille, nb_classes_train, nb_classes_val = get_data(flags)
    flags.num_classes = nb_classes_train
    backbone = get_backbone_model(flags)

    if flags.phase == 'train':
        print(colored('\n\nModel to train {}'.format(train_mode), 'blue'))
        print(colored('Where to save model {} \n'.format(checkpoint_path), 'blue'))
    elif flags.phase != 'train':
        print(colored('\n\nTrained model {}'.format(checkpoint_path), 'blue'))

    print('training for dataset  ' + flags.dataset)

    model_obj = ImageModel(flags=flags,
                           backbone=backbone,
                           datafiles=datafiles,
                           checkpoint_path=checkpoint_path,
                           train_mode=train_mode,
                           nb_classes_train=nb_classes_train,
                           nb_classes_val=nb_classes_val)

    if flags.phase == 'train':
        model_obj.train_am(flags=flags)
    elif flags.phase == 'test':
        model_obj.test(flags=flags)
    elif flags.phase == 'convert_to_onnx':
        model_obj.convert_to_onnx(flags)
    elif flags.phase == 'evaluate_pt_vs_cv':
        model_obj.evaluate_pt_vs_cv(flags)
    elif flags.phase == 'demo':
        model_obj.demo(flags)

    flags.model_path = None


def runs():
    for i in range(len(flags.seed)):
        print('Run {}'.format(i))
        running(i)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpus
    system_info()
    runs()
