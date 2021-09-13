from models.models_images import ImageModel
from data_factory.load_data import get_data_classical_loss
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


parser = argparse.ArgumentParser()

parser.add_argument('-mode',type=str, default='train_model', help='train_model, simulate_data, display_data')
parser.add_argument('-backbone',type=str, default='resnet18', help='ddss_alexnet, ddss_resnet18, ddss_vit')
parser.add_argument('-batch_size', type=int, default=4)
parser.add_argument('-seed', type=string_to_int_list, default='0')
parser.add_argument('-dataset', type=str, default='lfw')
parser.add_argument('-phase', type=str,  default='train',  help='train, extract_features, vizualize, timeline_prediction')
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
parser.add_argument("-weight_decay",  type=float,       default=0.00005,    help='0.00005')
parser.add_argument("-patience", type=int, default=10, help='')
parser.add_argument("-lr_decay_factor", type=float, default=0.001, help='')
parser.add_argument("-lr_decay_steps", type=float, default=None, help='')
parser.add_argument("-momentum", type=float, default=0.95, help='')
parser.add_argument("-warmup_pct", type=float, default=0.3, help='')
parser.add_argument("-deterministic",  default=True,   type=boolean_string,  help='')
parser.add_argument("-test_every", type=int, default=1, help="")
parser.add_argument("-save_every", type=int, default=1, help="")
parser.add_argument("-step_size", type=int, default=7, help="")
parser.add_argument("-num_epochs", type=int, default=100, help="")
parser.add_argument("-logs", type=str, default='logs/', help='')
parser.add_argument('-verbose', type=boolean_string, default=True)
parser.add_argument('-balance_sampler', type=boolean_string, default=False)
parser.add_argument('-alpha_sampler', type=float, default=1.0)
parser.add_argument('-continue_training', type=boolean_string, default=False)
parser.add_argument('-only_load_data', type=boolean_string, default=False)
parser.add_argument('-augment', type=boolean_string, default=False)
parser.add_argument('-log_spectrogram', type=boolean_string, default=True)
parser.add_argument('-validate', type=boolean_string, default=True)
parser.add_argument("-sheduler_type", type=str, default=None, help='StepLR, CyclicLR, ReduceLROnPlateau,None')
parser.add_argument( '-load_checkpoint', type=int, default=19,  metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')


parser.add_argument('-dataroot', '-d', type=str, required=True, help="(REQUIRED) Absolute path to the training dataset folder")
#parser.add_argument('-lfw', type=str, required=True, help="(REQUIRED) Absolute path to the labeled faces in the wild dataset folder")
parser.add_argument('-training_dataset_csv_path', type=str, default='Datasets/lfw_224.csv',help="Path to the csv file containing the image paths of the training dataset")
parser.add_argument('-iterations_per_epoch', default=5000, type=int, help="Number of training iterations per epoch (default: 5000)"  )
parser.add_argument('-model_architecture', type=str, default="resnet34", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "inceptionresnetv2", "mobilenetv2"],  help="The required model architecture for training: ('resnet18','resnet34', 'resnet50', 'resnet101', 'resnet152', 'inceptionresnetv2', 'mobilenetv2'), (default: 'resnet34')")
parser.add_argument('-pretrained', default=False, type=bool, help="Download a model pretrained on the ImageNet dataset (Default: False)"  )
parser.add_argument('-embedding_dimension', default=512, type=int, help="Dimension of the embedding vector (default: 512)" )
parser.add_argument('-num_human_identities_per_batch', default=32, type=int,  help="Number of set human identities per generated triplets batch. (Default: 32)." )
parser.add_argument('-lfw_batch_size', default=200, type=int, help="Batch size for LFW dataset (6000 pairs) (default: 200)" )
parser.add_argument('-resume_path', default='',  type=str, help='path to latest model checkpoint: (model_training_checkpoints/model_resnet34_epoch_1.pt file) (default: None)' )
parser.add_argument('-num_workers', default=0, type=int, help="Number of workers for data loaders (default: 4)"  )
parser.add_argument('-optimizer', type=str, default="adagrad", choices=["sgd", "adagrad", "rmsprop", "adam"], help="Required optimizer for training the model: ('sgd','adagrad','rmsprop','adam'), (default: 'adagrad')" )
parser.add_argument('-learning_rate', default=0.075, type=float, help="Learning rate for the optimizer (default: 0.075)" )
parser.add_argument('-margin', default=0.2, type=float, help='margin for triplet loss (default: 0.2)'  )
parser.add_argument('-image_size', default=140, type=int,  help='Input image size (default: 140 (140x140))'  )
parser.add_argument('-use_semihard_negatives', default=False, type=bool, help="If True: use semihard negative triplet selection. Else: use hard negative triplet selection (Default: False)" )
parser.add_argument('-training_triplets_path', default=None, type=str, help="Path to training triplets numpy file in 'datasets/generated_triplets' folder to skip training triplet generation step for the first epoch." )

# data generation

parser.add_argument('-loss_type', type=str, default='cosface', help='cosface, sphereface, arcface')



parser.add_argument('-writer_dir',type=str, default='runs',  help='Where to save metrics')

parser.add_argument('-gpus', type=str, default='0,1,2,3')


args = parser.parse_args()
argparse_dict = vars(args)



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
    if args.seed is None:
        launch_seed = 10*run
    else:
        launch_seed = args.seed[run]

    args.launch_seed =  launch_seed

    torch.backends.cudnn.benchmark = False
    
    train_mode = args.backbone + '_'  + args.dataset

    if args.phase == 'train':
        args.nb_max_step_without_improvements = 15

    args.launch_mode = train_mode
    args.all_parameters = train_mode + '_seed' + str(launch_seed)

    setproctitle.setproctitle(args.backbone)

    random.seed(launch_seed)
    torch.manual_seed(launch_seed)
    if args.cuda:
        torch.cuda.manual_seed(launch_seed)
    checkpoint_path = os.path.join(args.checkpoint_path, train_mode, '_seed' + str(launch_seed))

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    args.model_path = checkpoint_path

    if args.phase == 'train':
        if not os.path.exists(os.path.join(checkpoint_path, 'parameters.json')):
            with open(os.path.join(checkpoint_path, 'parameters.json'), 'w') as fp:
                json.dump(argparse_dict, fp,  indent=4)
        else:
            with open(os.path.join( checkpoint_path, 'parameters.json')) as f:
                saved_dict = json.load(f)
            assert(argparse_dict['all_parameters'] ==  saved_dict['all_parameters'])

    
    print(colored('Loading data', 'blue'))

    
    
    backbone = get_backbone_model(args)



    if args.phase == 'train':
        print(colored('\n\nModel to train {}'.format(train_mode), 'blue'))
        print(colored('Where to save model {} \n'.format(checkpoint_path), 'blue'))
    elif args.phase != 'train':
        print(colored('\n\nTrained model {}'.format(checkpoint_path), 'blue'))
    

    if args.only_load_data:
        return

    print('training for dataset  ' + args.dataset)

    model_obj = ImageModel(flags=args,
                                backbone=backbone,
                                checkpoint_path=checkpoint_path,
                                train_mode=train_mode)

    if args.phase == 'train':
        model_obj.train_triplet(flags=args)


    args.model_path = None


def runs():
    for i in range(len(args.seed)):
        print('Run {}'.format(i))
        running(i)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    system_info()
    runs()