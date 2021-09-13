try:
    from visualize_window import *
except:
    print('no qt')
from multiprocessing import Value
from models.models_images import ImageModel
from models.models_videos import VideoModel
from models.combined_models_images import CombinedImageModel
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
dict_mode = {
    'train_model': 'T',
    'simulate_data': 'S',
    'display_data': 'D'
}
parser.add_argument('-backbone',type=str, default='ddss_resnet18', help='ddss_alexnet, ddss_resnet18, ddss_vit')
dict_backbone = {
    'ddss_alexnet': 'T',
    'ddss_resnet18': 'R18',
    'ddss_resnet34': 'R34',
    'ddss_resnet50': 'R50',
    'ddss_vit': 'V'
}
parser.add_argument('-denoise', type=str, default=None, help='')

parser.add_argument('-batch_size', type=int, default=16)
parser.add_argument('-seed', type=string_to_int_list, default='0')
#parser.add_argument('-n_runs', type=int, default=1)
parser.add_argument('-dataset', type=str, default='dgnewb')
parser.add_argument('-type_data',type=str, default='sinus', help='real_data, sinus, ecgsyn, real_data_rr')
parser.add_argument( '-phase', type=str,  default='train',  help='train, extract_features, vizualize, timeline_prediction')
parser.add_argument('-type_input', type=str, default='image')
parser.add_argument('-type_signal', type=str, default='rr', help='ecg, resp, rr, ttot')
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
parser.add_argument("-optimizer", type=str, default='SGD', help='SGD, Adam')
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
parser.add_argument("-model_path", type=str, default='fc', help='')
parser.add_argument("-state_dict", type=str, default='', help='')
parser.add_argument('-verbose', default=True, action='store_false')
parser.add_argument('-modify_sampling', type=str,  default='no_balance', help='no_balance, over, under')
parser.add_argument('-balance_sampler', type=boolean_string, default=False)
parser.add_argument('-alpha_sampler', type=float, default=1.0)
parser.add_argument('-class_weighting', type=boolean_string, default=False)
parser.add_argument('-continue_training', type=boolean_string, default=False)
parser.add_argument('-random_splitting', type=boolean_string, default=False)
parser.add_argument('-only_load_data', type=boolean_string, default=False)
parser.add_argument('-augment', type=boolean_string, default=False)
parser.add_argument('-log_spectrogram', type=boolean_string, default=True)
parser.add_argument('-is_searching', type=boolean_string, default=False)
parser.add_argument('-validate', type=boolean_string, default=True)
parser.add_argument("-sheduler_type", type=str, default=None, help='StepLR, CyclicLR, ReduceLROnPlateau,None')
parser.add_argument( '-load_checkpoint', type=int, default=19,  metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')

# data generation

parser.add_argument('-num', type=int, default=100,  help='number of signals per class')
parser.add_argument('-fe', type=float, default=4, help='Sampling rate')
parser.add_argument('-data_root', type=str, default='Datasets')
parser.add_argument('-data_path', type=str, default='real_data_rr_30_360')



parser.add_argument('-type_transform', type=str, default='spectrogram',  help='psd, spectrogram, spectrogram_time_frequency')

parser.add_argument('-outliers_mode',   type=str,  default=None,  help='25_75, 10_90, 01_99, chr')

parser.add_argument('-video_from_ecgs', type=boolean_string, default=False)

parser.add_argument('-writer_dir',type=str, default='runs',  help='Where to save metrics')

# parameter of real data
parser.add_argument('-vizualize_target',type=str, default='test',  help='train, val, test')
parser.add_argument('-details_viz',type=boolean_string, default=True,  help='Display frequence and time')
parser.add_argument('-numbers_to_vizualize',type=int, default=5,  help='The number of images to vizualize for each class')
parser.add_argument('-mean_viz',type=boolean_string, default=True,  help='Display frequence and time')



parser.add_argument('-sftp_window', type=int, default=3000)
parser.add_argument('-duration', type=int, default=1800)
parser.add_argument('-inter_before',   type=int, default=6,  help='time before in hours')
parser.add_argument('-inter_after', type=int,  default=2,  help='time after in hours')
parser.add_argument('-timeline_start', type=int,  default=48,  help='time before in hours')
parser.add_argument('-timeline_end', type=int, default=2, help='time before in hours')
parser.add_argument('-timeline_target',type=str, default='test',  help='train, val, test')
parser.add_argument('-timeline_step', type=float, default=6, help='Steps in minuts')
parser.add_argument('-data_start',  type=int,  default=48,   help='time before in hours')
parser.add_argument('-data_end',  type=int, default=2,  help='time before in hours')
parser.add_argument('-duration_taken',  type=int,  default=30,  help='taken duration in minutes')
parser.add_argument('-num_frames', type=int,    default=16, help='number of frames')
parser.add_argument('-video_sampling_non_overlap', type=int,   default=8, help='number of frames')
# Spectogram
parser.add_argument('-type_spectograme', type=str,  default='matplotlib',   help='matplotlib, scipy')
parser.add_argument('-gpus', type=str, default='0')


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

seeds = [1, 10, 100]

def get_model_name():

    print('\n\n')
    for cle, values in argparse_dict.items():
        print(cle + ' --> ' + str(values))
    print('\n\n')

    if (args.backbone == 'fully_connected' and args.type_input
            == 'image') or (args.backbone != 'fully_connected'
                            and args.type_input == 'series'):
        raise ValueError('Error in training')

    list_dirs = []
    list_dirs.append(args.backbone + '_' + args.type_data + '_' +
                     args.type_input + '_' + args.dataset)

    if 'real_data' in args.type_data:

        if 'real_data' in args.type_data and args.sftp_window > 0:
            list_dirs.append(str(args.sftp_window))

        list_dirs.append('_' + str(args.inter_before) + '_'  + str(args.inter_after))

        if args.augment:
            list_dirs.append('A')
        else:
            list_dirs.append('N')

        if args.class_weighting:
            list_dirs.append('C')
        else:
            list_dirs.append('N')

        if args.log_spectrogram:
            list_dirs.append('L')

        list_dirs.append(args.type_transform)

        if args.modify_sampling != 'no_balance':
            args.balance_sampler = False
            list_dirs.append('N')

        if args.validate:
            list_dirs.append('V')
        else:
            list_dirs.append('N')

        if args.balance_sampler:
            list_dirs.append('S' + str(args.alpha_sampler))
        else:
            list_dirs.append('N')

        if args.data_path is not None:
            list_dirs.append(args.data_path)

        if args.dropout > 0:
            list_dirs.append('dropout_' + str(int(100 * args.dropout)))

        if args.denoise is not None:
            list_dirs.append(args.denoise)

        if not ((args.outliers_mode is None) or (args.outliers_mode == 'None')):
            list_dirs.append(args.outliers_mode)
        else:
            list_dirs.append('not_outlier_suppression')

        if args.random_splitting:
            list_dirs.append('random_splitting')

        if args.video_from_ecgs:
            list_dirs.append('videoecg')
            list_dirs.append(str(args.num_frames))

        #list_dirs.append( str(args.optimizer) + '_' + str(args.sheduler_type)  + '_' + str(args.init_lr) + '_' + str(args.step_size) + '_' + str(args.l1_lambda) + '_' + str(args.weight_decay) )

    return list_dirs


def running(run):

    # Setting seed
    if args.seed is None:
        launch_seed = 10*run
    else:
        launch_seed = args.seed[run]

    args.launch_seed =  launch_seed

    torch.backends.cudnn.benchmark = False
    list_dirs = get_model_name()
    
    train_mode = '_'.join(list_dirs)

    if args.phase == 'train':
        args.nb_max_step_without_improvements = 15

    args.launch_mode = train_mode
    args.all_parameters = train_mode + '_seed' + str(launch_seed)

    setproctitle.setproctitle(list_dirs[0])

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

    backbone = get_backbone_model(args)

    print(colored('Loading data', 'blue'))
    datafiles, taille, les_poids, mean_std, list_idx = get_data_classical_loss(args)

    #print('les poids {}'.format(les_poids))

    if taille == 0:
        print(colored('No data', 'red'))
        return

    if args.phase == 'train':
        print(colored('\n\nModel to train {}'.format(train_mode), 'blue'))
        print(colored('Where to save model {} \n'.format(checkpoint_path), 'blue'))
    elif args.phase != 'train':
        print(colored('\n\nTrained model {}'.format(checkpoint_path), 'blue'))
    
    if args.phase == 'timeline_prediction':
        print(colored('Timeline prediction {}  from {} to {}'.format(args.timeline_target, args.timeline_start, args.timeline_end), 'red'))


    if args.only_load_data:
        return

    print('training for dataset  ' + args.dataset)

    if args.type_input == 'image':
        if args.type_signal != 'rr_and_ttot':
            model_obj = ImageModel(flags=args,
                                backbone=backbone,
                                datafiles=datafiles,
                                checkpoint_path=checkpoint_path,
                                class_idxs=list_idx,
                                train_mode=train_mode)
        else:
            model_obj = CombinedImageModel(flags=args,
                                backbone=backbone,
                                datafiles=datafiles,
                                checkpoint_path=checkpoint_path,
                                class_idxs=list_idx,
                                train_mode=train_mode)
    else:
        model_obj = VideoModel(flags=args,
                               backbone=backbone,
                               datafiles=datafiles,
                               checkpoint_path=checkpoint_path,
                               train_mode=train_mode,
                               data_weitghs=les_poids,
                               mean_std=mean_std)
    if args.phase == 'train':
        model_obj.train_triplet(flags=args)
    elif args.phase == 'extract_features':
        model_obj.extract_features(flags=args)
    elif args.phase == 'saliency':
        model_obj.visualize_saliency_maps(flags=args)
    elif args.phase == 'vizualize':
        model_obj.visualize_features_maps(flags=args)
    elif args.phase == 'timeline_prediction':
        if args.type_input == 'video':
            model_obj.timeline_video_prediction(flags=args)
        else:
            model_obj.timeline_prediction(flags=args)

    args.model_path = None


def runs():
    if args.type_data == 'ecgsyn':
        list_dirs = sorted(glob.glob(os.path.join('./Datasets', 'ecgsyn_matplotlib', '*')))
        for i, data_dir in enumerate(list_dirs):
            print(data_dir)
            args.data_root = data_dir
            args.dataset = data_dir.split(os.sep)[-1]
            running(i)
    if args.type_data == 'sinus':
        # list_dirs = sorted(glob.glob(os.path.join('./Datasets', 'simulations', 'sinus', '*')) )
        # print(list_dirs)
        # for i, data_dir in enumerate(list_dirs):
        #     print(data_dir)
        #     args.data_root = data_dir
        #     args.dataset = data_dir.split(os.sep)[-1]
        running(0)
    else:
        for i in range(len(args.seed)):
            print('Run {}'.format(i))
            running(i)


if __name__ == '__main__':

    # need to add argparse
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    system_info()
    runs()
