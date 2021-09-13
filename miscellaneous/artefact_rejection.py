from sklearn.datasets import load_digits
# from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import pickle as pkl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
import sklearn
import time
import h5py
import tqdm

import glob


def check_baby_in_list(list_sepsis, baby_number):
    for sepsis in list_sepsis:
        #print(baby_number)
        #print(sepsis)
        if str(baby_number) in sepsis:
            return True, sepsis

    return False, None


def get_baby_los(data, baby_number):
    for i in range(data.shape[0]):
        if data[i, 0] == baby_number:
            return i

    return None


def get_number_not_infected(list_images):

    nb = 0

    infected_images = []
    not_infected_images = []

    for link_image in list_images:
        if 'not_infected' in link_image:
            not_infected_images.append(link_image)
        else:
            infected_images.append(link_image)

    return infected_images, not_infected_images


def artefact_rejection(flags):

    from load_data import select_in_time_intervalle, get_dict_babies

    root_dir = os.path.join('Datasets', 'real_data')

    dataset_name = 'spectrogram_48_2_30'

    data_path = os.path.join(root_dir, dataset_name)

    print(data_path)

    test_no_sepsis = sorted(
        glob.glob(os.path.join(data_path, 'test_no_sepsis', '*_ecg.pkl')))
    test_bb_no_sepsis = select_in_time_intervalle(
        flags, get_dict_babies(test_no_sepsis))

    #print(test_bb_no_sepsis)

    test_sepsis = sorted(
        glob.glob(os.path.join(data_path, 'test_sepsis', '*_ecg.pkl')))
    test_bb_sepsis = select_in_time_intervalle(flags,
                                               get_dict_babies(test_sepsis))

    train_no_sepsis = sorted(
        glob.glob(os.path.join(data_path, 'train_no_sepsis', '*_ecg.pkl')))
    train_bb_no_sepsis = select_in_time_intervalle(
        flags, get_dict_babies(train_no_sepsis))

    train_sepsis = sorted(
        glob.glob(os.path.join(data_path, 'train_sepsis', '*_ecg.pkl')))
    train_bb_sepsis = select_in_time_intervalle(flags,
                                                get_dict_babies(train_sepsis))


    # ##Collect sepsis data
    # list_train = sorted(
    #     glob.glob(os.path.join(root_dir, dataset_name, 'train', '*.pkl')))
    # list_test = sorted(
    #     glob.glob(os.path.join(root_dir, dataset_name, 'test', '*.pkl')))

    # list_train_infected, list_train_not_infected = get_number_not_infected(
    #     list_train)
    # list_test_infected, list_test_not_infected = get_number_not_infected(
    #     list_test)

    # print(len(list_train_infected))
    # print(len(list_train_not_infected))
    # print(len(list_test_infected))
    # print(len(list_test_not_infected))

    # list_train_infected = list_train_infected[0:100]
    # list_train_not_infected = list_train_not_infected[0:100]
    # list_test_infected = list_test_infected[0:100]
    # list_test_not_infected = list_test_not_infected[0:100]

    train_images_no_sepsis = []
    for baby in train_bb_no_sepsis.keys():
        train_images_no_sepsis.extend(train_bb_no_sepsis[baby])
    train_images_sepsis = []
    for baby in train_bb_sepsis.keys():
        train_images_sepsis.extend(train_bb_sepsis[baby])

    test_images_no_sepsis = []
    for baby in test_bb_no_sepsis.keys():
        test_images_no_sepsis.extend(test_bb_no_sepsis[baby])
    test_images_sepsis = []
    for baby in test_bb_sepsis.keys():
        test_images_sepsis.extend(test_bb_sepsis[baby])

    data_all = []
    data_all.extend(train_images_no_sepsis[0:min(3000, len(train_images_no_sepsis))])
    data_all.extend(train_images_sepsis[0:min(3000, len(train_images_sepsis))])


    # data_all.extend(test_bb_no_sepsis)
    # data_all.extend(test_bb_sepsis)
    # data_all.extend(train_bb_no_sepsis)
    # data_all.extend(train_bb_sepsis)

    features = None
    labels = []

    print('nombre de series {}'.format(len(data_all)))

    for it, pkl_file in tqdm.tqdm(enumerate(data_all)):

        with open(pkl_file, 'rb') as f:
            data_here = pkl.load(f)

            if data_here.shape[0] < 800000:
                continue

            data_here = data_here[0:800000]

            values = []
            for i in range(data_here.shape[0]):
                if i % 250 ==0:
                    values.append(data_here[i])

            data_here = np.asfarray(values)


            data_here = np.reshape(data_here, (1, data_here.shape[0]))
            if features is None:
                features = data_here
            else:
                features = np.append(features, data_here, axis=0)

        if 'not_infected' in pkl_file:
            labels.append(0)
        else:
            labels.append(1)

    print(features.shape)
    labels = np.asarray(labels)

    #labels = np.reshape(labels, (labels.shape[0]))

    print(features.shape)
    from sklearn.cluster import DBSCAN

    X = features
    clustering = DBSCAN(eps=10000, min_samples=5).fit(X)
    print(clustering.labels_)





if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-inter_before',
                        type=int,
                        default=6,
                        help='time before in hours')
    parser.add_argument('-inter_after',
                        type=int,
                        default=2,
                        help='time after in hours')
    parser.add_argument('-data_start',
                        type=int,
                        default=48,
                        help='time before in hours')

    parser.add_argument('-data_end',
                        type=int,
                        default=2,
                        help='time before in hours')

    parser.add_argument('-type_data',
                        type=str,
                        default='real_data',
                        help='real_data, sinus, ecgsyn')

    parser.add_argument('-data_root', type=str, default='Datasets')

    args = parser.parse_args()

    artefact_rejection(args)

