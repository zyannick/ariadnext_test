import os
import random
from builtins import type

import torch
from .utils import *
from glob import glob
from collections import Counter
from termcolor import colored
import cv2
from tqdm import tqdm
import json
from itertools import combinations
import copy


def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights


def compute_surface():
    return


def get_data(flags):
    json_filename = os.path.join(flags.dataroot, flags.dataset, 'croppped_images.json')
    taille = 0

    if not os.path.exists(json_filename):

        # to find path of xml file containing haarCascade file
        cfp = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
        # load the harcaascade in the cascade classifier
        face_cascade = cv2.CascadeClassifier(cfp)

        list_names_dir = sorted(glob(os.path.join(flags.dataroot, flags.dataset, flags.dataset, '*')))
        random.shuffle(list_names_dir)

        dict_data = {}

        for _, class_dir in tqdm(enumerate(list_names_dir)):
            list_imgs_per_class = sorted(glob(os.path.join(class_dir, '*')))
            name = class_dir.split(os.sep)[-1]

            dict_data[name] = []
            for img_path in list_imgs_per_class:
                frame = cv2.imread(img_path, 1)
                dict_data[name].append((img_path, 0))


        json_filename = os.path.join(flags.dataroot, flags.dataset, 'croppped_images.json')
        with open(json_filename, 'w') as outfile:
            json.dump(dict_data, outfile, indent=4)

    else:
        json_filename = os.path.join(flags.dataroot, flags.dataset, 'croppped_images.json')
        with open(json_filename) as json_f:
            dict_data = json.load(json_f)

    list_names = list(dict_data.keys())
    nb_names = len(list_names)

    train_data = []
    nb_classes_train = 0
    for label, name in enumerate(list_names[0:int(0.8 * nb_names)]):
        for filename, label in dict_data[name]:
            train_data.append(
                (filename, label)
            )
            taille += 1
        nb_classes_train += 1

    test_data = []
    nb_classes_test = 0
    #list names with more than 1 image
    list_more_1_image = []
    list_only_1_image = []
    for label, name in enumerate(list_names[int(0.8 * nb_names):-1]):
        if len(dict_data[name]) >= 2:
            list_more_1_image.append(name)
        if len(dict_data[name]) == 1:
            list_only_1_image.append(name)
        for filename, label in dict_data[name]:
            test_data.append(
                (filename, label)
            )
            taille += 1
        nb_classes_test += 1


    matched_pairs = []
    for name in list_more_1_image:
        all_combi_names = list(combinations(dict_data[name], 2))
        for pairs_files in all_combi_names:
            (first_file, l), (second_file, s) = pairs_files
            matched_pairs.append(
                (first_file, second_file, 1)
            )


    combi_only_one_file_names = list(combinations(list_only_1_image, 2))
    mismached_pairs = []
    for name1, name2 in combi_only_one_file_names:
        file1, label1 = dict_data[name1][0]
        file2, label2 = dict_data[name2][0]
        mismached_pairs.append(
            (file1, file2, 0)
        )

    random.shuffle(matched_pairs)
    random.shuffle(mismached_pairs)

    val_lists = copy.copy(matched_pairs[0:3000]) + copy.copy(mismached_pairs[0:3000])

    data_files = {'train': train_data, 'val': val_lists, 'test': []}

    return data_files, taille, nb_classes_train, nb_classes_test
