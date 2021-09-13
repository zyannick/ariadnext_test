import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import sklearn.metrics as metrics


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=1)


def unfold_label(labels, classes):
    # can not be used when classes are not complete
    new_labels = []

    assert len(np.unique(labels)) == classes
    # minimum value of labels
    mini = np.min(labels)

    for index in range(len(labels)):
        dump = np.full(shape=[classes], fill_value=0).astype(np.int8)
        _class = int(labels[index]) - mini
        dump[_class] = 1
        new_labels.append(dump)

    return np.array(new_labels)


def shuffle_data(samples, labels):
    num = len(labels)
    shuffle_index = np.random.permutation(np.arange(num))
    shuffled_samples = samples[shuffle_index]
    shuffled_labels = labels[shuffle_index]
    return shuffled_samples, shuffled_labels


def shuffle_list(li):
    np.random.shuffle(li)
    return li


def shuffle_list_with_ind(li):
    shuffle_index = np.random.permutation(np.arange(len(li)))
    li = li[shuffle_index]
    return li, shuffle_index


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3):

    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda


def crossentropyloss():
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn


def weighted_cross_entropy_with_logits(logits, targets, pos_weight):
    return targets * -logits.sigmoid().log() * pos_weight + (
        1 - targets) * -(1 - logits.sigmoid()).log()


def mseloss():
    loss_fn = torch.nn.MSELoss()
    return loss_fn


def sgd(parameters, lr, weight_decay=0.00005, momentum=0.9):
    opt = optim.SGD(params=parameters,
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay)
    return opt


def write_log(log, log_path):
    f = open(log_path, mode='a')
    f.write(str(log))
    f.write('\n')
    f.close()


def fix_python_seed(seed):
    print('seed-----------python', seed)
    random.seed(seed)
    np.random.seed(seed)


def fix_torch_seed(seed):
    print('seed-----------torch', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fix_all_seed(seed):
    print('seed-----------all device', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_preds(logits):
    class_output = F.softmax(logits, dim=1)
    pred_task = class_output.data.max(1, keepdim=True)[1]
    return pred_task


def compute_accuracy(predictions, labels, probabilities = None):
    if np.ndim(labels) == 2:
        y_true = np.argmax(labels, axis=-1)
    else:
        y_true = labels

    if np.ndim(predictions) == 2:
        y_pred = np.argmax(predictions, axis=-1)
    else:
        y_pred = predictions

    if np.ndim(probabilities) == 2:
        probabilities = np.argmax(probabilities, axis=-1)


    accuracy = accuracy_score(y_true=y_true, y_pred = y_pred )
    f1_sco = f1_score(y_true, y_pred, average='macro')
    cf = confusion_matrix(y_true, y_pred)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)

    auc = metrics.auc(fpr, tpr)
    if probabilities is not None:
        auc_soft = roc_auc_score(y_true, probabilities)
    else:
        auc_soft = -1

    return accuracy, f1_sco, cf, auc, auc_soft


def cross_batch_accuracy(per_frame_logits, labels):
    y_predicted = torch.max(per_frame_logits, dim=2)[0]
    y_predicted = y_predicted.cuda()
    y_predicted = y_predicted.cpu().detach().numpy()
    y_predicted = np.argmax(y_predicted, axis=1)

    y_ground_truth = torch.max(labels, dim=2)[0]
    y_ground_truth = y_ground_truth.cuda()
    y_ground_truth = y_ground_truth.cpu().detach().numpy()
    y_ground_truth = np.argmax(y_ground_truth, axis=1)

    # print(y_predicted)
    # print(y_ground_truth)
    # print('\n')

    from sklearn.metrics import multilabel_confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    # print(multilabel_confusion_matrix(y_ground_truth, y_predicted))
    # print('Cross batch accuracy %4.4f'%(accuracy_score(y_ground_truth, y_predicted)))
    return accuracy_score(y_ground_truth,
                          y_predicted), f1_score(y_ground_truth,
                                                 y_predicted,
                                                 average='weighted')


def per_video_accuracy(per_frame_logits, labels):
    y_predicted = per_frame_logits.cuda()
    y_predicted = y_predicted.cpu().detach().numpy()

    y_ground_truth = labels.cuda()
    y_ground_truth = y_ground_truth.cpu().detach().numpy()

    nb_batches = y_predicted.shape[0]

    #i = random.randint(0, nb_batches - 1)

    r_shape = y_predicted.shape

    #print(r_shape)

    ground_truth = []
    predictions = []

    for i in range(nb_batches):

        y_predicted_video = y_predicted[i]
        y_ground_truth_video = y_ground_truth[i]

        y_predicted_video = np.reshape(y_predicted_video,
                                       (r_shape[1], r_shape[2]))
        y_ground_truth_video = np.reshape(y_ground_truth_video,
                                          (r_shape[1], r_shape[2]))

        y_predicted_video = np.argmax(y_predicted_video, axis=0)
        y_ground_truth_video = np.argmax(y_ground_truth_video, axis=0)

        from sklearn.metrics import accuracy_score
        # print('y_predicted_video {}'.format(y_predicted_video))
        # print('y_ground_truth_video {}'.format(y_ground_truth_video))

        ground_truth.extend(y_ground_truth_video.tolist())
        predictions.extend(y_predicted_video.tolist())

    # print('predictions {}'.format(predictions))
    # print('ground_truth {}'.format(ground_truth))

    return predictions, ground_truth
