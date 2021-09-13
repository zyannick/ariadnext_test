import numpy as np
import  cv2

def get_lfw_list(pairs):
    data_list = []
    for pair in pairs:
        #splits = pair.split()

        if pair[0] not in data_list:
            data_list.append(pair[0])

        if pair[1] not in data_list:
            data_list.append(pair[1])
    return data_list

def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
    return fe_dict

def load_image(img_path):
    image = cv2.imread(img_path, 0)
    image = cv2.resize(image, dsize=(128, 128))
    if image is None:
        return None
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image = (image / 255.) * 2 - 1
    return image


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def test_performance(fe_dict, pairs):

    sims = []
    labels = []
    for pair in pairs:
        # splits = pair.split()
        fe_1 = fe_dict[pair[0]]
        fe_2 = fe_dict[pair[1]]
        label = int(pair[2])
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th