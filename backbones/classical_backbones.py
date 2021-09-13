from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
import glob
import os
import h5py

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def random_forest_backbone(X, y):
    '''X, y = make_classification(n_samples=1000, n_features=4,
                               n_informative=2, n_redundant=0,
                               random_state=0, shuffle=False)'''
    print(X.shape)
    clf = RandomForestClassifier( random_state=0)
    clf.fit(X, y)

    print(clf.predict([[0, 0, 0, 0]]))

def svm_backbone(X_train, y_train, X_test, y_test):
    print('SVM')
    #X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    #y = np.array([1, 1, 2, 2])
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    from sklearn.svm import SVC
    print(X_train.shape)
    print(y_train.shape)
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)

    predicted = clf.predict(X_test)
    print(accuracy_score(y_test, predicted))
    print(confusion_matrix(y_test, predicted))
    print('\n')

def knn_backbone(X_train, y_train, X_test, y_test):
    print('KNN')
    #X = [[0], [1], [2], [3]]
    #y = [0, 0, 1, 1]
    #print(X.shape)
    #print(y.shape)
    y_train = np.squeeze(y_train)

    y_train = y_train.tolist()
    X_train = X_train.tolist()
    neigh = KNeighborsClassifier(n_neighbors=2)
    neigh.fit(X_train, y_train)
    predicted = neigh.predict(X_test)

    print(accuracy_score(y_test, predicted))
    print(confusion_matrix(y_test, predicted))
    print('\n')

    #print(neigh.predict([[1.1]]))

    #print(neigh.predict_proba([[0.9]]))


if __name__ == '__main__':
    train_files = sorted(glob.glob(os.path.join('../Datasets', '*train.h5')))
    test_files = sorted(glob.glob(os.path.join('../Datasets', '*test.h5')))
    for i in range(len(train_files)):



        train_file = train_files[i]
        test_file = test_files[i]

        print(train_file)

        hf = h5py.File(train_file, 'r')
        input_train = hf['series']
        input_train = np.array(input_train)
        label_train = hf['labels']
        label_train = np.array(label_train)

        hf = h5py.File(test_file, 'r')
        input_test = hf['series']
        input_test = np.array(input_test)
        label_test = hf['labels']
        label_test = np.array(label_test)

        #random_forest_backbone(input, label)
        svm_backbone(input_train, label_train, input_test, label_test)
        knn_backbone(input_train, label_train, input_test, label_test)
        print('\n\n')




