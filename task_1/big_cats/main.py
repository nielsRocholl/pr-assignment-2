#!/bin/python3
import random

from pipeline_modules.feature_extraction import load_dataset
from pipeline_modules.clustering import k_means, bag_of_words
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def main():
    # Classification pipeline:
    data = load_dataset()
    # shuffle data
    random.shuffle(data)
    # split data into train and test
    data_train = data[0: int(len(data) * .8)]
    data_test = data[int(len(data) * .8):]
    # use features to train knn
    features = [i[0] for i in data_train]
    kmeans = k_means(data=features, num_of_classes=5)
    # calculate bag of words representation
    bow = bag_of_words(kmeans=kmeans, data=data_train, num_of_classes=5)

    x = [i[0] for i in bow]  # histogram
    y = [i[1] for i in bow]  # label
    # svm
    clf = svm.NuSVC()
    clf.fit(x, y)
    # knn
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x, y)
    # perform testing
    acc_svm = 0
    acc_knn = 0
    bow_test = bag_of_words(kmeans=kmeans, data=data_test, num_of_classes=5)
    for hist, c in bow_test:
        pred_svm = clf.predict([hist])
        pred_knn = neigh.predict([hist])
        print(f'pred_svm: {pred_svm}, pred knn: {pred_knn}, real class: {c}')
        if pred_svm[0] == c:
            acc_svm += 1
        if pred_knn[0] == c:
            acc_knn += 1
    print(f'final svm: {acc_svm / len(y)}, final_knn: {acc_knn / len(y)}')


if __name__ == '__main__':
    main()
