#!/bin/python3
import random
import numpy as np

from pipeline_modules.feature_extraction import load_dataset
from pipeline_modules.clustering import k_means, bag_of_words
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold


def main():
    # Classification pipeline:
    data = load_dataset()
    # shuffle data
    random.shuffle(data)
    # create fold
    kf = KFold(n_splits=5, random_state=None)
    for fold, (train_index, test_index) in enumerate(kf.split(data)):
        data_train, data_test = np.array(data, dtype=object)[train_index, :], np.array(data, dtype=object)[test_index, :]
        # use features to train knn
        features = [i[0] for i in data_train]
        kmeans = k_means(data=features, num_of_classes=5)
        # calculate bag of words representation
        bow_train = bag_of_words(kmeans=kmeans, data=data_train, num_of_classes=5)
        # retrieve x and y to train classifier
        x = [i[0] for i in bow_train]  # histogram
        y = [i[1] for i in bow_train]  # label
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
            # print(f'pred_svm: {pred_svm}, pred knn: {pred_knn}, real class: {c}')
            if pred_svm[0] == c:
                acc_svm += 1
            if pred_knn[0] == c:
                acc_knn += 1
        print(f'Fold: {fold}, final svm: {acc_svm / len(y)}, final_knn: {acc_knn / len(y)}')


if __name__ == '__main__':
    main()
