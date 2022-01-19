#!/bin/python3


from pipeline_modules.feature_extraction import FeatureExtractor
from pipeline_modules.clustering import Clustering
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def main():
    # Classification pipeline:
    f = FeatureExtractor()
    c = Clustering(f.descriptor_list, f.num_of_samples, f.grey_dataset_flat_list, f.sift)

    x = [i[0] for i in c.bag_of_words]
    y = [i[1] for i in c.bag_of_words]
    print(np.shape(x), np.shape(y))
    #svm
    clf = svm.NuSVC()
    clf.fit(x, y)
    # knn
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x, y)
    acc_svm = 0
    acc_knn = 0
    for hist, c in c.bag_of_words:
        pred_svm = clf.predict([hist])
        pred_knn = neigh.predict([hist])
        print(f'pred_svm: {pred_svm}, pred knn: {pred_knn}, real class: {c}')
        if pred_svm[0] == c:
            acc_svm += 1
        if pred_knn[0] == c:
            acc_knn += 1
    print(f'final svm: {acc_svm/len(y)}, final_knn: {acc_knn/len(y)}')

    # print(c.bag_of_words[10].sum())


if __name__ == '__main__':
    main()
