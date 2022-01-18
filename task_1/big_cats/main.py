#!/bin/python3


from pipeline_modules.feature_extraction import FeatureExtractor
from pipeline_modules.clustering import Clustering
from sklearn import svm
import numpy as np


def main():
    # Classification pipeline:
    f = FeatureExtractor()
    c = Clustering(f.descriptor_list, f.num_of_samples, f.grey_dataset_flat_list, f.sift)

    x = [i[0] for i in c.bag_of_words]
    y = [i[1] for i in c.bag_of_words]
    print(np.shape(x), np.shape(y))
    clf = svm.NuSVC()
    clf.fit(x, y)
    acc = 0
    for hist, c in c.bag_of_words:
        pred = clf.predict([hist])
        print(f'pred: {pred}, real class: {c}')
        if pred[0] == c:
            acc += 1
    print(f'final: {acc/len(y)}')

    # print(c.bag_of_words[10].sum())


if __name__ == '__main__':
    main()
