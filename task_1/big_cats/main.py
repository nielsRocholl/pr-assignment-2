#!/bin/python3
import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.svm import NuSVC

from tqdm import tqdm
from pipeline_modules.feature_extraction import load_dataset
from pipeline_modules.clustering import k_means, bag_of_words
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def main():
    random.seed()
    # Classification pipeline:
    data = load_dataset()
    # shuffle data
    random.shuffle(data)
    # create fold
    kf = KFold(n_splits=10, random_state=None)
    results = {'LinearDiscriminantAnalysis Acc': [], 'NuSVC Acc': [], 'KNeighborsClassifier Acc': [],
               'LinearDiscriminantAnalysis Loss': [], 'NuSVC Loss': [], 'KNeighborsClassifier Loss': []}
    classifiers = [
        LinearDiscriminantAnalysis(),
        NuSVC(probability=True),
        KNeighborsClassifier(n_neighbors=10)
    ]

    print('Performing K-fold Cross Validation:')
    for fold, (train_index, test_index) in tqdm(enumerate(kf.split(data))):
        data_train, data_test = np.array(data, dtype=object)[train_index, :], np.array(data, dtype=object)[test_index,
                                                                              :]
        # use features to train knn
        features = [i[0] for i in data_train]
        kmeans = k_means(data=features, num_of_classes=5)
        # calculate bag of words representation
        bow_train = bag_of_words(kmeans=kmeans, data=data_train, num_of_classes=5)
        # retrieve x and y to train classifier
        x = [i[0] for i in bow_train]  # histogram
        y = [i[1] for i in bow_train]  # label

        # perform testing
        bow_test = bag_of_words(kmeans=kmeans, data=data_test, num_of_classes=5)
        for clf in classifiers:
            name = clf.__class__.__name__
            clf.fit(x, y)

            pred = clf.predict([i[0] for i in bow_test])
            acc = accuracy_score([i[1] for i in bow_test], pred)
            results[f'{name} Acc'].append(acc)

            pred = clf.predict_proba([i[0] for i in bow_test])
            loss = log_loss([i[1] for i in bow_test], pred, labels=[0, 1, 2, 3, 4])
            results[f'{name} Loss'].append(loss)
    pd.DataFrame(results).to_csv('results/big_cats_results_kmeans.csv')
    print('Results written to file: big_cats_results1.csv ')


if __name__ == '__main__':
    main()
