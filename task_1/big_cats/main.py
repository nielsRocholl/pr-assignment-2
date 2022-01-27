#!/bin/python3
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import NuSVC

from tqdm import tqdm
from pipeline_modules.feature_extraction import load_dataset
from pipeline_modules.clustering import k_means, bag_of_words, birch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def main():
    random.seed()
    # Classification pipeline:
    data = load_dataset(augment=False)
    # shuffle data
    random.shuffle(data)
    # create fold
    kf = KFold(n_splits=10, random_state=None)
    # create dict to store results
    results = {}
    # create all classifiers
    classifiers = [
        LinearDiscriminantAnalysis(),
        NuSVC(),
        KNeighborsClassifier(n_neighbors=10)
    ]
    # create all clustering methods
    clustering_methods = [
        k_means,
        birch
    ]
    for fold, (train_index, test_index) in tqdm(enumerate(kf.split(data))):
        data_train, data_test = np.array(data, dtype=object)[train_index, :], np.array(data, dtype=object)[test_index,:]
        for clustering_method in clustering_methods:
            # use features to train knn
            features = [i[0] for i in data_train]
            clustering_method = clustering_method(data=features, num_of_classes=5)
            # calculate bag of words representation
            bow_train = bag_of_words(kmeans=clustering_method, data=data_train, num_of_classes=5)
            # retrieve x and y to train classifier
            x = [i[0] for i in bow_train]  # histogram
            y = [i[1] for i in bow_train]  # label
            # perform testing
            bow_test = bag_of_words(kmeans=clustering_method, data=data_test, num_of_classes=5)
            for clf in classifiers:
                # train, predict and calculate accuracy
                clf.fit(x, y)
                pred = clf.predict([i[0] for i in bow_test])
                acc = accuracy_score([i[1] for i in bow_test], pred)
                acc = round(acc, 3)
                # add results to dictionary
                clf_name = clf.__class__.__name__
                clustering_method_name = clustering_method.__class__.__name__
                print(f'\nFold: {fold} |  {clf_name} + {clustering_method_name} : {acc}')
                if f'{clf_name} + {clustering_method_name}' in results:
                    results[f'{clf_name} + {clustering_method_name}'].append(acc)
                else:
                    results[f'{clf_name} + {clustering_method_name}'] = [acc]
    # save results to file
    df = pd.DataFrame(results)
    df.index.name = 'Fold'
    filename = 'big_cats_accuracy'
    # pd.DataFrame(results).to_csv(f'results{os.sep}{filename}.csv')
    print(f'Results written to file: {filename}.csv')


if __name__ == '__main__':
    main()
