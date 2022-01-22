from multiprocessing.spawn import import_main_path
import os
from os.path import join
from sklearn import svm

from load_images import load_images
from feature_extraction import feature_extraction
from validation import K_folds
from clustering import k_means, bag_of_words

import random
random.seed(0)

def flatten_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def flatten_list_of_lists(list_of_lists):
    return flatten_list(flatten_list(list_of_lists))


def main():
    # Load the images in greyscale
    images = {class_name: load_images(join("data", class_name), greyscale=True) for class_name in CLASSES}

    # Extract the featues
    features = {class_name: feature_extraction(images[class_name], class_name=class_name) for class_name in CLASSES}
    # Remove original images to save memory 
    del images

    # Create the K-folds
    folded_classes = [K_folds(features[class_name]) for class_name in CLASSES]

    for fold, data in enumerate(zip(*folded_classes)):
        train_set = [[dp for dp in c[0]] for c in data]
        test_set = [[dp for dp in c[1]] for c in data]
        
        kmeans = k_means(flatten_list_of_lists(train_set))
        histograms = bag_of_words(kmeans, train_set)

        x = flatten_list(histograms)
        y = [item for x in ([CLASSES[idx]] * len(histograms[idx]) for idx in range(len(histograms))) for item in x]

        clf = svm.NuSVC()
        clf.fit(x, y)
        # perform testing
        acc_svm = 0
        bow_test = bag_of_words(kmeans=kmeans, data=test_set)
        for c, class_name in zip(bow_test, CLASSES):
            for hist in c:
                pred_svm = clf.predict([hist])
                if pred_svm == class_name:
                    acc_svm += 1
        print(f'Fold: {fold}, final svm: {acc_svm / len(flatten_list(test_set))}')
        return




if __name__=="__main__":
    os.chdir(os.path.split(__file__)[0])
    CLASSES = ["Cheetah", "Jaguar", "Leopard", "Lion", "Tiger"]
    AMOUNT_OF_KMEANS_CLUSTERS = 25
    main()
