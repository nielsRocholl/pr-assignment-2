import numpy as np
from sklearn.cluster import MiniBatchKMeans, Birch


def k_means(data, num_of_classes):
    """
    Create a kmeans model based on all sift features
    :return: kmeans model
    """
    descriptor_list = []
    for features in data:
        for d in features:
            descriptor_list.append(d)

    k = num_of_classes * 10
    return MiniBatchKMeans(n_clusters=k, verbose=0).fit(descriptor_list)


def birch(data, num_of_classes):
    """
    Create a birch model based on all sift features
    :return: birch model
    """
    descriptor_list = []
    for features in data:
        for d in features:
            descriptor_list.append(d)

    k = num_of_classes * 10
    return Birch(n_clusters=k).fit(descriptor_list)


def bag_of_words(kmeans, data, num_of_classes):
    """
    Every histogram should sum to 1, i.e. every entry represents the frequency of that particular descriptor
    :return: list of histograms, (bag of words)
    """
    k = num_of_classes * 10

    kmeans.verbose = False
    hist_list = []

    for des, kp, label in data:

        hist = np.zeros(k)
        nkp = np.size(kp)

        for d in des:
            idx = kmeans.predict([d])
            hist[idx] += 1 / nkp  # normalize hist

        hist_list.append([hist, label])
    return hist_list
