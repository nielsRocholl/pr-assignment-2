import numpy as np
from sklearn.cluster import MiniBatchKMeans


def k_means(data, num_of_classes):
    """
    Loads the kmeans model if it exists, else it creates the model
    :return: kmeans model
    """
    descriptor_list = []
    for features in data:
        for d in features:
            descriptor_list.append(d)

    k = num_of_classes * 10
    batch_size = np.size(len(data)) * 3
    print("Calculating kmeans. This might take a minute")
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, verbose=0).fit(descriptor_list)
    print("Kmeans successfully created")
    return kmeans


def bag_of_words(kmeans, data, num_of_classes):
    """
    Loads the histogram list (bag of (visual) words) if it exists, else it creates the bag of words.
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
            hist[idx] += 1 / nkp  # Because we need normalized histograms, I prefere to add 1/nkp directly

        hist_list.append([hist, label])
    print("Histogram successfully created")
    return hist_list
