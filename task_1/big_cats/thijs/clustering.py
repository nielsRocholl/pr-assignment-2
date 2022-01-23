from sklearn.cluster import MiniBatchKMeans
import numpy as np

def k_means(descriptor_list: list, k=25):
    """
    Loads the kmeans model if it exists, else it creates the model
    :return: kmeans model
    """
    return MiniBatchKMeans(n_clusters=k, verbose=0).fit(descriptor_list)


def bag_of_words(kmeans, data):
    """
    Loads the histogram list (bag of (visual) words) if it exists, else it creates the bag of words.
    Every histogram should sum to 1, i.e. every entry represents the frequency of that particular descriptor
    :return: list of histograms, (bag of words)
    """
    # print("Creating histograms for the bag of words")
    k = kmeans.n_clusters
    hist_list = []

    for class_data in data:
        class_histograms = []
        for des in class_data:
            hist = np.zeros(k)
            nkp = np.size(des)
            for d in des:
                idx = kmeans.predict([d])
                hist[idx] += 1 / nkp  # Because we need normalized histograms, I prefere to add 1/nkp directly

            class_histograms.append(hist)
        hist_list.append(class_histograms)

    # print("Histograms successfully created")
    return hist_list
