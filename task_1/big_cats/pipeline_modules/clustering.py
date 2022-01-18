import numpy as np
from sklearn.cluster import MiniBatchKMeans
from joblib import dump, load
from os.path import exists


class Clustering:
    def __init__(self, data, num_of_samples, grey_images, sift):
        # paths:
        self.__models_path = 'big_cats/models'

        # variables:
        self.num_of_classes = 5
        self.data = data
        self.grey_images = grey_images
        self.num_of_samples = num_of_samples
        self.k = self.num_of_classes * 10

        # models:
        self.kmeans = self.__kmeans()
        self.sift = sift
        self.bag_of_words = self.__hist()

    def __kmeans(self):
        """
        Loads the kmeans model if it exists, else it creates the model
        :return: kmeans model
        """
        if exists(f'{self.__models_path}/kmeans.joblib'):
            print("Kmeans successfully loaded from file")
            return load(f'{self.__models_path}/kmeans.joblib')
        else:
            batch_size = np.size(self.num_of_samples) * 3
            print("Calculating kmeans. This might take a minute")
            kmeans = MiniBatchKMeans(n_clusters=self.k, batch_size=batch_size, verbose=0).fit(self.data)

            dump(kmeans, f'{self.__models_path}/kmeans.joblib')
        print("Kmeans successfully created")
        return kmeans

    def __hist(self):
        """
        Loads the histogram list (bag of (visual) words) if it exitst, else it creates the bag of words.
        Every histogram should sum to 1, i.e. every entry represents the frequency of that particular descriptor
        :return: list of histograms, (bag of words)
        """
        if exists(f'{self.__models_path}/bag_of_words.joblib'):
            print("Histogram successfully loaded from file")
            return load(f'{self.__models_path}/bag_of_words.joblib')
        else:
            self.kmeans.verbose = False
            hist_list = []

            for image in self.grey_images:
                kp, des = self.sift.detectAndCompute(image, None)

                hist = np.zeros(self.k)
                nkp = np.size(kp)

                for d in des:
                    idx = self.kmeans.predict([d])
                    hist[idx] += 1 / nkp  # Because we need normalized histograms, I prefere to add 1/nkp directly

                hist_list.append(hist)
            dump(hist_list, f'{self.__models_path}/bag_of_words.joblib')
            print("Histogram successfully created")
        return hist_list


