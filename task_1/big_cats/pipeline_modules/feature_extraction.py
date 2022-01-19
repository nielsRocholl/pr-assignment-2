import sys
from tqdm import tqdm
from joblib import dump, load
from os.path import exists
import cv2
import os


class FeatureExtractor:
    def __init__(self):
        # paths:
        self.__path_data = f'data{os.sep}BigCats{os.sep}'
        self.__model_path = 'big_cats/models'

        # create SIFT feature extractor
        self.sift = cv2.SIFT_create()

        # data:
        self.original_dataset = self.__load_dataset()
        self.grey_dataset, _ = self.__convert_to_greyscale()
        _, self.grey_dataset_flat_list = self.__convert_to_greyscale()
        self.num_of_samples = len(self.grey_dataset_flat_list)
        self.descriptor_list = self.__extract_keypoints()

        # self.processed_dataset = self.__extract_features()

    def __load_dataset(self):
        """
        load original dataset from data/BigCats
        :return: a dictionary holding the entire raw dataset: {'Lion: [img_0,...,img_x], ...}
        """
        os.chdir(f'{os.path.split(__file__)[0]}{os.sep}..{os.sep}..')
        # retrieve class labels (and clean the list)
        classes = list(filter(None, [x[0].replace(f'data{os.sep}BigCats{os.sep}', '') for x in os.walk(self.__path_data)]))
        if not classes:
            sys.exit("Data could not be loaded, most likely a path mismatch")

        dataset = {}
        for c in classes:
            filenames = next(os.walk(f'{self.__path_data}{os.sep}{c}'), (None, None, []))[2]
            dataset[c] = [cv2.imread(f'{self.__path_data}{os.sep}{c}{os.sep}{img}') for img in filenames]
        print("Dataset successfully loaded")
        return dataset

    def __convert_to_greyscale(self):
        """
        Converts all images in the raw dataset into greyscale, since sift operates on greyscale
        :return: dictionary holding the transformed greyscale dataset
        """

        grey_images = {}
        grey_images_flat_list = []
        for class_label, c in enumerate(self.original_dataset):
            grey_images[c] = [[cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), class_label] for image in self.original_dataset[c]]
            grey_images_flat_list.append(grey_images[c])
        grey_images_flat_list = [item for sublist in grey_images_flat_list for item in sublist]

        return grey_images, grey_images_flat_list

    def __extract_keypoints(self):
        """
        extracts the keypoints and descriptors from the greyscale images using sift
        :return: a list containing all sift descriptors
        """
        if exists(f'{self.__model_path}/sift_keypoints.joblist'):
            print('Feature list successfully loaded from file')
            return load(f'{self.__model_path}/sift_keypoints.joblist')
        else:
            descriptor_list = []
            for image, label in tqdm(self.grey_dataset_flat_list):
                kp, des = self.sift.detectAndCompute(image, None)

                for d in des:
                    descriptor_list.append(d)
        if not os.path.exists(f'{self.__model_path}'):
            # Create a new directory because it does not exist
            os.makedirs(f'{self.__model_path}')
        dump(descriptor_list, f'{self.__model_path}/sift_keypoints.joblist')
        print('Feature list successfully created')
        return descriptor_list


    # def __extract_features(self):
    #     processed_dataset = {}
    #     # detect features from the image
    #     print("Extracting SIFT Features:")
    #     for c in tqdm(self.grey_dataset):
    #         image_list = []
    #         for idx, image in enumerate(self.grey_dataset[c]):
    #             keypoints, descriptors = self.sift.detectAndCompute(image, None)
    #             image_list.append([keypoints, descriptors])
    #         processed_dataset[c] = image_list
    #     print("Features successfully extracted")
    #     return processed_dataset

