import cv2
import os
from tqdm import tqdm


class FeatureExtractor:
    def __init__(self):
        self.__path_data = 'datagit BigCats/'
        # create SIFT feature extractor
        self.__sift = cv2.SIFT_create()
        self.original_dataset = self.__load_dataset()
        self.grey_dataset = self.__convert_to_greyscale()
        self.processed_dataset = self.__extract_features()

    def __load_dataset(self):
        os.chdir(f'{os.path.split(__file__)[0]}/../..')
        # retrieve class labels (and clean the list)
        classes = list(filter(None, [x[0].replace('data/BigCats/', '') for x in os.walk(self.__path_data)]))

        dataset = {}
        for c in classes:
            filenames = next(os.walk(f'{self.__path_data}/{c}'), (None, None, []))[2]
            dataset[c] = [cv2.imread(f'{self.__path_data}/{c}/{img}') for img in filenames]
        print("Dataset successfully loaded")
        return dataset

    def __extract_features(self):
        processed_dataset = {}
        image_list = []
        # detect features from the image
        print("Extracting SIFT Features:")
        for c in tqdm(self.grey_dataset):
            for idx, image in enumerate(self.grey_dataset[c]):
                keypoints, descriptors = self.__sift.detectAndCompute(image, None)
                image_list.append([keypoints, descriptors])
            processed_dataset[c] = image_list
        print("Features successfully extracted")
        return processed_dataset

    def __convert_to_greyscale(self):
        grey_images = {}
        for c in self.original_dataset:
            grey_images[c] = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in self.original_dataset[c]]
        return grey_images


f = FeatureExtractor()
