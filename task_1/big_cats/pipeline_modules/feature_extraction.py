import random
import sys

from tqdm import tqdm
import cv2
import os




def load_dataset():
    """
    load original dataset from data/BigCats
    :return: a dictionary holding the entire raw dataset: {'Lion: [img_0,...,img_x], ...}
    """
    path_data = f'data{os.sep}BigCats{os.sep}'
    os.chdir(f'{os.path.split(__file__)[0]}{os.sep}..{os.sep}..')
    # retrieve class labels (and clean the list)
    classes = list(filter(None, [x[0].replace(f'data{os.sep}BigCats{os.sep}', '') for x in os.walk(path_data)]))
    if not classes:
        sys.exit("Data could not be loaded, most likely a path mismatch")

    dataset = []
    for c in classes:
        filenames = next(os.walk(f'{path_data}{os.sep}{c}'), (None, None, []))[2]
        dataset += [[cv2.imread(f'{path_data}{os.sep}{c}{os.sep}{img}'), classes.index(c)] for img in filenames]

    # turn into greyscale:
    dataset = __convert_to_greyscale(dataset)
    # load features
    dataset = __extract_features(dataset)
    return dataset


def __convert_to_greyscale(dataset):
    """
    Converts all images in the raw dataset into greyscale, since sift operates on greyscale
    :return: dictionary holding the transformed greyscale dataset
    """
    greyscale_dataset = []
    for image, label in dataset:
        try:
            greyscale_dataset.append([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), label])
        except Exception as e:
            print('')
    return greyscale_dataset


def __extract_features(dataset):
    """
    extracts the keypoints and descriptors from the greyscale images using sift
    :return: a list containing all sift descriptors
    """
    sift = cv2.SIFT_create()
    feature_dataset = []
    print('Loading data:')
    for image, label in tqdm(dataset):
        kp, des = sift.detectAndCompute(image, None)
        # des = random.sample(list(des), 245)
        feature_dataset.append([des, kp, label])

    return feature_dataset
