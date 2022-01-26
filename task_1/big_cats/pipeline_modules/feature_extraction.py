import random
import sys
import numpy as np

from tqdm import tqdm
import cv2
import os

from skimage.util import random_noise
from skimage import img_as_ubyte


def __augment(img):
    options = [
        'gaussian',
        'poisson',
        's&p',
    ]
    option = random.choice(options)
    noise_img = (random_noise(img, mode=option))
    noise_img = img_as_ubyte(noise_img)
    return np.array(noise_img)


def load_dataset(augment=False):
    """
    load original dataset from data/BigCats
    :return: a list containing the bag of words model and the associated labels
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
        # Mac os creates this file.
        filenames.remove('.DS_Store')
        for img in filenames:
            image = cv2.imread(f'{path_data}{os.sep}{c}{os.sep}{img}')
            dataset.append([image, classes.index(c)])
            # augment data
            if random.random() < 0.2 and augment:
                dataset.append([__augment(image), classes.index(c)])
    # turn into greyscale:
    dataset = __convert_to_greyscale(dataset)
    # load features
    dataset = __extract_features(dataset)

    return dataset


def __convert_to_greyscale(dataset):
    """
    Converts all images in the raw dataset into greyscale, since sift operates on greyscale
    :return: a list containing all greyscale images and the associated labels
    """
    greyscale_dataset = []
    for image, label in dataset:
        try:
            greyscale_dataset.append([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), label])
        except Exception as e:
            print(e)

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
        des = random.sample(list(des), 10)
        feature_dataset.append([des, 10, label])

    return feature_dataset
