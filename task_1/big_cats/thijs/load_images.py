import os
import cv2


def load_images(folder, greyscale=False):
    """
    Loads all images from a folder into a list
    :param folder: the folder to load the images from
    :param extension: the extension of the images to load
    :return: a list containing all images
    """
    # print(f"Loading images in {folder}")
    images = []
    for filename in os.listdir(folder):
        image = cv2.imread(os.path.join(folder, filename))
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images.append(grey_image if greyscale else image)
    # print(f"\rLoaded {len(images)} images")
    return images