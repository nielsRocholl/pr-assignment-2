import cv2
sift = cv2.SIFT_create()

def feature_extraction(dataset, class_name=""):
    """
    extracts the keypoints and descriptors from the greyscale images using sift
    :return: a list containing all sift descriptors
    """
    if class_name != "":
        print(f"Extracting features for {class_name}")

    feature_dataset = []
    for image in dataset:
        _, des = sift.detectAndCompute(image, None)
        feature_dataset.append(des)

    if class_name != "":
        print(f'Feature list for {class_name} successfully created')

    return feature_dataset