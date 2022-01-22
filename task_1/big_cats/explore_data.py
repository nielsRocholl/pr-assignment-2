import os

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# find data
import pandas as pd

os.chdir(f'{os.path.split(__file__)[0]}{os.sep}..')
img = f'data{os.sep}BigCats{os.sep}/Cheetah/animal-africa-wilderness-zoo.jpg'
img = cv.imread(img)

# convert to greyscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# apply sift
sift = cv.SIFT_create()
kp = sift.detect(gray, None)

# look at data
img = cv.drawKeypoints(gray, kp, img)
cv.imwrite('results/sift_keypoints.jpg', img)
img = cv.drawKeypoints(gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite('results/sift_keypoints+orientation.jpg', img)


# show data distribution
path_data = f'data{os.sep}BigCats{os.sep}'
# retrieve class labels (and clean the list)
classes = list(filter(None, [x[0].replace(f'data{os.sep}BigCats{os.sep}', '') for x in os.walk(path_data)]))
dataset = np.zeros(len(classes))
for idx, c in enumerate(classes):
    filenames = next(os.walk(f'{path_data}{os.sep}{c}'), (None, None, []))[2]
    dataset[idx] = len(filenames)

plt.grid()
plt.bar(classes, dataset)
plt.xlabel('Class')
plt.ylabel('Samples')
plt.title('Data Distribution')
plt.savefig('results/data_dist.png')
plt.show()

df = pd.read_csv('results/big_cats_results_kmeans_capped.csv')
print(df.mean())


