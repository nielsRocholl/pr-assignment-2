import os

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# find data
import pandas as pd
#
img = f'..{os.sep}data{os.sep}BigCats{os.sep}Cheetah{os.sep}animal-africa-wilderness-zoo.jpg'
img = cv.imread(img)

# convert to greyscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# apply sift
sift = cv.SIFT_create()
kp = sift.detect(gray, None)

# look at data
img = cv.drawKeypoints(gray, kp, img)
cv.imwrite(f'..{os.sep}results{os.sep}sift_keypoints.jpg', img)
img = cv.drawKeypoints(gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite(f'..{os.sep}results{os.sep}sift_keypoints+orientation.jpg', img)


# show data distribution
path_data = f'..{os.sep}data{os.sep}BigCats{os.sep}'
# retrieve class labels (and clean the list)
classes = list(filter(None, [x[0].replace(f'..{os.sep}data{os.sep}BigCats{os.sep}', '') for x in os.walk(path_data)]))
dataset = np.zeros(len(classes))
for idx, c in enumerate(classes):
    filenames = next(os.walk(f'{path_data}{os.sep}{c}'), (None, None, []))[2]
    dataset[idx] = len(filenames)
print(dataset)

plt.grid()
plt.bar(classes, dataset)
plt.xlabel('Class')
plt.ylabel('Samples')
plt.title('Data Distribution')
plt.savefig(f'..{os.sep}results{os.sep}data_dist.png')
plt.show()


df = pd.read_csv(f'..{os.sep}results{os.sep}big_cats_accuracy.csv')
df2 = pd.read_csv(f'..{os.sep}results{os.sep}big_cats_accuracy_augmented.csv')



print(df.mean())
print(df2.mean())


