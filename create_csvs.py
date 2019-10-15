import numpy as np
import cv2
import skimage.feature
from os import listdir
import csv


# function from: https://www.kaggle.com/outrunner/use-keras-to-count-sea-lions/notebook
def get_data(dataset_path, filename, width, r):
    # read the Train and Train Dotted images
    image_1 = cv2.imread(dataset_path + "TrainDotted/" + filename)
    image_2 = cv2.imread(dataset_path + "Train/" + filename)
    img1 = cv2.GaussianBlur(image_1, (5, 5), 0)

    # absolute difference between Train and Train Dotted
    image_3 = cv2.absdiff(image_1, image_2)
    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    mask_1[mask_1 < 50] = 0
    mask_1[mask_1 > 0] = 255
    image_4 = cv2.bitwise_or(image_3, image_3, mask=mask_1)

    # convert to grayscale to be accepted by skimage.feature.blob_log
    image_6 = np.max(image_4, axis=2)

    # detect blobs
    blobs = skimage.feature.blob_log(image_6, min_sigma=3, max_sigma=7, num_sigma=1, threshold=0.05)

    h, w, d = image_2.shape

    res = np.zeros((int((w*r)//width)+1, int((h*r)//width)+1, 5), dtype='int16')

    for blob in blobs:
        # get the coordinates for each blob
        y, x, s = blob
        # get the color of the pixel from Train Dotted in the center of the blob
        b, g, R = img1[int(y)][int(x)][:]
        x1 = int((x*r)//width)
        y1 = int((y*r)//width)
        # decision tree to pick the class of the blob by looking at the color in Train Dotted
        if R > 225 and b < 25 and g < 25: # RED
            res[x1, y1, 0] += 1
        elif R > 225 and b > 225 and g < 25: # MAGENTA
            res[x1, y1, 1] += 1
        elif R < 75 and b < 50 and 150 < g < 200: # GREEN
            res[x1, y1, 4] += 1
        elif R < 75 and  150 < b < 200 and g < 75: # BLUE
            res[x1, y1, 3] += 1
        elif 60 < R < 120 and b < 50 and g < 75:  # BROWN
            res[x1, y1, 2] += 1

    ma = cv2.cvtColor((1*(np.sum(image_1, axis=2)>20)).astype('uint8'), cv2.COLOR_GRAY2BGR)
    img = cv2.resize(image_2 * ma, (int(w*r), int(h*r)))
    h1, w1, d = img.shape

    trainX = []
    trainY = []

    for i in range(int(w1//width)):
        for j in range(int(h1//width)):
            trainY.append(res[i, j, :])
            trainX.append(img[j*width:j*width+width, i*width:i*width+width, :])

    return np.array(trainX), np.array(trainY)


r = 0.6            # scale factor
width = 256        # patch size
dataset_path = '../nautillia_sea_lion_popcount/data/'  # TO BE SET

# HEADER:: image_filename, index, r, colormap
csv_with_seals = []
csv_without_seals = []

for f in listdir(dataset_path + 'Train/'):
    print(f)
    if f.endswith('.jpg'):
        images, labels = get_data(dataset_path, f, width, r)
        for i, l in enumerate(labels):
            if np.array_equal(l, np.array([0, 0, 0, 0, 0])):
                csv_without_seals.append((f, i, r, 'raw'))
            else:
                csv_with_seals.append((f, i, r, 'raw'))

print(len(csv_with_seals))
print(len(csv_without_seals))

with open('with_seals.csv', "w") as f:
    writer = csv.writer(f)
    writer.writerows(csv_with_seals)

with open('without_seals.csv', "w") as f:
    writer = csv.writer(f)
    writer.writerows(csv_without_seals)