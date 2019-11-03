import numpy as np
import cv2
import skimage.feature
from os import listdir
import csv


def get_data(dataset_path, filename, width, r):
    """ Splits an original training image into tiles, conserving each tile's label (Y) information
        Credits to Kaggle user 'outrunner': https://www.kaggle.com/outrunner/use-keras-to-count-sea-lions/notebook
        The original function was modified to conserve and return the indices of the output tiles
    # Arguments
        dataset_path: path to the dataset
        filename: name of the original (Train) image
        width: dimension of output square tiles
        r: scaling factor of the original image

        returns:
            trainX = np array of extracted image tiles
            trainY = np array of counts of each type of sea lions corresponding to each extracted tile
            indices = np array of extracted tile indices according to the original images after scaling (r)
    """

    # Read the Train and Train Dotted images
    image_1 = cv2.imread(dataset_path + "TrainDotted/" + filename)
    image_2 = cv2.imread(dataset_path + "Train/" + filename)
    img1 = cv2.GaussianBlur(image_1, (5, 5), 0)

    # Absolute difference between Train and Train Dotted
    image_3 = cv2.absdiff(image_1, image_2)
    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    mask_1[mask_1 < 50] = 0
    mask_1[mask_1 > 0] = 255
    image_4 = cv2.bitwise_or(image_3, image_3, mask=mask_1)

    # Convert to grayscale to be accepted by skimage.feature.blob_log
    image_6 = np.max(image_4, axis=2)

    # detect blobs
    blobs = skimage.feature.blob_log(image_6, min_sigma=3, max_sigma=7, num_sigma=1, threshold=0.05)

    h, w, d = image_2.shape

    res = np.zeros((int((w * r) // width) + 1, int((h * r) // width) + 1, 5), dtype='int16')

    for blob in blobs:
        # get the coordinates for each blob
        y, x, s = blob
        # get the color of the pixel from Train Dotted in the center of the blob
        b, g, R = img1[int(y)][int(x)][:]
        x1 = int((x * r) // width)
        y1 = int((y * r) // width)
        # decision tree to pick the class of the blob by looking at the color in Train Dotted
        if R > 225 and b < 25 and g < 25:  # RED
            res[x1, y1, 0] += 1
        elif R > 225 and b > 225 and g < 25:  # MAGENTA
            res[x1, y1, 1] += 1
        elif R < 75 and b < 50 and 150 < g < 200:  # GREEN
            res[x1, y1, 4] += 1
        elif R < 75 and 150 < b < 200 and g < 75:  # BLUE
            res[x1, y1, 3] += 1
        elif 60 < R < 120 and b < 50 and g < 75:  # BROWN
            res[x1, y1, 2] += 1

    ma = cv2.cvtColor((1 * (np.sum(image_1, axis=2) > 20)).astype('uint8'), cv2.COLOR_GRAY2BGR)
    img = cv2.resize(image_2 * ma, (int(w * r), int(h * r)))
    h1, w1, d = img.shape

    trainX = []
    trainY = []
    indices = []

    for i in range(int(w1 // width)):
        for j in range(int(h1 // width)):
            trainY.append(res[i, j, :])
            indices.append((i, j))

            raw_image = cv2.imread(dataset_path + "Train/" + filename)
            h, w, d = raw_image.shape
            raw_image = cv2.resize(raw_image, (int(w * r), int(h * r)))
            tile = raw_image[j * width:j * width + width, i * width:i * width + width, :]
            trainX.append(tile)

    return np.array(trainX), np.array(trainY), np.array(indices)


r = 0.6  # Scaling factor
width = 256  # Output tile dimension
dataset_path = '/projets/reva/dwilson/mn_sea_lion_popcount/dataset/'  # TO BE SET

data_with_seals = []
data_without_seals = []

# Loop through the 'Train' directory of the dataset, extract image tiles of each original image
# Resulting np files include the image + regression labels (count of different types of seal lions)
# Resulting CSV files include the path to each image np file + a single binary label
for f in listdir(dataset_path + 'Train/'):
    with open('test.log', 'a') as log:
        log.write(f + '\n')
    print(f)
    if f.endswith('.jpg'):
        images, labels, indices = get_data(dataset_path, f, width, r)
        for i, l in enumerate(labels):
            image_id = int(f.replace('.jpg', ''))
            if np.array_equal(l, np.array([0, 0, 0, 0, 0])):  # If no seals
                out_path = dataset_path + 'processed_with_both_labels/without_seals/'
                # Appending 'r' and indices to the filename so each tile has a unique identifier
                np_filename = str(image_id) + '__r_' + str(r) + '__i_' + str(indices[i][0]) \
                              + '__j_' + str(indices[i][1])
                x = out_path + np_filename + '.npy'
                y = 0
                data_without_seals.append((x, y))   # for CSV: (path, binary label)
            else:
                out_path = dataset_path + 'processed_with_both_labels/with_seals/'
                np_filename = str(image_id) + '__r_' + str(r) + '__i_' + str(indices[i][0]) \
                              + '__j_' + str(indices[i][1])
                x = out_path + np_filename + '.npy'
                y = 1
                data_with_seals.append((x, y))
            np.save(x, np.array([images[i], l]))  # save np file: (image, regression label)

# Writing CSV files
with open(dataset_path + 'processed_with_both_labels/data_with_seals.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(data_with_seals)

with open(dataset_path + 'processed_with_both_labels/data_without_seals.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(data_without_seals)
