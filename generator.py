import numpy as np
import keras
import cv2
import skimage.feature
np.random.seed(448)


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


class ExtractsGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, data_path, dataset, batch_size, x_shape=(256, 256, 3), y_size=5, shuffle=True):
        """Initialization"""
        self.data_path = data_path
        self.x_shape = x_shape
        self.y_size = y_size
        self.batch_size = batch_size
        self.dataset = dataset
        self.shuffle = shuffle
        self.on_epoch_end()
        self.indices = None
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        rows = [self.dataset[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(rows)
        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, rows):
        """Generates data containing batch_size samples"""  # x : (n_samples, *dim, n_channels)

        # Initialization
        x = np.empty((self.batch_size, self.x_shape[0], self.x_shape[1], self.x_shape[2]))
        y = np.empty((self.batch_size, self.y_size))

        # Generate data
        for i, r in enumerate(rows):
            pass
            image_filename = r[0]
            image_index = int(r[1])
            image_scaling_factor = float(r[2])
            image_colormap = r[3]

            images, labels = get_data(self.data_path, image_filename, self.x_shape[1], image_scaling_factor)
            x[i, ] = images[image_index] / 255
            y[i] = labels[image_index]
        return x, y
