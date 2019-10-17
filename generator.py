import numpy as np
import keras
import cv2

np.random.seed(448)


class ExtractsGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, data_path, dataset, batch_size=1, x_shape=(256, 256, 3), y_size=5, shuffle=True):
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
            image_id = r[0]
            image_filename = str(image_id) + '.jpg'
            image_tile_indices = int(r[1])
            image_scaling_factor = float(r[2])
            image_colormap = r[3]
            image_tile_labels = r[4]

            # read and rescale fullsize image
            full_image = cv2.imread(self.data_path + "Train/" + image_filename)
            h, w, d = full_image.shape
            full_image = cv2.resize(full_image, (int(w * image_scaling_factor), int(h * image_scaling_factor)))

            # extract image tile
            i, j = image_tile_indices
            width = self.x_shape[0]
            image_tile = full_image[j * width:j * width + width, i * width:i * width + width, :]

            # TODO: ADD "image_colormap" TRANSFORMATIONS

            x[i, ] = image_tile / 255
            y[i] = image_tile_labels
        return x, y
