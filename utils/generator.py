import numpy as np
import keras

np.random.seed(448)


class ExtractsGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, dataset, x_shape, y_size, batch_size, shuffle=True, normalization=255):
        """Initialization"""
        self.dataset = dataset
        self.x_shape = x_shape
        self.y_size = y_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalization = normalization
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
            image = np.load(r[0])[0]
            if self.normalization is not None:
                image = image / self.normalization
            # label: [no seal, seal exists]
            if int(r[1]) == 1:
                label = (0, 1)
            else:
                label = (1, 0)
            y[i] = label
        return x, y
