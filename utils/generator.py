import numpy as np
import keras
import utils.image_transformations as imt


np.random.seed(448)


class ExtractsGenerator(keras.utils.Sequence):

    def __init__(self, dataset, x_shape, y_size, batch_size, shuffle=True, normalization=255,
                 task='classification', data_augmentation=False):
        """ Keras Data Generator
        # Arguments
            dataset: 2D array including x, y data. Having:
                x = path to an input image
                y = associated label (of size = y__size)
            x_shape: input image shape
            y_size: number of output nodes (classes)
            batch_size: size of each data batch to be generated
            shuffle: whether to shuffle the dataset in between different epochs
            normalization: normalizing the input RGB images.
                255: resulting RGB values between 0-1
                None: no normalization
            task: type of task at hand. one of 'classification' or 'regression'
            data_augmentation: whether any data augmentation was done. i.e. whether any transformations
                should be applied to the input images
        """
        self.dataset = dataset
        self.x_shape = x_shape
        self.y_size = y_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalization = normalization
        self.task = task
        self.data_augmentation = data_augmentation
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
            # r here contains [input numpy file path, binary y label]. In case data augmentation
            # was applied, r would contain: [input numpy file path, binary y label, (transformations)]

            # Reading input image (X)
            np_file = np.load(r[0])
            image = np_file[0]

            # Applying normalization
            if self.normalization is not None:
                image = image / self.normalization

            # Reading label (Y)
            if self.task == 'classification':
                # categorical labelling is as follows: [no seal, seal exists]
                if int(r[1]) == 1:
                    label = (0, 1)
                else:
                    label = (1, 0)
            elif self.task == 'regression':
                # in the case of regression, labels (counts) are recorded in each image's numpy file
                label = np_file[1]

            # Applying image transformations
            if self.data_augmentation:
                transformations = r[2]
                if transformations[imt.INDEX_FLIP_LR]:
                    image = imt.flip_left_to_right(image)
                if transformations[imt.INDEX_ROTATION_ANGLE] != 0:
                    image = imt.rotate_image(image, transformations[imt.INDEX_ROTATION_ANGLE])
                if transformations[imt.INDEX_BRIGHTNESS] != 0:
                    image = imt.apply_brightness(image, transformations[imt.INDEX_BRIGHTNESS])
                if transformations[imt.INDEX_CONTRAST] != 0:
                    image = imt.apply_contrast(image, transformations[imt.INDEX_CONTRAST])

            # Returning x, y batch arrays
            x[i, ] = image
            y[i, ] = label
        return x, y
