from __future__ import print_function
import keras
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Flatten, Dense
from keras.models import Model
import numpy as np
import random
from tensorflow import set_random_seed
from utils.generator import ExtractsGenerator
from utils.data_reader import get_data_partitions
import os
from models.resnet import ResNet


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 100:
        lr *= 0.5e-3
    elif epoch > 70:
        lr *= 1e-3
    elif epoch > 50:
        lr *= 1e-2
    elif epoch > 30:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


# Setting model parameters
data_path = '/projets/reva/dwilson/mn_sea_lion_popcount/dataset/processed_with_both_labels/'  # TO BE SET -- end with /
output_dir = ''  # TO BE SET -- leave empty = current directory |OR| 'dir/path/'
seed = 0  # Seed for random shuffling
set_random_seed(seed)
random.seed(seed)
dataset_size = 8000  # Including images of both types (seals & no_seals)
percentage_without_seals = 0  # Percentage of images of class no seals. Set to 0 = all images have seals
augment_dataset = True  # Whether to apply data augmentation
dataset_size_after_augmentation = 20000  # Including both classes, used when augment_dataset = True
n = 6  # ResNet depth parameter
input_shape = (256, 256, 3)  # Input image shape
output_nodes = 5  # Number of output nodes (classes)
epochs = 60
batch_size = 16
loss = 'mean_squared_error'  # Loss function name

# Prepare output directory
if output_dir != '':
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

# Create data generators
partitions = get_data_partitions(seed=seed, data_path=data_path, base_dataset_size=dataset_size,
                                 percentage_without_seals=percentage_without_seals,
                                 data_augmentation=augment_dataset,
                                 augmented_dataset_size=dataset_size_after_augmentation)
print('Training: ' + str(len(partitions['train'])))
print('Validation: ' + str(len(partitions['validation'])))
print('Test: ' + str(len(partitions['test'])))

generator_train = ExtractsGenerator(dataset=partitions['train'], batch_size=batch_size, x_shape=input_shape,
                                    y_size=output_nodes, normalization=None, task='regression',
                                    data_augmentation=augment_dataset)
generator_test = ExtractsGenerator(dataset=partitions['test'], batch_size=batch_size, x_shape=input_shape,
                                   y_size=output_nodes, normalization=None, task='regression',
                                   data_augmentation=augment_dataset)
generator_validation = ExtractsGenerator(dataset=partitions['validation'], batch_size=batch_size, x_shape=input_shape,
                                         y_size=output_nodes, normalization=None, task='regression',
                                         data_augmentation=augment_dataset)

# Create ResNet
# resnet = ResNet(final_activation='sigmoid', n=n,
#                 input_shape=input_shape, output_nodes=output_nodes)
# model = resnet.create_model()

# Create VGG19
# model = keras.applications.vgg19.VGG19(weights=None, input_shape=input_shape,
#                                        include_top=True, pooling=None, classes=2)

# Create InceptionResNetV2
model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights=None,
                                                                 input_shape=input_shape, pooling=None, classes=2)
# Adding regression layers to default InceptionResNetV2
x = Flatten()(model.output)
x = Dense(output_nodes, activation='sigmoid')(x)
model = Model(inputs=model.inputs, outputs=x, name=model.name)

# Prepare model callbacks
file_path = output_dir + model.name + '_' + str(dataset_size) + '_{epoch:02d}_{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, mode='min', save_best_only=True)
csv_logger = CSVLogger(output_dir + 'training.log', separator=',')
lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
callbacks = [csv_logger, checkpoint, lr_reducer, lr_scheduler]

# Prepare training optimizer
opt = Adam(lr=lr_schedule(0))

print('Model: ' + model.name)
print('Dataset size: ' + str(dataset_size))
print('Epochs: ' + str(epochs))
print('Batch size: ' + str(batch_size))
print('loss: ' + loss)
model.summary()

# Start training
steps_per_epoch = int(len(partitions['train']) / batch_size)
model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
history = model.fit_generator(generator=generator_train, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1,
                              validation_data=generator_validation, callbacks=callbacks)

# Evaluate trained model
scores = model.evaluate_generator(generator=generator_test)
print(scores)
