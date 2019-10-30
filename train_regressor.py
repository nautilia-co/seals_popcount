from __future__ import print_function
import keras
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from keras.callbacks import ReduceLROnPlateau
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
data_path = '/projets/reva/dwilson/mn_sea_lion_popcount/dataset/processed3/'  # TO BE SET -- ending with /
output_dir = ''  # TO BE SET :: empty --> current directory || ending with /
seed = 0
set_random_seed(seed)
random.seed(seed)
dataset_size = 8000
n = 6
input_shape = (256, 256, 3)
output_nodes = 5
epochs = 60
batch_size = 16

# Prepare output directory
if output_dir != '':
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

# Create data generators
partitions = get_data_partitions(seed=seed, data_path=data_path, dataset_size=dataset_size, data_variety_ratio=0)
print('Training: ' + str(len(partitions['train'])))
print('Validation: ' + str(len(partitions['validation'])))
print('Test: ' + str(len(partitions['test'])))

generator_train = ExtractsGenerator(dataset=partitions['train'], batch_size=batch_size, x_shape=input_shape,
                                    y_size=output_nodes, normalization=None, type='regression')
generator_test = ExtractsGenerator(dataset=partitions['test'], batch_size=batch_size, x_shape=input_shape,
                                   y_size=output_nodes, normalization=None, type='regression')
generator_validation = ExtractsGenerator(dataset=partitions['validation'], batch_size=batch_size, x_shape=input_shape,
                                         y_size=output_nodes, normalization=None, type='regression')

# Create model
resnet = ResNet(final_activation='sigmoid', n=n,
                input_shape=input_shape, output_nodes=output_nodes)
model = resnet.create_model()

# model = keras.applications.vgg19.VGG19(weights=None, input_shape=input_shape,
#                                        include_top=True, pooling=None, classes=2)

# model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights=None,
#                                                                  input_shape=input_shape, pooling=None, classes=2)
loss = 'mean_squared_error'

print('Model: ' + model.name)
print('Dataset size: ' + str(dataset_size))
print('Epochs: ' + str(epochs))
print('loss: ' + loss)
model.summary()

# Prepare model callbacks
file_path = output_dir + model.name + '_' + str(dataset_size) + '_{epoch:02d}_{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, mode='min', save_best_only=True)
csv_logger = CSVLogger(output_dir + 'training.log', separator=',')
lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

callbacks = [csv_logger, checkpoint, lr_reducer, lr_scheduler]

steps_per_epoch = int(len(partitions['train']) / batch_size)
# opt = SGD()
# opt = Adam()
opt = Adam(lr=lr_schedule(0))
model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
history = model.fit_generator(generator=generator_train, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1,
                              validation_data=generator_validation, callbacks=callbacks)
scores = model.evaluate_generator(generator=generator_test)
print(scores)
