from __future__ import print_function
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from keras.callbacks import ReduceLROnPlateau
from resnet import ResNet
import numpy as np
import random
from tensorflow import set_random_seed
import csv
import math
from generator import ExtractsGenerator


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


data_path = '../nautillia_sea_lion_popcount/data/'  # TO BE SET
seed = 0
set_random_seed(seed)
random.seed(seed)

# Setting up data generators
with open('with_seals.csv', 'r') as f:
    reader = csv.reader(f)
    with_seals = list(reader)

with open('without_seals.csv', 'r') as f:
    reader = csv.reader(f)
    without_seals = list(reader)

dataset_size = 25000
data_variety_ratio = 0.6  # with seals / without seals
count_without_seals = math.floor(dataset_size * data_variety_ratio)
count_with_seals = math.floor(dataset_size * (1 - data_variety_ratio))
without_seals = without_seals[:count_without_seals]
with_seals = with_seals[:count_with_seals]

dataset = with_seals + without_seals
random.shuffle(dataset)

train_test_ratio = 0.9
validation_split = 0.2
train_size = math.floor(train_test_ratio * len(dataset))
validation_size = train_size * 0.2

partition = dict()
partition['train'] = dataset[:int(train_size-validation_size)]
partition['validation'] = dataset[int(train_size-validation_size):int(train_size)]
partition['test'] = dataset[int(train_size):]
print('Training: ' + str(len(partition['train'])))
print('Validation: ' + str(len(partition['validation'])))
print('Test: ' + str(len(partition['test'])))

batch_size = 10
generator_train = ExtractsGenerator(data_path=data_path, dataset=partition['train'], batch_size=batch_size)
generator_test = ExtractsGenerator(data_path=data_path, dataset=partition['test'], batch_size=batch_size)
generator_validation = ExtractsGenerator(data_path=data_path, dataset=partition['validation'], batch_size=batch_size)

# Setting model parameters
n = 6
depth = n * 9 + 2
model_type = 'ResNet%d' % depth
input_shape = (256, 256, 3)

total_items = len(generator_train)
epochs = 60
num_batches = int(total_items/batch_size)

# Prepare callbacks for model saving and for learning rate adjustment.
file_path = model_type + '-' + str(dataset_size) + '-{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, mode='min', save_best_only=True)
csv_logger = CSVLogger('training.log', separator=',')

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

# Create compile and train model
resnet = ResNet(input_shape=input_shape, n=n, output_nodes=5)
model = resnet.create_model()
callbacks = [csv_logger, checkpoint, lr_reducer, lr_scheduler]
loss = 'mean_squared_error'

print('Model: ' + model_type)
print('Dataset size: ' + str(dataset_size))
print('Epochs: ' + str(epochs))
print('loss: ' + loss)
model.summary()

model.compile(loss=loss, optimizer=Adam(lr=lr_schedule(0)), metrics=['mean_squared_logarithmic_error', 'mean_squared_error'])
history = model.fit_generator(generator=generator_train, steps_per_epoch=num_batches, epochs=epochs, verbose=1,
                              validation_data=generator_validation, callbacks=callbacks)
scores = model.evaluate_generator(generator=generator_test)