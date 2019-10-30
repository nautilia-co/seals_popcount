import csv
import random
import math


def get_data_partitions(seed, data_path, dataset_size, data_variety_ratio=0.6, train_test_ratio=0.9):
    random.seed(seed)
    count_without_seals = math.floor(dataset_size * data_variety_ratio)
    count_with_seals = math.floor(dataset_size * (1 - data_variety_ratio))
    with open(data_path + 'data_with_seals.csv', 'r') as f:
        reader = csv.reader(f)
        with_seals = list(reader)
        random.shuffle(with_seals)
        with_seals = with_seals[:count_with_seals]

    with open(data_path + 'data_without_seals.csv', 'r') as f:
        reader = csv.reader(f)
        without_seals = list(reader)
        random.shuffle(without_seals)
        without_seals = without_seals[:count_without_seals]

    dataset = with_seals + without_seals
    random.shuffle(dataset)

    train_size = math.floor(train_test_ratio * len(dataset))
    validation_size = train_size * 0.2

    partitions = dict()
    partitions['train'] = dataset[:int(train_size - validation_size)]
    partitions['validation'] = dataset[int(train_size - validation_size):int(train_size)]
    partitions['test'] = dataset[int(train_size):]

    return partitions
