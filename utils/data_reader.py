import csv
import random
import math
import utils.image_transformations as imt
import itertools


def __augment_list(lst):
    augmented = []
    transformation_combinations = list(itertools.product(*[
        imt.LIST_FLIP_LR_CHOICES,
        imt.LIST_ROTATION_ANGLES,
        imt.LIST_BRIGHTNESS_LEVELS,
        imt.LIST_CONTRAST_LEVELS
    ]))
    for r in lst:
        # tuple indices defined in image_transformations.py :: (flip_lr, rotation, brightness, contrast)
        # tuple used in generator.py since 'r' here only includes a path to .npy file and its corresponding label
        for tc in transformation_combinations:
            augmented.append(r.append((
                tc[imt.INDEX_FLIP_LR],
                tc[imt.INDEX_ROTATION_ANGLE],
                tc[imt.INDEX_BRIGHTNESS],
                tc[imt.INDEX_CONTRAST]
            )))
    return augmented


def get_data_partitions(seed, data_path, base_dataset_size, augmented_dataset_size=None, percentage_without_seals=0.6,
                        train_test_ratio=0.9, data_augmentation=False):
    """ params explanation
        data_augmentation = apply transformations. Results in a dataset larger than base_dataset_size
        percentage_without_seals = without_seals/total. All with seals :set: percentage_without_seals = 0
        augmented_dataset_size = dataset size after augmentation. None: use full augmented dataset
    """
    random.seed(seed)

    with open(data_path + 'data_with_seals.csv', 'r') as f:
        reader = csv.reader(f)
        with_seals = list(reader)
        if data_augmentation:
            with_seals = __augment_list(with_seals)
            random.shuffle(with_seals)
            count_with_seals_augmented = math.floor(augmented_dataset_size * (1 - percentage_without_seals))
            with_seals = with_seals[:count_with_seals_augmented]
        else:
            random.shuffle(with_seals)
            count_with_seals = math.floor(base_dataset_size * (1 - percentage_without_seals))
            with_seals = with_seals[:count_with_seals]

    with open(data_path + 'data_without_seals.csv', 'r') as f:
        reader = csv.reader(f)
        without_seals = list(reader)
        if data_augmentation:
            without_seals = __augment_list(without_seals)
            random.shuffle(without_seals)
            count_without_seals_augmented = math.floor(augmented_dataset_size * percentage_without_seals)
            without_seals = without_seals[:count_without_seals_augmented]
        else:
            random.shuffle(without_seals)
            count_without_seals = math.floor(base_dataset_size * percentage_without_seals)
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
