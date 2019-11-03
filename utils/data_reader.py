import csv
import random
import math
import utils.image_transformations as imt
import itertools
import copy


def __augment_list(lst):
    """ Prepares for data augmentation by defining all possible combinations image transformations for each image tile
    # Arguments
        lst: original (CSV) list to be augmented

        returns: CSV list with each row having a unique (image, image transformations to be applied)
    """
    augmented = []
    # get all possible combinations of transformations
    transformation_combinations = list(itertools.product(*[
        imt.LIST_FLIP_LR_CHOICES,
        imt.LIST_ROTATION_ANGLES,
        imt.LIST_BRIGHTNESS_LEVELS,
        imt.LIST_CONTRAST_LEVELS
    ]))
    # record each image tile row with all possible transformations
    for r in lst:
        # tuple indices defined in image_transformations.py :: (flip_lr, rotation, brightness, contrast)
        # tuple used in generator.py since 'r' here only includes a path to .npy file and its corresponding label
        for tc in transformation_combinations:
            nr = copy.deepcopy(r)
            nr.append((
                tc[imt.INDEX_FLIP_LR],
                tc[imt.INDEX_ROTATION_ANGLE],
                tc[imt.INDEX_BRIGHTNESS],
                tc[imt.INDEX_CONTRAST]
            ))
            augmented.append(nr)
    return augmented


def get_data_partitions(seed, data_path, base_dataset_size, data_augmentation=False, augmented_dataset_size=None,
                        percentage_without_seals=0.6, train_test_ratio=0.9, validation_percentage=0.2):
    """ params explanation
        data_augmentation = apply transformations. Results in a dataset larger than base_dataset_size
        percentage_without_seals = without_seals/total. All with seals :set: percentage_without_seals = 0
        augmented_dataset_size = dataset size after augmentation. None: use full augmented dataset
    """
    """ Prepares data partitions for data generators to be used in training/validation/testing
    # Arguments
        seed: seed for random shuffling
        data_path: path to the dataset
        base_dataset_size: including images of both types (seals & no_seals) BEFORE augmentation 
        data_augmentation: whether image transformations will be applied to the raw image tiles:
            default: False 
        augmented_dataset_size: including images of both types (seals & no_seals) AFTER augmentation
            default: None (i.e. no data augmentation)
        percentage_without_seals: percentage of images having no seal lions out of the total dataset
            default: 0.6
        train_test_ratio: percentage of data to be used to training/validation
            default: 0.9 (i.e. use 10% for testing)
        validation_percentage: percentage of TRAINING data to be used for validation
            default: 0.2 (i.e.: training set includes 0.8 for training, 0.2 for validation) 
    """

    random.seed(seed)

    # Reading original CSVs, reducing each type of images according to the dataset size and ratio parameters
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
            random.shuffle(without_seals)  # shuffling before reducing the set, to avoid having too many similar images
            count_without_seals_augmented = math.floor(augmented_dataset_size * percentage_without_seals)
            without_seals = without_seals[:count_without_seals_augmented]
        else:
            random.shuffle(without_seals)
            count_without_seals = math.floor(base_dataset_size * percentage_without_seals)
            without_seals = without_seals[:count_without_seals]

    # creating and shuffling the dataset containing both types of image tiles
    dataset = with_seals + without_seals
    random.shuffle(dataset)

    # calculating the training/validation/testing data splits
    train_size = math.floor(train_test_ratio * len(dataset))
    validation_size = train_size * validation_percentage

    # creating and returning the different data partitions
    partitions = dict()
    partitions['train'] = dataset[:int(train_size - validation_size)]
    partitions['validation'] = dataset[int(train_size - validation_size):int(train_size)]
    partitions['test'] = dataset[int(train_size):]

    return partitions
