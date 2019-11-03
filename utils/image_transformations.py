import numpy as np
import cv2


# Index of each image transformation parameter, used when creating the data partitions, and generating data
INDEX_FLIP_LR = 0
INDEX_ROTATION_ANGLE = 1
INDEX_BRIGHTNESS = 2
INDEX_CONTRAST = 3

LIST_FLIP_LR_CHOICES = [True, False]  # whether to flip the image in the left/right direction
LIST_ROTATION_ANGLES = [0, 90, 180, 270]  # possible angles of rotating the image around its center
LIST_BRIGHTNESS_LEVELS = [-127, 0, 64, 127]  # possible brightness values
LIST_CONTRAST_LEVELS = [-64, 0, 64]  # possible contrast values


def flip_left_to_right(img):
    """ Flip an image in the left/right direction
    # Arguments
        img: image to be flipped (np array)

        returns: flipped image
    """
    return np.fliplr(img)


def rotate_image(img, angle):
    """ Rotate an image around its center
    # Arguments
        img: image to be rotated  (np array)
        angle: angle of rotation

        returns: rotated image
    """
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    transformed_image = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return transformed_image


def apply_brightness(img, brightness):
    """ Change the brightness of an image
    # Arguments
        img: image to be transformed  (np array)
        angle: brightness level

        returns: transformed image
    """
    if brightness > 0:
        shadow = brightness
        highlight = 255
    else:
        shadow = 0
        highlight = 255 + brightness
    alpha_b = (highlight - shadow)/255
    gamma_b = shadow
    transformed_image = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)
    return transformed_image


def apply_contrast(img, contrast):
    """ Change the contrast of an image
    # Arguments
        img: image to be transformed  (np array)
        angle: contrast level

        returns: transformed image
    """
    f = 131*(contrast + 127)/(127*(131-contrast))
    alpha_c = f
    gamma_c = 127*(1-f)
    transformed_image = cv2.addWeighted(img, alpha_c, img, 0, gamma_c)
    return transformed_image
