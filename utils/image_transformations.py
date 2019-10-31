import numpy as np
import cv2

INDEX_FLIP_LR = 0
INDEX_ROTATION_ANGLE = 1
INDEX_BRIGHTNESS = 2
INDEX_CONTRAST = 3

LIST_FLIP_LR_CHOICES = [True, False]
LIST_ROTATION_ANGLES = [0, 90, 180, 270]
LIST_BRIGHTNESS_LEVELS = [-127, 0, 64, 127]
LIST_CONTRAST_LEVELS = [-64, 0, 64]


def flip_left_to_right(img):
    return np.fliplr(img)


def rotate_image(img, angle):
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    transformed_image = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return transformed_image


def apply_brightness(input_img, brightness):
    if brightness > 0:
        shadow = brightness
        highlight = 255
    else:
        shadow = 0
        highlight = 255 + brightness
    alpha_b = (highlight - shadow)/255
    gamma_b = shadow
    transformed_image = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    return transformed_image


def apply_contrast(input_img, contrast):
    f = 131*(contrast + 127)/(127*(131-contrast))
    alpha_c = f
    gamma_c = 127*(1-f)
    transformed_image = cv2.addWeighted(input_img, alpha_c, input_img, 0, gamma_c)
    return transformed_image
