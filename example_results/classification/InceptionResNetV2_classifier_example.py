import numpy as np
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import os

# Following line resolves the error (OMP: Error #15: Initializing libiomp5.dylib)
# Error could be limited to Mac OS, uncomment/remove as needed
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def predict_image(model, image_path):
    image = cv2.imread(image_path)
    image = image / 255
    res = model.predict(np.expand_dims(image, 0))
    if res[0][0] > res[0][1]:
        classification = 'NO SEAL'
    else:
        classification = 'SEAL EXISTS'
    return image, res, classification


# Load the model
print('Loading the model...')
model_path = 'trained_models/'  # TO BE SET
model = load_model(model_path)
print('Model is ready')

######################################################################

dir_path = '../example_data/'

# Load and test true positive image
sub_dir_path = dir_path + 'seals/'
images_with_seals = [i for i in os.listdir(sub_dir_path) if i.endswith('.jpg')]

fig = plt.figure(figsize=(20, 20))
columns = 4
rows = 4
for i, image_name in enumerate(images_with_seals):
    image_path = sub_dir_path + image_name
    image, raw_results, predicted_class = predict_image(model, image_path)
    print('Image name: ' + image_name)
    print('Raw prediction: ' + str(raw_results))  # Raw prediction format: [no seal, seal exists]
    print('Predicted class: ' + predicted_class)
    ax = fig.add_subplot(rows, columns, i + 1)
    ax.title.set_text(image_name + ' -- ' + predicted_class)
    plt.imshow(image)

######################################################################

# Load and test true negative image
sub_dir_path = dir_path + 'no_seals/'
images_without_seals = [i for i in os.listdir(sub_dir_path) if i.endswith('.jpg')]

fig2 = plt.figure(figsize=(20, 20))
columns = 4
rows = 4
for i, image_name in enumerate(images_without_seals):
    image_path = sub_dir_path + image_name
    image, raw_results, predicted_class = predict_image(model, image_path)
    print('Image name: ' + image_name)
    print('Raw prediction: ' + str(raw_results))  # Raw prediction format: [no seal, seal exists]
    print('Predicted class: ' + predicted_class)
    ax2 = fig2.add_subplot(rows, columns, i + 1)
    ax2.title.set_text(image_name + ' -- ' + predicted_class)
    plt.imshow(image)

plt.show()
