import os

import numpy as np

import cv2
import pandas as pd

import torch
from torch.nn import Softmax

from matplotlib import pyplot as plt
from model_definition import Combined

ITEM_NAMES_FILE = './models/food-items.txt'
MODEL_FILE = './models/checkpoint70.pt'
IMAGE_SIZE = 224
STRIDE = 70

with open(ITEM_NAMES_FILE) as f:
    item_names = f.read().splitlines()

# Count the number of items
n_classes = len(item_names)

# Make dictionaries to turn labels into indicies
label_dict_itos = dict(zip(range(0, n_classes), item_names))

model = torch.load(MODEL_FILE)
softmax = Softmax()

def find_recipes(file_path, options):
    """
    """
    # get_items - runs the nn on the image, returning a set of items
    # call_api - 

    predictions = get_items(file_path)
    print(predictions)

def get_items(file_path):
    """Return a list of items detected in image.
    file_path     (str): Image in the form of 3d array to apply transformation to.
    """
    image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

    preds_on_orig = sample_and_predict(image)
    preds_on_scaled = sample_and_predict(cv2.resize(image, (0,0), fx=1.5, fy=1.5))

    return predictions

def sample_and_predict(image):
    """Make predictions on windows taken from image.
    image         (np.ndarray): Image to sample and make predictions on.

    """
    height = image.shape[0]
    width = image.shape[1]

    rows = 1 + (height - IMAGE_SIZE) // STRIDE
    cols = 1 + (width - IMAGE_SIZE) // STRIDE

    predictions = []
    i = 1    
    y_pos = 0
    while y_pos + IMAGE_SIZE < height:
        x_pos = 0
        while x_pos + IMAGE_SIZE < width:
            # Get a crop from the original image
            crop = image[y_pos:y_pos+IMAGE_SIZE, x_pos:x_pos+IMAGE_SIZE]

            # Turn it into a tensor for the model
            crop_tensor = torch.tensor([[crop[:,:,0], crop[:,:,1], crop[:,:,2]]])\
                            .type_as(torch.FloatTensor())
            # Make predictions and ave them
            probs = softmax(model(crop_tensor).data).numpy()
            pred = label_dict_itos[np.argmax(probs, axis=1)[0]]
            predictions.append(pred)
    
            # Show the image
            plt.subplot(rows, cols, i)
            plt.imshow(crop)
            plt.title("{}, {:.2f}".format(pred, probs.max()))
            plt.axis('off')

            # Move the window to the right
            x_pos += STRIDE
            i += 1 # next subplot
        # Move the window down
        y_pos += STRIDE

    plt.show()

    return predictions

options = [('milk', 0), ('nuts', 0), ('pork', 1)]
# file_path = "../temp/salmononions.jpg"
# file_path = "../temp/salmonburgers.png"
file_path = "../temp/salmoncakes.jpg"

find_recipes(file_path, options)

image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

torch.tensor([image[:224, :224, 0], 
                image[:224, :224, 1], 
                image[:224, :224, 2]]
                ).type_as(torch.FloatTensor())


image_t = torch.tensor(image[:224, :224])