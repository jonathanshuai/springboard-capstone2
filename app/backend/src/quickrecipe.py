import os
import time
from itertools import combinations

import numpy as np

import pandas as pd

import cv2
import torch
from torch.nn import Softmax

from .model_definition import TransferModel

import requests

ITEM_NAMES_FILE = os.getcwd() + '/models/food-items.txt'
MODEL_FILE = os.getcwd() + '/models/model.pt'
IMAGE_SIZE = 224
STRIDE = 112

APP_ID = '569fabe5'
APP_KEY = 'ce79f2d3e113d81b70c13dbfc2f4bb35'

N_RECIPES = 10

with open(ITEM_NAMES_FILE) as f:
    item_names = f.read().splitlines()

# Count the number of items
n_classes = len(item_names)

# Make dictionaries to turn labels into indicies
label_dict_itos = dict(zip(range(0, n_classes), item_names))

VEGETARIAN = {
'bacon',
'beef',
'chicken_breast',
'chicken_leg',
'chicken_wing',
'pork',
'pork_ribs',
'salmon',
'tuna',
'tilapia'
}

VEGAN = {
'eggs',
'cheddar_cheese',
'mozzarella_cheese',
'parmesan_cheese'
}

model = torch.load(MODEL_FILE)
softmax = Softmax()

def find_recipes(file_path, options):
    """
    """
    # get_items - runs the nn on the image, returning a set of items

    print("Making predictions...")
    orig_predictions = get_items(file_path)
    print(orig_predictions)
    if options[1][1]: # If option for vegetarian
        predictions = orig_predictions - VEGETARIAN

    if options[0][1]: # If option for vegan
        predictions = predictions - VEGAN - VEGETARIAN

    print("Finding recipes...")
    recipes = get_matching_recipes(predictions, options)

    return orig_predictions, recipes 

def get_matching_recipes(ingredients, options):
    # Keep finding recipes until there are at least N_RECIPES
    recipes = []

    # API call with subset of ingredients 
    for i in range(len(ingredients), 0, min(-1, -(len(ingredients) // 2))):
        if i < 1:
            i = 1

        for ingredient_set in combinations(ingredients, i):
            print(ingredient_set)
            # Call API and get json data 
            json_data = call_api(ingredient_set, options)
            
            # What to do if we get an error?
            if json_data is None:
                return recipes

            # Look through each hit
            for hit in json_data['hits']:
                recipe_url = hit['recipe']['url']
                recipe_title = hit['recipe']['label']
                recipe_image = hit['recipe']['image']

                recipe = {'title': recipe_title, 
                            'url': recipe_url, 
                            'imgsrc': recipe_image,
                            'uses': ', '.join(ingredient_set).replace('_', ' ') }
                recipes.append(recipe)

                if len(recipes) >= N_RECIPES:
                    return recipes
        

def call_api(ingredients, options):
    base_url = "https://api.edamam.com/search?app_id={}&app_key={}".format(APP_ID, APP_KEY)

    query = 'q=' + (','.join(ingredients)).replace('_', '%20')
    
    options = map(lambda x: x[0].replace('_', '-'), (filter(lambda x: x[1] == 1, options)))
    health =  '&'.join(['health=' + option for option in options])

    request_url = '&'.join([base_url, query, health])

    r = requests.get(request_url)

    if r.status_code == 403:
        return None

    # We get 401 when we made too many requests. What to do?
    while r.status_code == 401:
        return None

    json_data = r.json()

    return json_data

def get_items_mock(file_path):
    return {'potato', 'chicken', 'apples'}

def get_items(file_path):
    """Return a list of items detected in image.
    file_path     (str): Image in the form of 3d array to apply transformation to.
    """
    image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

    preds_on_orig = sample_and_predict(image)
    preds_on_scaled = sample_and_predict(cv2.resize(image, (0,0), fx=1.2, fy=1.2))

    return set(preds_on_orig + preds_on_scaled)

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

            if probs.max() > 0.6:
                pred = label_dict_itos[np.argmax(probs, axis=1)[0]]
                predictions.append(pred)
        
            # Move the window to the right
            x_pos += STRIDE
            i += 1 # next subplot
        # Move the window down
        y_pos += STRIDE

    return predictions