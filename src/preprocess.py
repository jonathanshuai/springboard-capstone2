import os

from contextlib import suppress

import cv2

import numpy as np

import pandas as pd

import skimage.color
import skimage.filters
import skimage.segmentation

from tqdm import tqdm


# Get the paths to the raw image files and put them in a DataFrame
raw_paths = []

# Iterate through 'database' directory recursively
for root, directories, filenames in os.walk('../database/raw'):
    # Only get images (in subfolders)
    if not root == '../database/raw':
        for filename in filenames:
            raw_paths.append(os.path.join(root, filename))


labels = [r.split('/')[3] for r in raw_paths]

# Get the cropped path names: these are the paths we will be writing to
cropped_paths = [s.replace('/raw', '/cropped') for s in raw_paths]


# Create a DataFrame
df = pd.DataFrame({'raw_path': raw_paths,
                   'label': labels,
                   'cropped_path': cropped_paths})


# Define functions to write, crop, and do transforms for data augmentation
def write_image(image, path):
    """Write an image to disk.
    image      (numpy.ndarray): Image in the form of 3d array to write.

    path                 (str): Path to target location on filesystem.
    """
    directory = os.path.dirname(path)
    with suppress(FileExistsError):
        os.makedirs(directory)
    cv2.imwrite(path, image)


def get_bounding_box(image, snake_params):
    """Returns a bounding box for a crop using active contour snakes algorithm.
    image      (numpy.ndarray): Image in the form of 3d array to find bounding
    box.

    snake_params        (dict): Parameters to use for skimage's active_contour.
    """

    # apply grayscale and Gaussian blur; note a=2.5 or sig=0.4472
    image_gray = skimage.color.rgb2gray(image)
    image_gray_blur = skimage.filters.gaussian(image_gray, 1.028)

    # Create an init snake (circle) and use active_contour to shrink it
    s = np.linspace(0, 2*np.pi, 300)
    x = image.shape[1] / 2 + (image.shape[1] / 2) * np.cos(s)
    y = image.shape[0] / 2 + (image.shape[0] / 2) * np.sin(s)
    
    init = np.array([x,y]).T
    snake = skimage.segmentation.active_contour(image_gray_blur, 
                                                init, **snake_params)

    # Get the minimum bounding box from the snake
    left = np.max([np.min(snake[:,0]).astype(int), 0])
    right = np.min([np.max(snake[:,0]).astype(int), image.shape[1]])
    top = np.max([np.min(snake[:,1]).astype(int), 0])
    bottom = np.min([np.max(snake[:,1]).astype(int), image.shape[0]])

    # Ensure the resulting image is square
    # Caculate length of the square and the margin size
    side_length = np.min([right - left, bottom - top])
    
    # If snakes does something weird, just do center crop w/ 80% side length
    if side_length < 150:
        side_length = int(np.min(image.shape[:2]) * 0.8)
        h_center = image.shape[1] // 2
        v_center = image.shape[0] // 2

        margin_size = side_length // 2
        left = h_center - margin_size
        right = h_center + margin_size
        top = v_center - margin_size
        bottom = v_center + margin_size

    # Otherwise, center square crop on region bounded by snake
    else:
        h_margin = ((right - left) - side_length)
        v_margin = ((bottom - top) - side_length)

        # Distribute margin s.t. crop is centered
        left += h_margin  // 2
        right -= h_margin // 2 + (h_margin % 2)
        top += v_margin // 2
        bottom -= v_margin // 2 + (v_margin % 2)

    # Width and height should be equal
    assert left - right == top - bottom

    # Return the bounding box as two points
    bounding_box = [left, right, top, bottom]
    
    return bounding_box, side_length

def preprocess(image_df, snake_params):
    # Initialize some diagnostic information
    n_samples = image_df.shape[0]

    for i, row in tqdm(image_df.iterrows(), total=image_df.shape[0]):
        # Read in image
        image = cv2.imread(row['raw_path'])

        # Get bounding box
        bounding_box, side_length = get_bounding_box(image, snake_params)

        # Crop image 
        left, right, top, bottom = bounding_box
        image = image[top:bottom, left:right]

        write_image(image, row['cropped_path'])


# Parameters for active contour
snake_params = {
            'alpha': 5e-4, 
            'beta': 1e1, 
            'w_line': -2e-1, 
            'w_edge': 0, 
            'gamma': 1e-5,
            'max_iterations': 200
        }                                        

# Make crops from snakes and write to disk
preprocess(df, snake_params)

df[['cropped_path', 'label']].to_csv('../database/cropped/path_labels.csv', index=False)