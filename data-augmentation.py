import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

# First, load the image again
image_files = [ 
                './database/cropped/bananas/1.jpg', 
                './database/cropped/bananas/2.jpg', 
                './database/cropped/bananas/3.jpg',
                './database/cropped/bananas/4.jpg',
                './database/cropped/strawberries/0.jpg',
                './database/cropped/strawberries/1.jpg',
                './database/cropped/strawberries/2.jpg',
                './database/cropped/strawberries/3.jpg'
                ]
images = [mpimg.imread(image_file) for image_file in image_files]


# Create a function to apply to dataset for data augmentation
def make_transformer(transform_list):
    # Count number of transforms
    n_transforms = len(transform_list)

    def transformer(image, label):
        # Choose a random transformation, kwargs pair
        index = int(np.random.random() * n_transforms)
        transform, kwargs = transform_list[index]
        print(index)

        # Apply transformation and return with label
        image = transform(image, **kwargs)
        return image, label

    # Return new data augmentation mapping function        
    return transformer

# Transform list: (transformation, kwargs)
transform_list = [
    (tf.image.random_hue, {'max_delta': 0.07}),
    (tf.image.random_brightness, {'max_delta': 0.2}),
    (tf.image.random_contrast, {'lower': 0.30, 'upper': 0.80}),
    (tf.image.random_saturation, {'lower': 0.30, 'upper': 0.80}),
    (tf.image.rot90, {'k': 1}),
    (tf.image.flip_up_down, {}),
    (tf.image.flip_left_right, {}),
]


def apply_gauss_noise(tensor, mean=0, std=0.8):
    """Returns a image with random Gaussian noise applied.
    tensor        (tf.Tensor): Image in the form of TensorFlow tensor to apply noise to.
    
    mean              (float): Mean for Gaussian noise.

    std               (float): Standard deviation for Gaussian noise.
    """
    image = tensor.eval()
    noise = np.random.normal(mean, std, image.shape)
    image += noise.astype('uint8')
    image = np.clip(image, 0, 255)

    return tf.convert_to_tensor(image)

def apply_blur(tensor, size=3, sig=0.788):
    """Returns a image with random Gaussian blur applied.
    tensor        (tf.Tensor): Image in the form of TensorFlow tensor to apply blur to.
    
    size                (int): Size for Gaussian blur.

    sig               (float): Maximum sig for Gaussian blur.
    """
    image = tensor.eval()
    size = (size, size)
    image = cv2.GaussianBlur(image, size, sig, sig)

    return tf.convert_to_tensor(image)


def apply_sp_noise(tensor, prob=0.05, sp_ratio=0.5):
    """Returns a tensor with random salt and pepper noise applied.
    tensor        (tf.Tensor): Image in the form of TensorFlow tensor to apply noise to.
    
    p                 (float): Probability of adding either salt or pepper to a pixel.

    sp_ratio          (float): Ratio between salt and pepper.
    """
    image = tensor.eval()
    print(image.shape)
    salt_prob = prob * sp_ratio

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            random = np.random.random()
            if random <= salt_prob:
                image[i,j,:] = 1
            elif random <= prob:
                image[i,j,:] = 0 

    return tf.convert_to_tensor(image)

def apply_random_rotate(tensor, degrees=180):
    """ Returns a tensor rotated by a random degree.
    tensor        (tf.Tensor): Image in the form of TensorFlow tensor to apply rotate to.
    
    degrees           (float): Rotation by random amount will be in range [-degrees, +degrees].
    """
    image = tensor.eval()

    width = image.shape[1]
    height = image.shape[0]
    
    to_rotate = 2 * np.random.random() * degrees - degrees
    M = cv2.getRotationMatrix2D((width / 2, height / 2), to_rotate, 1)
    image = cv2.warpAffine(image, M, (width, height))

    return tf.convert_to_tensor(image)

def apply_random_translate(tensor, max_ratio=0.35):
    """ Returns a tensor translated by a random amount.
    tensor        (tf.Tensor): Image in the form of TensorFlow tensor to apply translate to.
    
    max_ratio         (float): Translation amount will be in range [-max_ratio, max_ratio] * size.
    """
    image = tensor.eval()

    side_length = image.shape[0]
    max_trans = side_length * max_ratio

    x_trans = 2 * np.random.random() * max_trans - max_trans
    y_trans = 2 * np.random.random() * max_trans - max_trans

    M = np.float32([[1, 0, x_trans],
                     [0, 1, y_trans]]) 
   
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return tf.convert_to_tensor(image)

def apply_random_crop_resize(tensor, min_ratio=0.20, max_ratio=0.40):
    """ Returns a tensor cropped and resized by a random amount.
    tensor        (tf.Tensor): Image in the form of TensorFlow tensor to crop and resize.
    
    max_ratio         (float): Crop resize amount will be in range [-max_ratio, max_ratio] * size.
    """
    image = tensor.eval()

    width = image.shape[1]
    height = image.shape[0]
    
    ratio = np.random.random() * (max_ratio - min_ratio) + min_ratio

    x_margin = int(ratio * width // 2)
    y_margin = int(ratio * width // 2)


    x_lower, x_upper = x_margin, width - x_margin
    y_lower, y_upper = y_margin, height - y_margin

    cropped_image = image[y_lower:y_upper, x_lower:x_upper]
    resized_image = cv2.resize(cropped_image, (width, height))

    return tf.convert_to_tensor(resized_image)

transform = apply_random_crop_resize
kwargs = {}

# Create a TensorFlow Variable
image_variables = [tf.Variable(image, name='x') for image in images]

init_op = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init_op)
    image_variables = [transform(image_variable, **kwargs) for image_variable in image_variables]
    results = [session.run(image_variable) for image_variable in image_variables]

for orig, result in zip(images, results):
    plt.imshow(orig)
    plt.pause(0.1)
    plt.imshow(result)
    plt.pause(0.1)
