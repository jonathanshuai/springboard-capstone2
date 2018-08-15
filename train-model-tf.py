import collections
from datetime import datetime
import os.path
import random
import sys

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python import debug as tf_debug

import cv2
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

import dataloader
from dataloader import DataLoader

# Define run constants
tf.logging.set_verbosity(tf.logging.ERROR) # Remove INFO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

RUN_NAME = ""
PATHS_FILE = 'path_labels.csv'

IMAGE_SIZE = 224
BATCH_SIZE = 8
N_EPOCHS = 500
LEARNING_RATE = 1e-4
DEBUG = False

# Use resnet_v2_50 and we choose 'feature_vector' - the bottleneck part.
# This will be loaded into a hub.ModuleSpec - a pretrained model provided by tfhub.
MODULE_URL = 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1'

# Load data...

# Read in item names 
with open('food-items.txt') as f:
    item_names = f.read().splitlines()

# Count the number of items
n_classes = len(item_names)

# Make dictionaries to turn labels into indicies and back
label_dict_itos = dict(zip(range(0, n_classes), item_names))
label_dict_stoi = dict(zip(item_names, range(0, n_classes)))

# Read csv 
df = pd.read_csv(PATHS_FILE)
# df = df[  (df['label'] == 'apples') 
#         | (df['label'] == 'asparagus')
#         | (df['label'] == 'avocado') 
# #         | (df['label'] == 'bacon')
#         ]
# n_classes = 3

# Get file paths from df.
file_paths = df['cropped_path'].values

# Get labels
labels = df['label'].apply(lambda x: label_dict_stoi[x]).values

# # Use one-hot encoding
# one_hot_labels = np.zeros((labels.shape[0], n_classes))
# one_hot_labels[np.arange(labels.shape[0]), labels] = 1

# Split into test/validation sets 
(file_paths_train, file_paths_valid, 
    labels_train, labels_valid)  = train_test_split(
                                    file_paths,
                                    labels,
                                    stratify=labels,
                                    test_size=0.2)

train_length = file_paths_train.shape[0]
valid_length = file_paths_valid.shape[0]

# List transformations (these are defined in dataloader.py)
transforms = [
    # (lambda x: x,                          {}),
    # (dataloader.apply_blur,                {}),
    # (dataloader.apply_brightness,          {}),
    # (dataloader.apply_color_jitter,        {}),
    # (dataloader.apply_sp_noise,            {}),
    # (dataloader.apply_gauss_noise,         {}),
    # (dataloader.apply_random_rotate,       {}),
    # (dataloader.apply_random_translate,    {}),
    # (dataloader.apply_random_crop_resize,  {}),
    # (dataloader.apply_affine,              {})
]

# Create data loader (once again, defined in dataloader.py)
data_loader_train = DataLoader(file_paths_train, labels_train, 
                            batch_size=BATCH_SIZE, 
                            image_size=(IMAGE_SIZE, IMAGE_SIZE), 
                            transforms=transforms)

data_loader_valid = DataLoader(file_paths_valid, labels_valid, 
                            batch_size=BATCH_SIZE, 
                            image_size=(IMAGE_SIZE, IMAGE_SIZE), 
                            transforms=[])


# Create TensorFlow placeholders. 
with tf.Graph().as_default() as graph:
    inputs = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
    labels = tf.placeholder(tf.int64, shape=[None,])

# Load a hub.ModuleSpec that is basically a pretrained model provided by tfhub. 
module_spec = hub.load_module_spec(MODULE_URL)

# From TensorFlow tutorial
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    # tf.summary.histogram('histogram', var)

def create_module_graph(graph, module_spec, inputs):
    """Create a tf.Graph and load the module into it
    """

    # We can get the height and width that our module expects 
    height, width = hub.get_expected_image_size(module_spec)

    # Create a new tf.Graph with the module in it
    # We can create session out of this graph later using tf.Session(graph=graph)
    # with tf.Graph().as_default() as graph:
    with graph.as_default():
        # base_model = hub.Module(module_spec, trainable=True, tags={'train'})
        base_model = hub.Module(module_spec)
        bottleneck_tensor = base_model(inputs)
        tf.summary.image('input_image', inputs)
        # tf.summary.histogram('bottleneck_tensor', bottleneck_tensor)

    return graph, bottleneck_tensor 

# Add the fully connected layer to the graph
def add_fc_layer(graph, bottleneck_tensor, labels):
    """Add fc_layer to the graph 
    """
    
    # Find the size of the bottleneck features
    _, bottleneck_tensor_size = bottleneck_tensor.shape.as_list()
    

    # Using the graph from bottleneck features...
    with graph.as_default():
        # Make placeholder for bottleneck input and ground truth
        with tf.name_scope('fc_input'):
            fc_input = tf.placeholder_with_default(
                bottleneck_tensor,
                shape=[None, bottleneck_tensor_size],
                name='fc_input_placeholder')

        # Name of fully connected layer as fc_layer    
        with tf.name_scope('fc_layer'):
            # Create regularizer
            # regularizer = tf.contrib.layers.l2_regularizer(scale=0.0005)
            # logits is the output of dense layer output (w/ regularization) 
            with tf.name_scope('logits'):
                logits = tf.layers.dense(
                    inputs=bottleneck_tensor, 
                    units=n_classes,
                    # kernel_regularizer=regularizer,
                    name="dense_layer"
                )
                logits_mean = tf.reduce_mean(logits)
                tf.summary.scalar('mean', logits_mean)
                tf.summary.histogram('histogram', logits_mean)

            # Get prediction (softmax of logits)
            final_tensor = tf.nn.softmax(logits, name='prediction')


        # Defining loss function
        with tf.name_scope('loss'):
            loss = tf.losses.sparse_softmax_cross_entropy(
                labels=labels, logits=logits)

            gradients = tf.gradients(loss, [logits])

        # Accuracy
        with tf.name_scope('accuracy'):
            prediction = tf.argmax(final_tensor, 1)
            equality = tf.equal(prediction, labels)
            accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
        
        # Optimization step
        with tf.name_scope('train'):
            # Start step at 0
            global_step = tf.Variable(0, trainable=False)

            # Create a scheduler to multiply learning rate by 0.9 every 10 steps
            scheduler = tf.train.exponential_decay(LEARNING_RATE, global_step,
                                                       4 * train_length, 0.9, staircase=True)

            # Create optimizer (gradient descent) and have it minimize the loss 
            optimizer = tf.train.GradientDescentOptimizer(scheduler)

            # optimizer = tf.train.MomentumOptimizer(LEARNING_RATE, 0.9)
            # optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
            train_step = optimizer.minimize(loss, global_step=global_step)

        # Logging
        with tf.name_scope('logging'):
            variable_summaries(logits)
            # variable_summaries(loss)
            tf.summary.scalar('current_loss', loss)
            tf.summary.histogram('prediction', prediction)
            summary = tf.summary.merge_all()


    return train_step, loss, accuracy, summary, prediction


graph, bottleneck_tensor = create_module_graph(graph, module_spec, inputs)
train_step, loss, accuracy, summary, prediction = add_fc_layer(graph, bottleneck_tensor, labels)



# Time to train the model

print("Beginning training...")
# Start a session with the graph we created
with tf.Session(graph=graph) as session:
    if DEBUG:
        session = tf_debug.LocalCLIDebugWrapperSession(session)

    # Initialize all weights: for the module to their pretrained values,
    # and for the newly added retraining layer to random initial values.
    init = tf.global_variables_initializer()
    session.run(init)

    # Create the log writers for the summary 
    train_writer = tf.summary.FileWriter("./logs/{}/train".format(RUN_NAME), session.graph)
    valid_writer = tf.summary.FileWriter("./logs/{}/valid".format(RUN_NAME), session.graph)

    for epoch in range(N_EPOCHS):

        print("Training epoch {}...".format(epoch))
        # Training Phase
        running_train_loss = 0
        running_train_acc = 0

        # Get data from our training DataLoader in batches
        for batch_images, batch_labels in data_loader_train.get_data():
            _, train_loss, train_acc, train_summary, train_inputs = session.run(
                [train_step, loss, accuracy, summary, inputs],
                feed_dict={inputs: batch_images, labels: batch_labels})

            running_train_acc += train_acc * train_inputs.shape[0]
            running_train_loss += train_loss

        running_train_acc /= train_length

       
        print("Validation epoch {}...".format(epoch))
        # Validation Phase
        running_valid_loss = 0
        running_valid_acc = 0

        # Get data from our validation DataLoader in batches
        for batch_images, batch_labels in data_loader_valid.get_data():
            valid_loss, valid_acc, valid_summary, valid_inputs = session.run(
                [loss, accuracy, summary, inputs],
                feed_dict={inputs: batch_images, labels: batch_labels})

            running_valid_acc += valid_acc * valid_inputs.shape[0]
            running_valid_loss += valid_loss

        running_valid_acc /= valid_length


        # Print results of current epoch    
        print("-" * 10)
        print("Epoch {}".format(epoch))
        print("Train loss: {:.2f}\nTrain acc: {:.6f}".format(running_train_loss, 
                                                                running_train_acc))
        print("Valid loss: {:.2f}\nValid acc: {:.6f}".format(running_valid_loss, 
                                                                running_valid_acc))
        print("-" * 10)
        
        # Build on summary
        train_writer.add_summary(train_summary, epoch)
        valid_writer.add_summary(valid_summary, epoch)


    print("Saving model...")
    # Save the model
    saver = tf.train.Saver()

    # Save the model
    save_path = saver.save(session, "logs/trained_model.ckpt")
    print("Model Saved: {}".format(save_path))
    print("Training finished!")


    images = []
    truths = []
    preds = []
    for batch_images, batch_labels in data_loader_valid.get_data():
        valid_preds = session.run(prediction, 
            feed_dict={inputs: batch_images, labels: batch_labels})

        images.extend(batch_images)
        truths.extend(batch_labels)
        preds.extend(valid_preds)



# Create a DataFrame with all the results
result_df = pd.DataFrame({
  'image': images,
  'truth': list(map(lambda x: label_dict_itos[x], truths)),
  'pred': list(map(lambda x: label_dict_itos[x], preds))
})

# Show incorrect classifications
for _, row in result_df[result_df['truth'] != result_df['pred']].iterrows():
    image = row['image']
    truth = row['truth']
    pred = row['pred']
    plt.imshow(image) 
    plt.title("Truth: {} | Guessed: {}".format(truth, pred))
    plt.pause(0.8)









