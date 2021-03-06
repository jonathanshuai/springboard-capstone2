# skeleton file for neural network practice
import copy

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib as mpl
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

from torchvision import models

import dataloader
from dataloader import DataLoader

# Define run constants

# csv created in preprocessing that where all the images are
PATHS_FILE = '../database/cropped/path_labels.csv' 
# file from raw data that tells all the class names (alphabetized)
ITEM_NAMES_FILE = '../database/raw/food-items.txt'

SEED = 17               # Seed for train_test_split 

IMAGE_SIZE = 224        # Size of input images expected by base model
BATCH_SIZE = 8          # Size of each batch 
N_EPOCHS = 80           # Number of epochs to train for
LEARNING_RATE = 1e-4    # Initial learning rate
STEP_SIZE = 8           # Number of epochs before one step for exponential decay
GAMMA = 0.1             # Amount to reduce learning rate by 

RUN_NAME = "batch_size-{}n_epochs-{}learning_rate-{}step_size-{}gamma-{}"\
    .format(BATCH_SIZE, N_EPOCHS, LEARNING_RATE, STEP_SIZE, GAMMA)

# Load data...
# Read in item names 
with open(ITEM_NAMES_FILE) as f:
    item_names = f.read().splitlines()

# Count the number of items
n_classes = len(item_names)

# Make dictionaries to turn labels into indicies and back
label_dict_itos = dict(zip(range(0, n_classes), item_names))
label_dict_stoi = dict(zip(item_names, range(0, n_classes)))

# Read csv 
df = pd.read_csv(PATHS_FILE)

# Get file paths from df.
file_paths = df['cropped_path'].values

# Get labels
labels = df['label'].apply(lambda x: label_dict_stoi[x]).values

# Split into test/validation sets 
(file_paths_train, file_paths_valid, 
    labels_train, labels_valid)  = train_test_split(
                                    file_paths,
                                    labels,
                                    stratify=labels,
                                    test_size=0.2,
                                    random_state=SEED)

# List transformations (these are defined in dataloader.py)
transforms = [
    (lambda x: x,                          {}),
    (dataloader.apply_blur,                {}),
    (dataloader.apply_brightness,          {}),
    (dataloader.apply_color_jitter,        {}),
    (dataloader.apply_sp_noise,            {}),
    (dataloader.apply_gauss_noise,         {}),
    (dataloader.apply_affine,              {}),
    (lambda img: dataloader.apply_color_jitter(dataloader.apply_affine(img)), {})
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

dataloaders = {'train': data_loader_train, 'valid': data_loader_valid}
dataset_sizes = {phase: dataloaders[phase].shape[0] for phase in dataloaders}


def train_model(model, criterion, optimizer, scheduler, num_epochs=30):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss_record = []
    valid_loss_record = []

    train_acc_record = []
    valid_acc_record = []

    epoch_loss = 0
    epoch_acc = 0 

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step(epoch_loss)
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase].get_data():
                inputs, labels = data

                # Use pytorch standard [batch_size, channel, height, width]
                inputs = torch.tensor([[inp[:, :, 0], inp[:, :, 1], inp[:, :, 2]] for inp in inputs])\
                    .type_as(torch.FloatTensor())

                labels = torch.tensor(labels).type_as(torch.LongTensor())
                labels = labels.view(-1)

                inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]

            print('{} Loss : {:.4f} Acc : {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                train_loss_record.append(epoch_loss)
                train_acc_record.append(epoch_acc)
            else:
                valid_loss_record.append(epoch_loss)
                valid_acc_record.append(epoch_acc)

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if (epoch % 10 == 0):
            checkpoint_path = './checkpoints/checkpoint' + RUN_NAME + str(epoch) + '.pt'
            torch.save(model, checkpoint_path)
            print("Saved checkpoint: {}".format(checkpoint_path))

    model.load_state_dict(best_model_wts)
    return model, train_loss_record, valid_loss_record, train_acc_record, valid_acc_record


class TransferModel(nn.Module):
    def __init__(self, base_model, n_classes):
        super(TransferModel, self).__init__()
        # Remove the fc layer
        self.base_layer = nn.Sequential(*list(base_model.children())[:-1])
    
        # self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(base_model.fc.in_features, n_classes)
    
    def forward(self, inputs):
        x = self.base_layer(inputs)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x



# Create a combined model with resnet as the base
resnet_model = models.resnet50(pretrained=True)
transfer_model = TransferModel(resnet_model, n_classes)

criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(transfer_model.parameters(), lr=LEARNING_RATE, 
                                momentum=0.9, weight_decay=0.001)

# optimizer_conv = optim.Adam(transfer_model.parameters(), lr=1e-3, 
#                                 betas=(0.9, 0.999), weight_decay=0.001)

# Decrease learning rate by 0.1 every 7 epochs
scheduler =  lr_scheduler.StepLR(optimizer_conv, step_size=STEP_SIZE, gamma=GAMMA)

(transfer_model, train_loss_record, valid_loss_record, 
    train_acc_record, valid_acc_record) = train_model(transfer_model, 
                        criterion, optimizer_conv, scheduler, num_epochs=N_EPOCHS)


# Make results df so that we can figure out where we did well
truth_hist = []
preds_hist = []
inputs_hist = []
for inputs, labels in dataloaders['valid']:
    probs = transfer_model(inputs)
    preds = np.argmax(probs.data.numpy(), axis=1)
    preds_hist.extend(preds)
    truth_hist.extend(labels.data.numpy())
    inputs_hist.extend(inputs)


result_df = pd.DataFrame()
result_df['truth_code'] = truth_hist
result_df['preds_code'] = preds_hist
result_df['image'] = inputs_hist
result_df['correct'] = result_df['truth_code'] == result_df['preds_code']
result_df['label'] = result_df['truth_code'].map(label_dict_itos)
result_df['guessed'] = result_df['preds_code'].map(label_dict_itos)

accuracy = result_df['correct'].mean()
group_accuracy = result_df.groupby('label')['correct'].mean().sort_values()

print("Accuracy: {}".format(accuracy))
print(group_accuracy)