# skeleton file for neural network practice
import time
import os
import copy
import logging 

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import matplotlib as mpl
from matplotlib import pyplot as plt
import hedgeplot as hplt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

import time
import os
import copy

import cv2
import PIL.Image

data_dir = './database/cropped'
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ]),
}

# Read the dataset into data loaders
image_dataset = datasets.ImageFolder(data_dir)
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                  for x in ['train', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=0)
                for x in ['train', 'test']}


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}



# Create the label dictionary from list of food items
with open('food-items.txt') as f:
    item_names = f.read().splitlines()

n_classes = len(item_names)
label_dict = dict(zip(range(0, n_classes), item_names))


def apply_blur(image, size=3, sig=0.788):
    """Returns a image with random Gaussian blur applied.
    image      (torch.tensor): Image in the form of pytorch tensor to apply blur to.
    
    size                (int): Size for Gaussian blur.

    sig               (float): Maximum sig for Gaussian blur.
    """
    image = image.numpy()
    size = (size, size)
    image = cv2.GaussianBlur(image, size, sig, sig)
    return torch.tensor(image)


def apply_gauss_noise(image, mean=0, std=0.8):
    """Returns a image with random Gaussian noise applied.
    image      (torch.tensor): Image in the form of pytorch tensor to apply noise to.
    
    mean              (float): Mean for Gaussian noise.

    std               (float): Standard deviation for Gaussian noise.
    """
    image = image.numpy()
    noise = np.random.normal(mean, std, image.shape)
    image += noise.astype('uint8')
    image = np.clip(image, 0, 1)
    return torch.tensor(image)

def apply_sp_noise(image, prob=0.05, sp_ratio=0.5):
    """Returns a tensor with random salt and pepper noise applied.
    image      (torch.tensor): Image in the form of pytorch tensor to apply noise to.
    
    p                 (float): Probability of adding either salt or pepper to a pixel.

    sp_ratio          (float): Ratio between salt and pepper.
    """
    image = image.numpy()

    salt_prob = prob * sp_ratio

    for i in range(image.shape[1]):
        for j in range(image.shape[2]):
            random = np.random.random()
            if random <= salt_prob:
                image[:,i,j] = 1
            elif random <= prob:
                image[:,i,j] = 0 

    return torch.tensor(image)


# Decorate torchvision transforms to take tensors
PIL_to_tensor = transforms.ToTensor()
tensor_to_PIL = transforms.ToPILImage()

def wrap_transform(transform):
    def decorated(image):
        return PIL_to_tensor(transform(tensor_to_PIL(image)))

    return decorated


apply_color_jitter = wrap_transform(transforms.ColorJitter(brightness=0.3,
                         contrast=0.3, saturation=0.3, hue=0.05))

apply_random_affine = wrap_transform(transforms.RandomAffine(degrees=45, 
                        translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10))

apply_random_affine_large = wrap_transform(transforms.RandomAffine(degrees=90, 
                        translate=(0.25, 0.25), scale=(0.75, 1.25), shear=20))

# List of data augmentations to select from
data_augmentations = [
    apply_blur,
    apply_gauss_noise,
    apply_sp_noise,
    apply_color_jitter,
    apply_random_affine,
    apply_random_affine_large
]


n_augmentations = len(data_augmentations)

def apply_random_augmentation(inputs, p=0.8): 
    transformed_tensors = []
    for t in inputs:
        if np.random.rand() <= p:
            image = data_augmentations[int(np.random.rand() * n_augmentations)](t)
            transformed_tensors.append(image)
        else:
            transformed_tensors.append(t)
    return torch.stack(transformed_tensors)


def imshow(inp, title=None, pause=0.5):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    plt.title(title)
    plt.pause(pause)  # pause a bit so that plots are updated


# for inputs, labels in dataloaders['train']:
#     [imshow(i) for i in apply_random_augmentation(inputs)]


def train_model(model, criterion, optimizer, scheduler, num_epochs=30):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_record = []
    valid_loss_record = []

    epoch_loss = 0
    epoch_acc = 0 

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step(epoch_loss)
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                inputs, labels = data

                if phase == 'train':
                    inputs = apply_random_augmentation(inputs)
                
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
                logging.info('train: ' + str(epoch_loss))
                train_loss_record.append(epoch_loss)
            else:
                logging.info('valid: ' + str(epoch_loss))
                valid_loss_record.append(epoch_loss)

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        if (epoch % 10 == 0):
            torch.save(model, './checkpoints/checkpoint' + str(epoch) + '.pt')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    model.load_state_dict(best_model_wts)
    return model, train_loss_record, valid_loss_record, best_model_wts


class Combined(nn.Module):
    def __init__(self, base_model, n_classes):
        super(Combined, self).__init__()
        self.base_layer = nn.Sequential(*list(base_model.children())[:-1])
    
        # Remove the fc layer
        # self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(base_model.fc.in_features, n_classes)
    
    def forward(self, inputs):
        x = self.base_layer(inputs)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x



# Create a combined model with resnet as the base
resnet_model = models.resnet50(pretrained=True)
combined_model = Combined(resnet_model, n_classes)

criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(combined_model.parameters(), lr=0.0001, 
                                momentum=0.9, weight_decay=0.001)

# optimizer_conv = optim.Adam(combined_model.parameters(), lr=1e-3, 
#                                 betas=(0.9, 0.999), weight_decay=0.001)

# Decrease learning rate by 0.1 every 7 epochs
scheduler =  lr_scheduler.StepLR(optimizer_conv, step_size=4, gamma=0.1)
# scheduler = None

combined_model, train_loss_record, valid_loss_record, best_model_wts = train_model(combined_model, 
                        criterion, optimizer_conv, scheduler, num_epochs=70)



# combined_model = torch.load('./checkpoints/checkpoint20.pt')

# Make results df so that we can figure out where we did well
truth_hist = []
preds_hist = []
inputs_hist = []
for inputs, labels in dataloaders['test']:
    probs = combined_model(inputs)
    preds = np.argmax(probs.data.numpy(), axis=1)
    preds_hist.extend(preds)
    truth_hist.extend(labels.data.numpy())
    inputs_hist.extend(inputs)


result_df = pd.DataFrame()
result_df['truth_code'] = truth_hist
result_df['preds_code'] = preds_hist
result_df['image'] = inputs_hist
result_df['correct'] = result_df['truth_code'] == result_df['preds_code']
result_df['label'] = result_df['truth_code'].map(label_dict)
result_df['guessed'] = result_df['preds_code'].map(label_dict)

accuracy = result_df['correct'].mean()
group_accuracy = result_df.groupby('label')['correct'].mean().sort_values()

print("Accuracy: {}".format(accuracy))
print(group_accuracy)


result_df[result_df['label'] == 'fish'][['label', 'guessed']]
result_df[result_df['label'] == 'pinto_beans'][['label', 'guessed']]
result_df[result_df['label'] == 'parmesan_cheese'][['label', 'guessed']]

def show_result(result_df, item):
    for i, row in result_df[result_df['label'] == item].iterrows():
        imshow(row['image'], title="Label: {}, Guessed: {}".format(row['label'], row['guessed']), pause=2.5)

for item in group_accuracy.index[:8]:
    show_result(result_df, item)


for _, row in result_df.iterrows():
    if not row['correct']:
        imshow(row['image'], 
            title="Label: {}, Guessed: {}".format(row['label'], row['guessed']), pause=2.5) 


# show_result(result_df, 'beef')
# show_result(result_df, 'pork')
# show_result(result_df, 'brown_onion')
# show_result(result_df, 'chicken_leg')
# show_result(result_df, 'mushroom')
# show_result(result_df, 'cilantro')
