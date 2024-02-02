"""
Industrial 100
100 classes, 80-20 split for "white" background
Joint Training! This will serve as the baseline.
Modelled after> https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from time import sleep
import wandb

import random

# ensure deterministic behaviour
np.random.seed(404543)
random.seed(404543) # set seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.manual_seed(404543)
torch.cuda.manual_seed_all(404543)

from torchvision.models import resnet18

#plt.ion()   # interactive mode

wandb.init(project="cvpr_2024")



# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.RandomResizedCrop(2976),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #torchvision.transforms.Grayscale(num_output_channels=3),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #transforms.Normalize((0.5), (0.5))
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.CenterCrop(2976),
        transforms.ToTensor(),
        #torchvision.transforms.Grayscale(num_output_channels=3),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #transforms.Normalize((0.5), (0.5))
    ]),
}

train_dir = '/mnt/1TBNVME/vivek/2024/Stanford_Cars/imgs_train/'
val_dir = '/mnt/1TBNVME/vivek/2024/Stanford_Cars/imgs_test/'

image_datasets = {}

image_datasets['train'] = ImageFolder(train_dir, data_transforms['train'])
image_datasets['val'] = ImageFolder(val_dir, data_transforms['val'])

dataloaders = {}

dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=4, shuffle=True, num_workers=4)
dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=4, shuffle=True, num_workers=4)

dataset_sizes = {}
dataset_sizes['train'] = len(image_datasets['train'])
dataset_sizes['val'] = len(image_datasets['val'])

class_names = image_datasets['train'].classes

print("Classes: ", "\n", class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""Temporarily using CPU"""
#device = torch.device("cpu")

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
plt.show()

# Training the Model:

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, num_epochs+1):

        #print learning rate
        for param_group in optimizer.param_groups:
            print("Learning rate: ", param_group['lr'])

        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()


                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)


                    tepoch.set_description(f"Epoch {epoch} " + phase)
                    tepoch.set_postfix(loss=loss.item())
                    tepoch.update()

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                wandb.log({f'Train Loss': epoch_loss, f'Train Acc': epoch_acc})
            else:
                wandb.log({f'Val Loss': epoch_loss, f'Val Acc': epoch_acc})



            print()
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))


            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])
                plt.show()

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


model_ft = models.resnet18(100)   #models.resnet34(100)
wandb.watch(model_ft)

#model_ft = models.resnet152(100)

num_ftrs = model_ft.fc.in_features
# Here the size of each output sample can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.5)


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

# save the model
#torch.save(model_ft.state_dict(), "Industrial_100_PCA_41.pth")

visualize_model(model_ft)
