from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ion() 

import get_mean_std_dataset as gm

#--------------------------------------------------------------------
#load the data from file. Fila has to be organised like:
#file/train/class1
#file/train/class2
#...
#file/val/class1
#file/val/class2
#...

file = "../../Datasets/mini"

#get mean and std for Data normalisation
params = gm.get_mean_std(file)
mymean, mystderr = params["mean"], params["stderr"]

data_transform = {
	'train': transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mymean,
                        std=mystderr)
]),
	'val': transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mymean,
                        std=mystderr)
])
	}

#Defines two datasets:
#amphiprion_dataset_t["train"]
#amphiprion_dataset_t["val"]
#apply the appropriate transformation for each of them

amphiprion_dataset_t = {X: datasets.ImageFolder(root=os.path.join(file,X),
                                           transform=data_transform[X])
						for X in ["train", "val"]}

#Loader allow to retrieve transformed images by batch, randomly sampled

dataset_loader_t = {X: torch.utils.data.DataLoader(amphiprion_dataset_t[X],
                                             batch_size=16, shuffle=True,
                                             num_workers=4)
					for X in ["train", "val"]}



dataset_sizes = {X: len(amphiprion_dataset_t[X]) for X in ['train', 'val']}

class_names = amphiprion_dataset_t["train"].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title=None):
    """Imshow for Tensor. 
    Tensor stores image as (C x H x W) in the range [0.0, 1.0].
    Numpy as (C x H x W) in the range [0.0, 1.0]"""

    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(mymean)
    std = np.array(mystderr)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Make a grid from batch
def Mkgrid():
	"""Makes a grid with a dataloade batch, takes no argument,requires Imshow"""
	inputs, classes = next(iter(dataset_loader_t['train']))
	out = torchvision.utils.make_grid(inputs)
	half = round(len(classes)/2) -1
	print(half)
	titles = [class_names[x].split("_")[1][0:3] for x in classes]
	titles[half] = "".join([titles[half], "\n"])
	title_joined = ", ".join(titles)
	
	imshow(out, title=title_joined)
	plt.savefig("grid.jpg")


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs): #25 x
        print('Epoch {}/{}'.format(epoch, num_epochs - 1)) 
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataset_loader_t[phase]:
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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

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
        for i, (inputs, labels) in enumerate(dataset_loader_t['val']):
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

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

#----------------------------------------------------------------------------
#Function Calling

model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=30)

visualize_model(model_conv)

plt.ioff()
plt.savefig("prediction.jpg")

torch.save(model_conv.state_dict(), "trainedTL_2fam1.pt")
