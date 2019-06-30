from __future__ import print_function, division

import time
start = time.time()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import copy
import pickle
import random

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

#Chose source file
file = "../../Datasets/mini"

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


#get mean and std for Data normalisation
params = gm.get_mean_std(file)
mymean, mystderr = params["mean"], params["stderr"]

#Define transformation for each dataset, training and validation.
data_transform = {
    'train': transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.5, 1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mymean,
                        std=mystderr)
]),
    'val': transforms.Compose([
    transforms.Resize(96),
    transforms.CenterCrop(96),
    transforms.ToTensor(),
    transforms.Normalize(mean=mymean,
                        std=mystderr)
])
    }

#Defines two datasets:
#amphiprion_dataset_t["train"]
#amphiprion_dataset_t["val"]
#apply the appropriate transformation for each of them

amphiprion_dataset_t = {X: ImageFolderWithPaths(root=os.path.join(file,X),
                                           transform=data_transform[X])
                        for X in ["train", "val"]}

#Loader allow to retrieve transformed images by batch, randomly sampled

dataset_loader_t = {X: torch.utils.data.DataLoader(amphiprion_dataset_t[X],
                                             batch_size=16, shuffle=True,
                                             num_workers=4)
                    for X in ["train", "val"]}

#Extract parameter of the data
dataset_sizes = {X: len(amphiprion_dataset_t[X]) for X in ['train', 'val']}

class_names = amphiprion_dataset_t["train"].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net96(nn.Module):
    def __init__(self): #only defines the architecture of the net
        super(Net96, self).__init__()
        self.conv1 = nn.Conv2d(3,70, 5, stride = 1)
        # 1 input image channel, 
        # 6 output channels,
        # 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(70, 110, 3) 
        #70 inputs as it is the output of conv1
        self.conv3 = nn.Conv2d(110,180,3)
        #110 input, 180 out
        self.fc1 = nn.Linear(180*9*9, 2000) 
        # an affine operation: y = Wx + b
        #400 input, 120 output
        self.fc2 = nn.Linear(2000, 200)
        self.fc3 = nn.Linear(200, 2)

    def forward(self, x): #defines the method, how the input goes throught the element described above
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 3, stride = 2) #passes through con1, relu, and max pooled
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 3, stride = 2) #now through conv2
        x = F.max_pool2d(F.relu(self.conv3(x)), 3, stride = 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


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
def Mkgrid(inputs, classes, path):
    """Makes a grid with a dataloade batch, takes no argument,requires Imshow"""
    out = torchvision.utils.make_grid(inputs)
    half = round(len(classes)/2) -1
    titles = [class_names[x].split("_")[1][0:3] for x in classes]
    titles[half] = "".join([titles[half], "\n"])
    title_joined = ", ".join(titles)
    
    imshow(out, title=title_joined)
    plt.savefig(path)
#----------------------------------------------------------------------------

#Instantiate the model
model_conv = Net96()

#defines the loss function, the optimizer and schduler.
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer,step_size=7, gamma=0.1 )

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

print("net set")

model_conv.load_state_dict(torch.load("trained_RI_2fam1.pt"))

print("parameter loaded")

model_conv.eval()
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
pred_clarkii = 0 
with torch.no_grad():
    ngrid = 1
    for images, labels, paths in dataset_loader_t["val"]:
        outputs = model_conv(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        #print(c)
        #print(labels)
        #print(predicted)
        #gridpath = "grid%d.jpg" %ngrid
        #if 2 in predicted:
        #    Mkgrid(images, labels, gridpath)
        #    ngrid +=1 
        for i in range(16):
            
            try:                 
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                if predicted[i] == 0:
                    pred_clarkii+=1
                if c[i].item() == 0:
                    print(paths[i])
                    pass
            except:
                pass

correct_total = 0
total = 0  

classes = class_names
for i in range(2):
    correct_total += class_correct[i]
    total += class_total[i]

    print('Accuracy of %5s : %d/%d = %2d %%' % (
        classes[i], class_correct[i],class_total[i], 100 * class_correct[i] / class_total[i]))


elapsed = time.time() -start

print("time elapsed:")
print(elapsed)
print("global accuracy")
print(correct_total/total)
print("#predicted clarkii")
print(pred_clarkii)


