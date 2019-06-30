from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import pickle

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ion() 

import get_mean_std_dataset as gm
#-------------------------------------------------------------------
#Loading data. 
#Chose source file
file = "../../Datasets/mini"

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

amphiprion_dataset_t = {X: datasets.ImageFolder(root=os.path.join(file,X),
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

#-------------------------------------------------------------------
#Define a few useful function 

#Take a tensor, the format in which the images are "stored" in the loader
#and add it to a plot. 
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


# Make a grid from the bach images.
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


#------------------------------------------------------------------
#Define the architecture of the network.

class Net96(nn.Module):
    def __init__(self): #only defines the architecture of the net
        super(Net96, self).__init__()
        self.conv1 = nn.Conv2d(3,70, 5, stride = 1)
        # 3 input image channel, 
        # 70 output 
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

#Instantiate the model
net = Net96()

#defines the loss function, the optimizer and schduler.
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer,step_size=7, gamma=0.1 )

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs): #50 x
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
            
            if phase == "train":
                loss_train.append(epoch_loss)
                acc_train.append(epoch_acc)
            else:
                loss_val.append(epoch_loss)
                acc_val.append(epoch_acc)

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


#creates a list to  store the loss value at each Epoch
loss_train = []
loss_val = []
acc_train = []
acc_val = []

#train the instanciated model 
train_model(net, criterion, optimizer, scheduler, num_epochs=50)

#save the best parameters of the model.
torch.save(net.state_dict(), "trainedRI_2fam1.pt")

#save the loss and accuracy
pickle.dump(loss_train, open("loss_train", "wb"))
pickle.dump(loss_val, open("loss_val", "wb"))

pickle.dump(acc_train, open("loss_val", "wb"))
pickle.dump(acc_val, open("acc_val", "wb"))

# plot the loss
fig, ax = plt.subplots()
ax.plot(range(len(loss_train)), loss_train)

ax.set(xlabel='Epoch', ylabel='loss',
       title='Training loss')
ax.grid()

fig.savefig("training_process.png")



fig, ax = plt.subplots()
ax.plot(range(len(loss_val)), loss_val)

ax.set(xlabel='Epoch', ylabel='loss',
       title='Validation loss')
ax.grid()

fig.savefig("validation_loss.png")

