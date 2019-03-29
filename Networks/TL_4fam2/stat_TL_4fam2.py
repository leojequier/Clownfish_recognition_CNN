import time 

start = time.time()
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

#----------------------------------------------------------------
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

#--------------------------------------------------------------
#load the data from file. File has to be organised like:
#file/train/class1
#file/train/class2
#...
#file/val/class1
#file/val/class2
#...

file = "classify4wo"

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

amphiprion_dataset_t = {X: ImageFolderWithPaths(root=os.path.join(file,X),
                                           transform=data_transform[X])
						for X in ["train", "val"]}

#Loader allow to retrieve transformed images by batch, randomly sampled

dataset_loader_t = {X: torch.utils.data.DataLoader(amphiprion_dataset_t[X],
                                             batch_size=16, shuffle=True,
                                             num_workers=4)
					for X in ["train", "val"]}



dataset_sizes = {X: len(amphiprion_dataset_t[X]) for X in ['train', 'val']}

class_names = amphiprion_dataset_t["train"].classes

print(class_names)

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



#----------------------------------------------------------------------------
#Function Calling

model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 5)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

print("resnet18 set")

model_conv.load_state_dict(torch.load("trained_T4F2.pt"))

print("parameter loaded")

model_conv.eval()

print("parameter loaded")
model_conv.eval()
pred_list = []
lab_list = []
class_correct = list(0. for i in range(5))
class_total = list(0. for i in range(5))
pred_correct = list(0. for i in range(5))
pred_total = list(0. for i in range(5))

with torch.no_grad():
    for images, labels, paths in dataset_loader_t["val"]:
        outputs = model_conv(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(16):            
            try:                               
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                lab_list.append(label)

                pred = predicted[i]
                pred_correct[pred] += c[i].item()
                pred_total[pred] += 1
                pred_list.append(pred)
                if c[i].item() == 0:
                    pass
            except:
                pass

correct_total = 0
total = 0  

classes = class_names
for i in range(5):
    correct_total += class_correct[i]
    total += class_total[i]

    print('Well classified  %5s : %d/%d = %2d %%' % (
        classes[i], class_correct[i],class_total[i], 100 * class_correct[i] / class_total[i]))
    print('Well predicted  %5s : %d/%d = %2d %%' % (
        classes[i], pred_correct[i],pred_total[i], 100 * pred_correct[i] / pred_total[i]))


elapsed = time.time() -start

print("time elapsed:")
print(elapsed)
print("global accuracy")
print(correct_total, total, correct_total/total)
print("#predicted")

lab_line = "label," + ",".join(str(int(e)) for e in lab_list)
pred_line = "pred," + ",".join(str(int(e)) for e in pred_list)

with open("pred.csv", "a") as f:
    f.write(lab_line)
    f.write(pred_line)

