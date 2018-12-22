from __future__ import print_function, division
import os
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils, datasets
from PIL import Image

plt.ion()

root_dir = '../images'
filelist = []

for dirname, dirnames, filenames in os.walk(root_dir):
    # print path to all file.
    for filename in filenames:
        filelist.append(os.path.join(dirname, filename))

for i_file, file in enumerate(filelist):
	image = Image.open(file)
	if i_file == 0:
		minwidth, minheigth = image.size
		print(minwidth)
	if minwidth > image.size[0]: 
		minwidth = image.size[0]
		print(minwidth)
	if minheigth > image.size[1]: 
		minheigth = image.size[1]

print(minwidth, minheigth)





	