# Clownfish recognition CNN or A computationnal approach to finding nemo
## About
1st semester project during my Molecular Life Science (MLS) Master at the University of Lausanne. This programm uses Convolutionnal Neural Networks to classify pictures of clown fishes (Amphiprion sp.).

The project is coded in python (version 3.7.1). The list of required packages are in ./'Packages_used.txt'. 

You can start by taking a look at the final report of my project, ./'report - Léonard Jequier.pdf'. 

## Quick start:

* Install python 3.7.1 & required dependancies (./Packages_used.txt).
* Go to one of the network folder: ```cd Networks/TL_2fam1/ ```
* Train the network: ```python netTL_2fam1.py```
* Look at the results: ```python stat_TL_2fam1.py```
* Extract the change in loss and accuracy during the training process from the text output to a csv at Clownfish_recognition_CNN/Analyse/your_results: ```python extract_info_csv_train.py```

Otherwise, the different folders contains :

* Networks : organized by architecture and type of initialization. TL = Transfer Learning
RI = random initialization. 2fam = two families, 4fam = four families, 4fam2 = four families + 1 outgroup constituted of random pictures of other families of Amphiprion sp.
If you don't know where to start, take a look at TL_2fam1. Each folder contains:
  - **net*.py:** Defines the architecture of the networks and trains them. By default, they will train on the ./Datasets/mini, you can change that by selecting setting an other directory in the variable "file" at the beggining of the script. 
  - **stat*.py:** Tests the resulting network on the validation dataset. 
  - **extract_info_csv_train:** Transforms the text log of the training process in a csv file for an easier analysis of the chage in loss and accuracy. 
  - **script*.sh:** Launches the training on the cluster
  - **get_mean_std_dataset-py:** Calculates the mean and standard deviation of the colors in the pictures for color normalization. 

* **Analyse:** contains the csv with the change in loss and accuracy during the training of the networks and the R script used to create the plots in teh report. 
* **Datasets:** contains the "mini" and "classify2" datasets, feel free to contact me to have access to the other datasets. 
* **Download images:** Contains the scripts used to download the clownfish images from Fishbase (www.fishbase.org) , Fishpix (http://fishpix.kahaku.go.jp/fishimage-e/index.html) and Gbif (https://www.gbif.org/). Might not work anymore if the html structure of the website changed since december 2018.
