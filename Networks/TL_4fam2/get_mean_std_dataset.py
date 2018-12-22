import torch
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt


def get_mean_std(path_to_dataset):

	transformation =  transforms.Compose([
	      transforms.CenterCrop(224),
	      transforms.ToTensor()#,
	      #transforms.Normalize(mean=[0.485, 0.456, 0.406],
	      #                     std=[0.229, 0.224, 0.225])
	])
	
	dataset = datasets.ImageFolder(root=path_to_dataset,
	                                transform=transformation)
	
	
	loader = torch.utils.data.DataLoader(dataset,
	                                    batch_size=10, shuffle=False,
	                                    num_workers=1)
	
	mean = 0.
	std = 0.
	nb_samples = 0.
	
	for data in loader:
	    data = data[0]
	    batch_samples = data.size(0)
	    data = data.view(batch_samples, data.size(1), -1)
	    mean += data.mean(2).sum(0)
	    std += data.std(2).sum(0)
	    nb_samples += batch_samples


	mean /= nb_samples
	std /= nb_samples

	to_return = {"mean":mean, "stderr":std}

	return(to_return)

