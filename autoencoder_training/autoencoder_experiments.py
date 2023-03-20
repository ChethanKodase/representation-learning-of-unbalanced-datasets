import torch

import torchvision
from torchvision import transforms, datasets
from torchvision.datasets import FashionMNIST
import matplotlib
import matplotlib.pyplot as plt
import os
from torch import nn
import torch.nn.functional as F
import numpy as np
from datasets import getDataset, get_train_test_datasets_and_data_in_batches, get_shuffeled_labels_after_imbalancing, get_dataset_class_stats

from dataset_imbalancing import create_data_imbalance

from models import AE, CNN_AE_fmnist
from tqdm import tqdm 
from activations import Sin
from train import train_MLPAE, train_AEREG, train_CNN_AE_fmnist, train_ContraAE
from loss_functions import jacobian_regularized_loss

torch.manual_seed(0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## For saving models and plots ##

# for creating imbalance among the classes in training and test data 

classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]



number_of_classes =len(class_labels)
majority_class_index = 9   # set which is the ,majority class
majority_class_frac = 0.9  
general_class_frac = 0.1



train_class_fracs = [general_class_frac for i in range(number_of_classes)]
train_class_fracs[majority_class_index] = majority_class_frac
test_class_fracs = [general_class_frac for i in range(10)]

print('train_class_fracs', train_class_fracs )

print('test_class_fracs', test_class_fracs )

set_batch_size = 200



train_batches, test_batches, no_channels, dx, dy = get_train_test_datasets_and_data_in_batches(train_class_fracs, test_class_fracs, set_batch_size, dataset = "FashionMNIST")



# To check the population of different classes in train and test datasets


dataset = "FashionMNIST"

get_dataset_class_stats(train_class_fracs, test_class_fracs, class_labels, dataset)




# Common feed-foreward hyper parameters
layer_size = 100
latent_dim = 4
no_layers = 3
activation = Sin()
no_epochs = 100
lr = 0.0001


# parameters specific for jacobian regularized autoencoders
deg_poly = 21
use_guidance = False
alpha = 0.5
no_samples = 10
reg_nodes_sampled = "legendre"
if(reg_nodes_sampled == "legendre"):
    points = np.polynomial.legendre.leggauss(deg_poly)[0][::-1]
    weights = np.polynomial.legendre.leggauss(deg_poly)[1][::-1]

if(reg_nodes_sampled == "chebyshev"):
    points = np.polynomial.chebyshev.chebgauss(deg_poly)[0][::-1]
    weights = np.polynomial.chebyshev.chebgauss(deg_poly)[1][::-1]


# parameters specific to CNN-AE
cnn_activation = torch.nn.ReLU()
lr_cnn = 1e-3
weight_decay_cnn = 1e-5



train_AE_MLP= False
train_AE_REG = False
train_CNN_AE = False
train_Contra_AE = True

if(train_AE_MLP):
    train_MLPAE(no_epochs, train_batches, no_channels, dx, dy, layer_size, latent_dim, no_layers, activation, lr, device,
                dataset, number_of_classes, majority_class_index, majority_class_frac, general_class_frac, set_batch_size)

if(train_AE_REG):
    train_AEREG(no_epochs, train_batches, no_channels, dx, dy, layer_size, latent_dim, no_layers, activation, lr, device,
                    dataset, number_of_classes, majority_class_index, majority_class_frac, general_class_frac, set_batch_size, 
                    alpha, no_samples, deg_poly, points, reg_nodes_sampled)

if(train_CNN_AE):
    train_CNN_AE_fmnist(no_epochs, train_batches, no_channels, layer_size, latent_dim, no_layers, cnn_activation, lr_cnn, device,
                    dataset, number_of_classes, majority_class_index, majority_class_frac, general_class_frac, set_batch_size, weight_decay_cnn)


# Parameters specific to Contra AE
lam_contra = 1e-2
lr_contra = 1e-3
weight_decay_contra = 1e-5
activation_contra = torch.nn.ReLU()


if(train_Contra_AE):
    train_ContraAE(no_epochs, train_batches, no_channels, dx, dy, layer_size, latent_dim, no_layers, activation, lr_contra, device,
                    dataset, number_of_classes, majority_class_index, majority_class_frac, general_class_frac, set_batch_size, weight_decay_contra, lam_contra)