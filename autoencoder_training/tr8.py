

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
from train import train_MLPAE, train_AEREG, train_CNN_AE_fmnist, train_ContraAE, train_MLP_VAE, train_CNN_VAE_fmnist
from loss_functions import jacobian_regularized_loss
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# for creating imbalance among the classes in training and test data 

classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# total size of each class 6000 in training set
# worst / maximum bias that you can inflict on the dataset : 6000 images
# images in the majority class and 0 images in rest of the classes

# So total dataset size would be 6000,
# The training data amount is 6,000/60,000=0 .1


majority_class_index = 9   # set which is the ,majority class
majority_class_frac = 0.65
general_class_frac = (1.0 - majority_class_frac)/(len(class_labels)-1)


number_of_classes =len(class_labels)
train_class_fracs = [general_class_frac for i in range(number_of_classes)]
train_class_fracs[majority_class_index] = majority_class_frac
test_class_fracs = [general_class_frac for i in range(10)]


general_class_frac_in_test = 0.1
test_class_fracs = [general_class_frac_in_test for i in range(number_of_classes)]
#test_class_fracs[majority_class_index] = 0.2 # no bias in the test data

print('train_class_fracs', train_class_fracs )
print('test_class_fracs', test_class_fracs )

set_batch_size = 200
train_batches, test_batches, no_channels, dx, dy = get_train_test_datasets_and_data_in_batches(train_class_fracs, test_class_fracs, set_batch_size, dataset = "FashionMNIST")


# To check the population of different classes in train and test datasets
dataset = "FashionMNIST"
get_dataset_class_stats(train_class_fracs, test_class_fracs, class_labels, dataset)


# Common hyper parameters
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
activation_cnn = torch.nn.ReLU()
lr_cnn = 1e-3
weight_decay_cnn = 1e-5


# Parameters specific to Contra AE
lam_contra = 1e-2
lr_contra = 1e-3
weight_decay_contra = 1e-5
activation_contra = torch.nn.ReLU()



# Parameters specific to MLP-VAE
lr_mlpvae = 1e-3
activation_mlpvae = torch.nn.ReLU()


# Parameters specific to CNN-VAE
lr_cnn_vae = 0.0001
h_dim_cnn_vae = 8*2*2
activation_cnn_vae = torch.nn.ReLU()



train_AE_MLP= True
train_AE_REG = True
train_CNN_AE = True
train_Contra_AE = True
train_MLPVAE= True
train_CNN_VAE = True

if(train_AE_MLP):
    train_MLPAE(no_epochs, train_batches, no_channels, dx, dy, layer_size, latent_dim, no_layers, activation, lr, device,
                dataset, number_of_classes, majority_class_index, majority_class_frac, general_class_frac, set_batch_size)

if(train_AE_REG):
    train_AEREG(no_epochs, train_batches, no_channels, dx, dy, layer_size, latent_dim, no_layers, activation, lr, device,
                    dataset, number_of_classes, majority_class_index, majority_class_frac, general_class_frac, set_batch_size, 
                    alpha, no_samples, deg_poly, points, reg_nodes_sampled)

if(train_CNN_AE):
    train_CNN_AE_fmnist(no_epochs, train_batches, no_channels, layer_size, latent_dim, no_layers, activation_cnn, lr_cnn, device,
                    dataset, number_of_classes, majority_class_index, majority_class_frac, general_class_frac, set_batch_size, weight_decay_cnn)

if(train_Contra_AE):
    train_ContraAE(no_epochs, train_batches, no_channels, dx, dy, layer_size, latent_dim, no_layers, activation_contra, lr_contra, device,
                    dataset, number_of_classes, majority_class_index, majority_class_frac, general_class_frac, set_batch_size, weight_decay_contra, lam_contra)


if(train_MLPVAE):
    train_MLP_VAE(no_epochs, train_batches, no_channels, dx, dy, layer_size, latent_dim, no_layers, activation_mlpvae, lr_mlpvae, device,
                    dataset, number_of_classes, majority_class_index, majority_class_frac, general_class_frac, set_batch_size)

if(train_CNN_VAE):
    train_CNN_VAE_fmnist(no_epochs, train_batches, no_channels, layer_size, latent_dim, no_layers, activation_cnn_vae, lr_cnn_vae, device,
                    dataset, number_of_classes, majority_class_index, majority_class_frac, general_class_frac, set_batch_size, h_dim_cnn_vae)