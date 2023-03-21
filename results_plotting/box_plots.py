import sys
sys.path.append('./autoencoder_training/')

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

from models import AE, CNN_AE_fmnist, Autoencoder_linear_contra_fmnist, MLP_VAE_fmnist, CNN_VAE_fmnist

from tqdm import tqdm 
from activations import Sin
from train import train_MLPAE, train_AEREG, train_CNN_AE_fmnist, train_ContraAE, train_MLP_VAE, train_CNN_VAE_fmnist
from loss_functions import jacobian_regularized_loss
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path_models = './saved_models/'

classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

majority_class_index = 9   # set which is the ,majority class
majority_class_frac = 0.9  
general_class_frac = 0.1


number_of_classes =len(class_labels)
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


# Common hyper parameters
layer_size = 100
latent_dim = 4
no_layers = 3
no_epochs = 100
inp_dim = [no_channels, dx, dy]


#parameters specific to MLPAE
activation_mlpae = Sin()
lr_mlpae = 0.0001



# parameters specific for jacobian regularized autoencoders
activation_aereg = Sin()
lr_aereg = 0.0001
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

########################################################################################################################
# Loading models
########################################################################################################################


# loading MLPAE
model_mlpae = AE(inp_dim, layer_size, latent_dim, no_layers, activation_mlpae).to(device) # baseline autoencoder
name_mlpae = '_'+"MLP-AE"+'_'+str(no_layers)+'_'+str(layer_size)+'_'+str(latent_dim)+'_'+str(lr_mlpae)+'_'+str(activation_mlpae)+'_'+str(dataset)+'_'+str(number_of_classes)+'_'+str(majority_class_index)+'_'+str(majority_class_frac)+'_'+str(general_class_frac)+'_'+str(no_epochs)+'_'+str(set_batch_size)
model_mlpae.load_state_dict(torch.load(path_models+'/model'+name_mlpae, map_location=device))
########################################################################################################################


# Loading AE-REG
########################################################################################################################
model_aereg = AE(inp_dim, layer_size, latent_dim, no_layers, activation_aereg).to(device) # baseline autoencoder
name_aereg = '_'+"AE-REG"+'_'+str(no_layers)+'_'+str(layer_size)+'_'+str(latent_dim)+'_'+ str(reg_nodes_sampled)+'_'+ str(deg_poly)+'_'+ str(alpha)+'_'+ str(no_samples)+'_'+str(lr_aereg)+'_'+str(activation_aereg)+'_'+str(dataset)+'_'+str(number_of_classes)+'_'+str(majority_class_index)+'_'+str(majority_class_frac)+'_'+str(general_class_frac)+'_'+str(no_epochs)+'_'+str(set_batch_size)
model_aereg.load_state_dict(torch.load(path_models+'/model'+name_aereg, map_location=device))
########################################################################################################################


# Loading CNN-AE
########################################################################################################################
model_cnnae = CNN_AE_fmnist(latent_dim, no_channels, activation_cnn).to(device)
name_cnnae = '_'+"CNN-AE"+'_'+str(no_layers)+'_'+str(layer_size)+'_'+str(latent_dim)+'_'+str(lr_cnn)+'_'+str(activation_cnn)+'_'+str(dataset)+'_'+str(number_of_classes)+'_'+str(majority_class_index)+'_'+str(majority_class_frac)+'_'+str(general_class_frac)+'_'+str(no_epochs)+'_'+str(set_batch_size)+'_'+str(weight_decay_cnn)
model_cnnae.load_state_dict(torch.load(path_models+'/model'+name_cnnae, map_location=device))
########################################################################################################################

# Loading ContraAE
########################################################################################################################
model_contra_ = Autoencoder_linear_contra_fmnist(latent_dim, no_channels, dx, dy, layer_size, activation_contra).to(device)
name_contra = '_'+"ContraAE"+'_'+str(no_layers)+'_'+str(layer_size)+'_'+str(latent_dim)+'_'+str(lr_contra)+'_'+str(activation_contra)+'_'+str(dataset)+'_'+str(number_of_classes)+'_'+str(majority_class_index)+'_'+str(majority_class_frac)+'_'+str(general_class_frac)+'_'+str(no_epochs)+'_'+str(set_batch_size)+'_'+str(weight_decay_contra)+'_'+str(lam_contra)
model_contra_.load_state_dict(torch.load(path_models+'/model'+name_contra, map_location=device))
def model_contra(batch_x):
    batch_x_in = batch_x.reshape(-1, dx*dy).to(device)    
    return model_contra_(batch_x_in)
########################################################################################################################

# Loading MLP-VAE
########################################################################################################################
model_mlpvae_ = MLP_VAE_fmnist(dx*dy, layer_size, latent_dim, activation_mlpvae).to(device)
name_mlpvae = '_'+"MLP-VAE"+'_'+str(no_layers)+'_'+str(layer_size)+'_'+str(latent_dim)+'_'+str(lr_mlpvae)+'_'+str(activation_mlpvae)+'_'+str(dataset)+'_'+str(number_of_classes)+'_'+str(majority_class_index)+'_'+str(majority_class_frac)+'_'+str(general_class_frac)+'_'+str(no_epochs)+'_'+str(set_batch_size)
model_mlpvae_.load_state_dict(torch.load(path_models+'/model'+name_mlpvae, map_location=device))
def model_mlpvae(batch_x):
    batch_x_in = batch_x.reshape(-1, dx*dy).to(device)    
    recon, _, _ = model_mlpvae_(batch_x_in)
    return recon
########################################################################################################################

# Loading CNN-VAE
########################################################################################################################

model_cnnvae_ = CNN_VAE_fmnist(no_channels, no_layers, activation_cnn_vae, h_dim_cnn_vae, z_dim=latent_dim).to(device)
name_cnnvae = '_'+"CNN-VAE"+'_'+str(no_layers)+'_'+str(layer_size)+'_'+str(latent_dim)+'_'+str(lr_cnn_vae)+'_'+str(activation_cnn_vae)+'_'+str(dataset)+'_'+str(number_of_classes)+'_'+str(majority_class_index)+'_'+str(majority_class_frac)+'_'+str(general_class_frac)+'_'+str(no_epochs)+'_'+str(set_batch_size)
model_cnnvae_.load_state_dict(torch.load(path_models+'/model'+name_cnnvae, map_location=device))

def model_cnnvae(batch_x):
    recon, _, _ = model_cnnvae_(batch_x)
    return recon




test_samples = test_batches.reshape(test_batches.shape[0]*test_batches.shape[1], no_channels, dx, dy).to(device)

reconstructions = model_cnnvae(test_samples).view(test_samples.size())
reconstructions = reconstructions[:50]

print('reconstruction.shape', reconstructions.shape)


for i in range(len(reconstructions)):
    plt.imshow(reconstructions[i][0].cpu().detach())
    plt.savefig('/home/ramana44/representation-learning-of-unbalanced-datasets/experiments/rec'+str(i)+'.png')
    plt.close()

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from matplotlib.colors import Normalize


test_samples_normal = Normalize()(test_samples)
testImage_normal = Normalize()(test_samples)

