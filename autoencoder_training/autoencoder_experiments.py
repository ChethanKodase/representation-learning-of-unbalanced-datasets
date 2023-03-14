from datasets import getDataset, get_train_test_datasets_and_data_in_batches, get_shuffeled_labels_after_imbalancing, get_dataset_class_stats
import torch

import torchvision
from torchvision import transforms, datasets
from torchvision.datasets import FashionMNIST
import matplotlib
import matplotlib.pyplot as plt

from torch import nn
import torch.nn.functional as F

from dataset_imbalancing import create_data_imbalance

from models import AE
from tqdm import tqdm 
from activations import Sin
torch.manual_seed(0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# for creating imbalance among the classes in training and test data 
train_class_fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9]
test_class_fracs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
set_batch_size = 200

classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

train_batches, test_batches, no_channels, dx, dy = get_train_test_datasets_and_data_in_batches(train_class_fracs, test_class_fracs, set_batch_size, dataset = "FashionMNIST")



# To check the population of different classes in train and test datasets
get_dataset_class_stats(train_class_fracs, test_class_fracs, class_labels, dataset = "FashionMNIST")


inp_dim = [no_channels, dx, dy]

# Hyper parameters
hidden_size = 100
latent_dim = 4
no_layers = 3
activation = Sin()
no_epochs = 100
lr = 0.0001

# Available models

models_avail = ["MLP_AE", "AE_REG"]
select_model = "MLP_AE"


if(select_model == "MLP_AE"):
    model = AE(inp_dim, hidden_size, latent_dim, no_layers, activation).to(device) # baseline autoencoder
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
elif(select_model == "AE_REG"):
    model = AE(inp_dim, hidden_size, latent_dim, no_layers, activation).to(device) # jacobian regularised autoencoder
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


loss_C1 = torch.FloatTensor([0.]).to(device) 

for epoch in tqdm(range(no_epochs)):
    for inum, batch_x in enumerate(train_batches):

        batch_x = batch_x.to(device)
        reconstruction = model(batch_x).view(batch_x.size())
        loss_reconstruction = F.mse_loss(reconstruction, batch_x)

        optimizer.zero_grad()
        loss_reconstruction.backward()
        optimizer.step()

    print('loss_reconstruction', loss_reconstruction)

