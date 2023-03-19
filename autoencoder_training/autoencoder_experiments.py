import torch

import torchvision
from torchvision import transforms, datasets
from torchvision.datasets import FashionMNIST
import matplotlib
import matplotlib.pyplot as plt
import os
from torch import nn
import torch.nn.functional as F

from datasets import getDataset, get_train_test_datasets_and_data_in_batches, get_shuffeled_labels_after_imbalancing, get_dataset_class_stats

from dataset_imbalancing import create_data_imbalance

from models import AE
from tqdm import tqdm 
from activations import Sin
torch.manual_seed(0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## For saving models and plots ##

path_models = './saved_models/'
path_plots = './saved_plots/'

# for creating imbalance among the classes in training and test data 

classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]



number_of_classes =len(class_labels)
majority_class_index = 9
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


inp_dim = [no_channels, dx, dy]

# Hyper parameters
layer_size = 100
latent_dim = 4
no_layers = 3
activation = Sin()
no_epochs = 100
lr = 0.0001

# Available models

models_avail = ["MLP_AE", "AE_REG"]
select_model = "MLP_AE"



if(select_model == "MLP_AE"):
    model = AE(inp_dim, layer_size, latent_dim, no_layers, activation).to(device) # baseline autoencoder
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = torch.nn.MSELoss()
elif(select_model == "AE_REG"):
    model = AE(inp_dim, layer_size, latent_dim, no_layers, activation).to(device) # jacobian regularised autoencoder
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


loss_C1 = torch.FloatTensor([0.]).to(device) 

loss_array = []
for epoch in tqdm(range(no_epochs)):
    epoch_loss_array = []
    for inum, batch_x in enumerate(train_batches):

        batch_x = batch_x.to(device)
        reconstruction = model(batch_x).view(batch_x.size())
        loss_reconstruction = loss_function(reconstruction, batch_x)
        epoch_loss_array.append(loss_reconstruction.item())

        optimizer.zero_grad()
        loss_reconstruction.backward()
        optimizer.step()

    avg_loss = sum(epoch_loss_array)/len(epoch_loss_array)
    loss_array.append(avg_loss)
    print("loss : ", avg_loss )



os.makedirs(path_models, exist_ok=True)
name = '_'+select_model+'_'+str(no_layers)+'_'+str(layer_size)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(activation)+'_'+str(dataset)+'_'+str(number_of_classes)+'_'+str(majority_class_index)+'_'+str(majority_class_frac)+'_'+str(general_class_frac)
torch.save(model.state_dict(), path_models+'/model'+name)

plt.plot(list(range(0,no_epochs)), loss_array)
plt.xlabel("epoch")
plt.ylabel(select_model+" loss")
plt.savefig(path_plots+'/loss'+name+'.png')