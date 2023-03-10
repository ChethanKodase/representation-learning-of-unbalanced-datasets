from datasets import getDataset
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_loader, test_loader, no_channels, dx, dy = getDataset(dataset = "FashionMNIST", batch_size = 60000)  # FashionMNIST , MNIST
training_data, training_labels = next(iter(train_loader))


unbalancing_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9]
set_batch_size = 200


imbalanced_dataset, imblcnd_shffld_trng_lbls = create_data_imbalance(training_data, training_labels, unbalancing_fractions)


print("Check class populations after shuffling")
for i in range(10):
    print('torch.where(imbal_class_inds_mrgd_shffld=='+str(i)+')[0].shape', torch.where(imblcnd_shffld_trng_lbls==i)[0].shape)    
print()




print('imbalanced_dataset.shape', imbalanced_dataset.shape)

batched_imbalanced_dataset = imbalanced_dataset.reshape(imbalanced_dataset.shape[0]//set_batch_size, set_batch_size, no_channels, dx, dy )


torch.manual_seed(0)
inp_dim = [no_channels, dx, dy]

print('inp_dim', inp_dim)

hidden_size = 100
latent_dim = 4
no_layers = 3
activation = Sin()
no_epochs = 100
lr = 0.0001

ae_REG = AE(inp_dim, hidden_size, latent_dim, 
                    no_layers, activation).to(device) # regularised autoencoder

mlp_AE = AE(inp_dim, hidden_size, latent_dim, 
                    no_layers, activation).to(device) # baseline autoencoder

optimizer_mlp_AE = torch.optim.Adam(mlp_AE.parameters(), lr=lr)


loss_C1 = torch.FloatTensor([0.]).to(device) 

for epoch in tqdm(range(no_epochs)):
    for inum, batch_x in enumerate(batched_imbalanced_dataset):

        batch_x = batch_x.to(device)
        reconstruction = mlp_AE(batch_x).view(batch_x.size())
        loss_reconstruction = F.mse_loss(reconstruction, batch_x)

        optimizer_mlp_AE.zero_grad()
        loss_reconstruction.backward()
        optimizer_mlp_AE.step()

    print('loss_reconstruction', loss_reconstruction)



'''
FashionMNIST labels

0: T-shirt/top
1: Trouser
2: Pullover
3: Dress
4: Coat
5: Sandal
6: Shirt
7: Sneaker
8: Bag
9: Ankle boot

'''
