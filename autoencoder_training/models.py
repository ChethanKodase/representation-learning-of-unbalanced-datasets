import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch import nn
import os, os.path
from activations import Sin

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(torch.nn.Module):
    """Takes an image and produces a latent vector."""
    def __init__(self, inp_dim, hidden_size, latent_dim, no_layers = 3, activation = F.relu):
        super(Encoder, self).__init__()
        self.activation = activation

        self.lin_layers = nn.ModuleList()
        self.lin_layers.append(nn.Linear(np.prod(inp_dim), hidden_size))
        for i in range(no_layers):
            self.lin_layers.append(nn.Linear(hidden_size, hidden_size))
        self.lin_layers.append(nn.Linear(hidden_size, latent_dim))

        for m in self.lin_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        #x = x.view(x.size(0), -1)
        if len(x.shape)==3:
            x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        else:
            x = x.view(x.size(0), -1)
        for layer in self.lin_layers:
            x = self.activation(layer(x))
        #x = torch.tanh(self.lin_layers[-1](x))
        return x


class Decoder(torch.nn.Module):
    """ Takes a latent vector and produces an image."""
    def __init__(self, latent_dim, hidden_size, inp_dim, no_layers = 3, activation = F.relu):
        super(Decoder, self).__init__()
        self.activation = activation
        self.lin_layers = nn.ModuleList()
        self.lin_layers.append(nn.Linear(latent_dim, hidden_size))
        for i in range(no_layers):
            self.lin_layers.append(nn.Linear(hidden_size, hidden_size))
        self.lin_layers.append(nn.Linear(hidden_size, np.prod(inp_dim)))

        for m in self.lin_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.lin_layers[:-1]:
            x = self.activation(layer(x))
        x = self.lin_layers[-1](x)
        #x = torch.sigmoid(x) # squash into [0,1]
        return x


class AE(torch.nn.Module):
    def __init__(self, inp_dim, hidden_size, latent_dim, no_layers, activation):
        super(AE, self).__init__()
        self.encoder = Encoder(inp_dim, hidden_size, latent_dim, no_layers, activation)
        self.decoder = Decoder(latent_dim, hidden_size, inp_dim, no_layers, activation)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x




class CNN_AE_fmnist(nn.Module):
    def __init__(self, latent_dim, no_channels, activation):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(no_channels, 16, 3, stride=2, padding=1),  #N, 16, 16, 16
            activation,
            nn.Conv2d(16, 32, 3, stride=2, padding=1),   #N, 32, 8, 8
            activation,
            nn.Conv2d(32, 16, 8, stride=2, padding=1),    #N, 64, 2, 2
            activation,
            nn.Conv2d(16, 8, 2, stride=2, padding=1),    #N, 64, 1, 1
            nn.Flatten(1,-1),
            nn.Linear(8*2*2, latent_dim)

        )

        self.decoder = nn.Sequential(
            
            nn.Linear(latent_dim, 8*2*2),
            nn.Unflatten(1, (8, 2, 2)),
            nn.ConvTranspose2d(8, 16, 2, stride=2, padding=1),  #N, 32, 8, 8
            activation,
            nn.ConvTranspose2d(16, 32, 8, stride=2, padding=1),  #N, 32, 8, 8
            activation,
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),   #N, 16, 16, 16
            activation,
            nn.ConvTranspose2d(16, no_channels, 3, stride=2, padding=1, output_padding=1),   #N, 1, 32, 32
            activation
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Autoencoder_linear_contra_fmnist(nn.Module):
    def __init__(self,latent_dim, no_channels, dx, dy, layer_size, activation):

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dx*dy, layer_size),  #input layer
            activation,
            nn.Linear(layer_size, layer_size),   #h1
            activation,
            nn.Linear(layer_size, layer_size),    #h1
            activation,
            nn.Linear(layer_size, layer_size),    #h1
            activation,
            nn.Linear(layer_size,latent_dim)  # latent layer
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, layer_size),  #input layer
            activation,
            nn.Linear(layer_size, layer_size),   #h1
            activation,
            nn.Linear(layer_size, layer_size),    #h1
            activation,
            nn.Linear(layer_size, layer_size),    #h1
            activation,
            nn.Linear(layer_size, dx*dy),  # latent layer
            activation
        )

    def forward(self, x):
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



class MLP_VAE_fmnist(nn.Module):
    def __init__(self, image_size, h_dim, z_dim, activation):
        super(MLP_VAE_fmnist, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(image_size, h_dim),  #input layer
            activation,
            nn.Linear(h_dim, h_dim),   #h1
            activation,
            nn.Linear(h_dim, h_dim),    #h1
            activation,
            nn.Linear(h_dim, h_dim),    #h1
            activation
            #nn.Linear(100,z_dim)  # latent layer
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            #nn.Linear(z_dim, h_dim),  #input layer
            #nn.ReLU(),
            nn.Linear(h_dim, h_dim),   #h1
            activation,
            nn.Linear(h_dim, h_dim),    #h1
            activation,
            nn.Linear(h_dim, h_dim),    #h1
            activation,
            nn.Linear(h_dim, image_size),  # latent layer
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z.to(device)
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu.to(device), logvar.to(device))
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h.to(device))
        z = self.fc3(z)
        #print('z.shape', z.shape)
        return self.decoder(z), mu, logvar



class CNN_VAE_fmnist(nn.Module):
    def __init__(self, image_channels, no_layers, activation, h_dim, z_dim):
        super(CNN_VAE_fmnist, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 16, 3, stride=2, padding=1),  #N, 16, 16, 16
            activation,
            nn.Conv2d(16, 32, 3, stride=2, padding=1),   #N, 32, 8, 8
            activation,
            nn.Conv2d(32, 16, 8, stride=2, padding=1),    #N, 64, 2, 2
            activation,
            nn.Conv2d(16, 8, 2, stride=2, padding=1),    #N, 64, 1, 1
            nn.Flatten(1,-1),

        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            
            nn.Unflatten(1, (8, 2, 2)),
            nn.ConvTranspose2d(8, 16, 2, stride=2, padding=1),  #N, 32, 8, 8
            activation,
            nn.ConvTranspose2d(16, 32, 8, stride=2, padding=1),  #N, 32, 8, 8
            activation,
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),   #N, 16, 16, 16
            activation,
            nn.ConvTranspose2d(16, image_channels, 3, stride=2, padding=1, output_padding=1),   #N, 1, 32, 32
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z.to(device)
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu.to(device), logvar.to(device))
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h.to(device))
        z = self.fc3(z)
        #print('z.shape', z.shape)
        return self.decoder(z), mu, logvar