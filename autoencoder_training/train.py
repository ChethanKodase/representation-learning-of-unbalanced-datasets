import torch



import matplotlib.pyplot as plt
import os

from models import AE, CNN_AE_fmnist, Autoencoder_linear_contra_fmnist, MLP_VAE_fmnist, CNN_VAE_fmnist
from tqdm import tqdm 

from loss_functions import jacobian_regularized_loss, contra_loss_function, vae_loss_fn
from torch.autograd import Variable


path_models = './saved_models/'
path_plots = './saved_plots/'


def train_MLPAE(no_epochs, train_batches, no_channels, dx, dy, layer_size, latent_dim, no_layers, activation, lr, device,
                 dataset, number_of_classes, majority_class_index, majority_class_frac, general_class_frac, set_batch_size):

    inp_dim = [no_channels, dx, dy]
    model = AE(inp_dim, layer_size, latent_dim, no_layers, activation).to(device) # baseline autoencoder
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = torch.nn.MSELoss()

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
    name = '_'+"MLP-AE"+'_'+str(no_layers)+'_'+str(layer_size)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(activation)+'_'+str(dataset)+'_'+str(number_of_classes)+'_'+str(majority_class_index)+'_'+str(majority_class_frac)+'_'+str(no_epochs)+'_'+str(set_batch_size)
    torch.save(model.state_dict(), path_models+'/model'+name)
    
    plt.plot(list(range(0,no_epochs)), loss_array)
    plt.xlabel("epoch")
    plt.ylabel("MLP-AE"+" loss")
    plt.savefig(path_plots+'/loss'+name+'.png')






def train_AEREG(no_epochs, train_batches, no_channels, dx, dy, layer_size, latent_dim, no_layers, activation, lr, device,
                 dataset, number_of_classes, majority_class_index, majority_class_frac, general_class_frac, set_batch_size, 
                 alpha, no_samples, deg_poly, points, reg_nodes_sampled):

    inp_dim = [no_channels, dx, dy]
    model = AE(inp_dim, layer_size, latent_dim, no_layers, activation).to(device) # baseline autoencoder
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = torch.nn.MSELoss()

    tot_loss_array = []
    recon_loss_array = []
    reg_loss_array = []
    for epoch in tqdm(range(no_epochs)):
        epoch_tot_loss_array = []
        epoch_recon_loss_array = []
        epoch_reg_loss_array = []

        for inum, batch_x in enumerate(train_batches):

            batch_x = batch_x.to(device)
            #reconstruction = model(batch_x).view(batch_x.size())
            #loss_reconstruction = loss_function(reconstruction, batch_x)

            total_loss, loss_reconstruction, jac_reg_loss = jacobian_regularized_loss(model, batch_x, alpha, no_samples, deg_poly,  latent_dim, points, device, guidanceTerm = False)

            epoch_tot_loss_array.append(total_loss.item())
            epoch_recon_loss_array.append(loss_reconstruction.item())
            epoch_reg_loss_array.append(jac_reg_loss.item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        avg_tot_loss = sum(epoch_tot_loss_array)/len(epoch_tot_loss_array)
        avg_recon_loss = sum(epoch_recon_loss_array)/len(epoch_recon_loss_array)
        avg_reg_loss = sum(epoch_reg_loss_array)/len(epoch_reg_loss_array)

        tot_loss_array.append(avg_tot_loss)
        recon_loss_array.append(avg_recon_loss)
        reg_loss_array.append(avg_reg_loss)

        print("total loss : ", avg_tot_loss )
        print()
        print("reconstruction loss : ", avg_recon_loss )
        print()
        print("jacobian regularization loss : ", avg_reg_loss )

    os.makedirs(path_models, exist_ok=True)
    name = '_'+"AE-REG"+'_'+str(no_layers)+'_'+str(layer_size)+'_'+str(latent_dim)+'_'+ str(reg_nodes_sampled)+'_'+ str(deg_poly)+'_'+ str(alpha)+'_'+ str(no_samples)+'_'+str(lr)+'_'+str(activation)+'_'+str(dataset)+'_'+str(number_of_classes)+'_'+str(majority_class_index)+'_'+str(majority_class_frac)+'_'+str(no_epochs)+'_'+str(set_batch_size)
    torch.save(model.state_dict(), path_models+'/model'+name)
    
    plt.plot(list(range(0,no_epochs)), tot_loss_array)
    plt.xlabel("epoch")
    plt.ylabel("AE-REG"+"total loss")
    plt.savefig(path_plots+'/total_loss'+name+'.png')
    plt.close()


    plt.plot(list(range(0,no_epochs)), recon_loss_array)
    plt.xlabel("epoch")
    plt.ylabel("AE-REG"+" reconstruction loss")
    plt.savefig(path_plots+'/reconstruction_loss'+name+'.png')
    plt.close()


    plt.plot(list(range(0,no_epochs)), reg_loss_array)
    plt.xlabel("epoch")
    plt.ylabel("AE-REG"+" jacobian loss")
    plt.savefig(path_plots+'/jacobian_reg_loss'+name+'.png')
    plt.close()





def train_CNN_AE_fmnist(no_epochs, train_batches, no_channels, layer_size, latent_dim, no_layers, activation, lr_cnn, device,
                 dataset, number_of_classes, majority_class_index, majority_class_frac, general_class_frac, set_batch_size, weight_decay):

    model = CNN_AE_fmnist(latent_dim, no_channels, activation).to(device)
    loss_function = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr =lr_cnn, weight_decay = weight_decay)
    
    loss_array = []
    for epoch in tqdm(range(no_epochs)):
        epoch_loss_array = []
        for inum, batch_x in enumerate(train_batches):

            batch_x = batch_x.to(device)
            recon = model(batch_x)
            loss_reconstruction = loss_function(recon, batch_x)
            epoch_loss_array.append(loss_reconstruction.item())

            optimizer.zero_grad()
            loss_reconstruction.backward()
            optimizer.step()


        avg_loss = sum(epoch_loss_array)/len(epoch_loss_array)
        loss_array.append(avg_loss)

        print("loss : ", avg_loss )

    os.makedirs(path_models, exist_ok=True)
    name = '_'+"CNN-AE"+'_'+str(no_layers)+'_'+str(layer_size)+'_'+str(latent_dim)+'_'+str(lr_cnn)+'_'+str(activation)+'_'+str(dataset)+'_'+str(number_of_classes)+'_'+str(majority_class_index)+'_'+str(majority_class_frac)+'_'+str(no_epochs)+'_'+str(set_batch_size)+'_'+str(weight_decay)
    torch.save(model.state_dict(), path_models+'/model'+name)
    
    plt.plot(list(range(0,no_epochs)), loss_array)
    plt.xlabel("epoch")
    plt.ylabel("CNN-AE"+" loss")
    plt.savefig(path_plots+'/loss'+name+'.png')




def train_ContraAE(no_epochs, train_batches, no_channels, dx, dy, layer_size, latent_dim, no_layers, activation, lr_contra, device,
                 dataset, number_of_classes, majority_class_index, majority_class_frac, general_class_frac, set_batch_size, weight_decay, lam):


    model = Autoencoder_linear_contra_fmnist(latent_dim, no_channels, dx, dy, layer_size, activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr =lr_contra, weight_decay = weight_decay)

    loss_array = []
    for epoch in tqdm(range(no_epochs)):
        epoch_loss_array = []
        for inum, batch_x in enumerate(train_batches):
            batch_x = batch_x.to(device)
            batch_x_in = batch_x.reshape(-1, dx*dy).to(device)
            
            batch_x_in = Variable(batch_x_in)
            recon = model(batch_x_in).view(batch_x.size())
            W = list(model.parameters())[8]
            hidden_representation = model.encoder(batch_x_in)
            loss_reconstruction, testcontraLoss = contra_loss_function(W, batch_x, recon, hidden_representation, lam)
            epoch_loss_array.append(loss_reconstruction.item())

            optimizer.zero_grad()
            loss_reconstruction.backward()
            optimizer.step()

        avg_loss = sum(epoch_loss_array)/len(epoch_loss_array)
        loss_array.append(avg_loss)
        print("loss : ", avg_loss )

    os.makedirs(path_models, exist_ok=True)
    name = '_'+"ContraAE"+'_'+str(no_layers)+'_'+str(layer_size)+'_'+str(latent_dim)+'_'+str(lr_contra)+'_'+str(activation)+'_'+str(dataset)+'_'+str(number_of_classes)+'_'+str(majority_class_index)+'_'+str(majority_class_frac)+'_'+str(no_epochs)+'_'+str(set_batch_size)+'_'+str(weight_decay)+'_'+str(lam)
    torch.save(model.state_dict(), path_models+'/model'+name)
    plt.plot(list(range(0,no_epochs)), loss_array)
    plt.xlabel("epoch")
    plt.ylabel("ContraAE"+" loss")
    plt.savefig(path_plots+'/loss'+name+'.png')



def train_MLP_VAE(no_epochs, train_batches, no_channels, dx, dy, layer_size, latent_dim, no_layers, activation, lr_mlpvae, device,
                 dataset, number_of_classes, majority_class_index, majority_class_frac, general_class_frac, set_batch_size):


    image_size = dx*dy
    z_dim = latent_dim
    no_layers = 3
    layer_size = 100


    model = MLP_VAE_fmnist( image_size, layer_size, z_dim, activation).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_mlpvae) 

    loss_array = []
    for epoch in tqdm(range(no_epochs)):
        epoch_loss_array = []
        for inum, batch_x in enumerate(train_batches):
            images = batch_x.reshape(-1, 32*32)

            recon_images, mu, logvar = model(images.to(device))

            loss_reconstruction, bce, kld = vae_loss_fn(recon_images.to(device), images.to(device), mu.to(device), logvar.to(device))
            epoch_loss_array.append(loss_reconstruction.item())
            
            optimizer.zero_grad()
            loss_reconstruction.backward()
            optimizer.step()

        avg_loss = sum(epoch_loss_array)/len(epoch_loss_array)
        loss_array.append(avg_loss)
        print("loss : ", avg_loss )

    os.makedirs(path_models, exist_ok=True)
    name = '_'+"MLP-VAE"+'_'+str(no_layers)+'_'+str(layer_size)+'_'+str(latent_dim)+'_'+str(lr_mlpvae)+'_'+str(activation)+'_'+str(dataset)+'_'+str(number_of_classes)+'_'+str(majority_class_index)+'_'+str(majority_class_frac)+'_'+str(no_epochs)+'_'+str(set_batch_size)
    torch.save(model.state_dict(), path_models+'/model'+name)
    plt.plot(list(range(0,no_epochs)), loss_array)
    plt.xlabel("epoch")
    plt.ylabel("MLP-VAE"+" loss")
    plt.savefig(path_plots+'/loss'+name+'.png')



def train_CNN_VAE_fmnist(no_epochs, train_batches, no_channels, layer_size, latent_dim, no_layers, activation, lr_cnn_vae, device,
                 dataset, number_of_classes, majority_class_index, majority_class_frac, general_class_frac, set_batch_size, h_dim):

    model = CNN_VAE_fmnist(no_channels, no_layers, activation, h_dim, z_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_cnn_vae) 

    loss_array = []
    for epoch in tqdm(range(no_epochs)):
        epoch_loss_array = []
        for inum, batch_x in enumerate(train_batches):

            recon_images, mu, logvar = model(batch_x.to(device))
            loss_reconstruction, bce, kld = vae_loss_fn(recon_images.to(device), batch_x.to(device), mu.to(device), logvar.to(device))

            optimizer.zero_grad()
            loss_reconstruction.backward()
            optimizer.step()
            epoch_loss_array.append(loss_reconstruction.item())


        avg_loss = sum(epoch_loss_array)/len(epoch_loss_array)
        loss_array.append(avg_loss)

        print("loss : ", avg_loss )

    os.makedirs(path_models, exist_ok=True)
    name = '_'+"CNN-VAE"+'_'+str(no_layers)+'_'+str(layer_size)+'_'+str(latent_dim)+'_'+str(lr_cnn_vae)+'_'+str(activation)+'_'+str(dataset)+'_'+str(number_of_classes)+'_'+str(majority_class_index)+'_'+str(majority_class_frac)+'_'+str(no_epochs)+'_'+str(set_batch_size)
    torch.save(model.state_dict(), path_models+'/model'+name)
    
    plt.plot(list(range(0,no_epochs)), loss_array)
    plt.xlabel("epoch")
    plt.ylabel("CNN-VAE"+" loss")
    plt.savefig(path_plots+'/loss'+name+'.png')