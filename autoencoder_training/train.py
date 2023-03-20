import torch



import matplotlib.pyplot as plt
import os

from models import AE, CNN_AE_fmnist
from tqdm import tqdm 

from loss_functions import jacobian_regularized_loss


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
    name = '_'+"MLP-AE"+'_'+str(no_layers)+'_'+str(layer_size)+'_'+str(latent_dim)+'_'+str(lr)+'_'+str(activation)+'_'+str(dataset)+'_'+str(number_of_classes)+'_'+str(majority_class_index)+'_'+str(majority_class_frac)+'_'+str(general_class_frac)+'_'+str(no_epochs)+'_'+str(set_batch_size)
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
    name = '_'+"AE-REG"+'_'+str(no_layers)+'_'+str(layer_size)+'_'+str(latent_dim)+'_'+ str(reg_nodes_sampled)+'_'+ str(deg_poly)+'_'+ str(alpha)+'_'+ str(no_samples)+'_'+str(lr)+'_'+str(activation)+'_'+str(dataset)+'_'+str(number_of_classes)+'_'+str(majority_class_index)+'_'+str(majority_class_frac)+'_'+str(general_class_frac)+'_'+str(no_epochs)+'_'+str(set_batch_size)
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
    name = '_'+"CNN-AE"+'_'+str(no_layers)+'_'+str(layer_size)+'_'+str(latent_dim)+'_'+str(lr_cnn)+'_'+str(activation)+'_'+str(dataset)+'_'+str(number_of_classes)+'_'+str(majority_class_index)+'_'+str(majority_class_frac)+'_'+str(general_class_frac)+'_'+str(no_epochs)+'_'+str(set_batch_size)+'_'+str(weight_decay)
    torch.save(model.state_dict(), path_models+'/model'+name)
    
    plt.plot(list(range(0,no_epochs)), loss_array)
    plt.xlabel("epoch")
    plt.ylabel("CNN-AE"+" loss")
    plt.savefig(path_plots+'/loss'+name+'.png')