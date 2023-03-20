import torch

from torch.autograd import Variable

import numpy as np

import torch.nn.functional as F


def sampleLegendreNodes(batch_size, latent_dim, points, n):
    Gamma = np.random.randint(low=0, high=n, size=(latent_dim, batch_size))
    PP = np.zeros((latent_dim, batch_size))
    for i in range(batch_size):
        for j in range(latent_dim):
            PP[j, i] = points[int(Gamma[j, i])]

    PP = PP.transpose()
            
    return PP


def computeC1Loss_upd(cheb_nodes, model, device, guidanceTerm = True):
    noNodes, szLatDim = cheb_nodes.shape
    I = torch.eye(szLatDim).to(device) # extract values of all minor diagonals (I = 1) 
    f = lambda x: model.encoder(model.decoder(x.to(device))) # loop through autoencoder

    loss_C1_arr = torch.zeros(noNodes).to(device)

    Jac_array = torch.tensor([]).to(device)
    inum = 0
    for node_points in cheb_nodes:
        node_points = torch.reshape(node_points, (1, szLatDim))

        Jac = torch.autograd.functional.jacobian(f, node_points.to(device), create_graph = True).squeeze() # compute Jacobian

        loss_C1 = torch.mean((Jac - I)**2)
        if(guidanceTerm):
            min_diag_val = torch.mean((torch.diagonal(Jac, dim1 = 0, dim2 = 1) - 1)**2)
            loss_C1 = loss_C1 + min_diag_val
        loss_C1_arr[inum] = loss_C1
        Jac_array = torch.cat((Jac_array, Jac.unsqueeze(0)))
        inum += 1        
    Jac_array = Jac_array.unsqueeze(1)

    return torch.mean(loss_C1_arr), Jac_array

def jacobian_regularized_loss(model_reg, batch_x, alpha, no_samples, deg_poly,  latent_dim, points, device, guidanceTerm):

    nodes_subsample_np = sampleLegendreNodes(no_samples, latent_dim, points, deg_poly)
    nodes_subsample = torch.FloatTensor(nodes_subsample_np).to(device)

    loss_C1, Jac = computeC1Loss_upd(nodes_subsample, model_reg, device, guidanceTerm) # guidance term

    reconstruction = model_reg(batch_x).view(batch_x.size())
    loss_reconstruction = torch.nn.MSELoss()(reconstruction, batch_x)
    total_loss = (1.- alpha)*loss_reconstruction + alpha*loss_C1

    return total_loss, loss_reconstruction, loss_C1




def contra_loss_function(W, x, recons_x, h, lam):

    mseLoss_nn = torch.nn.MSELoss()
    mse_loss = torch.nn.BCELoss(size_average = False)
    mse = mseLoss_nn(recons_x, x)
    dh = h * (1 - h) 
    w_sum = torch.sum(Variable(W)**2, dim=1)
    w_sum = w_sum.unsqueeze(1) 
    contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
    return mse + contractive_loss.mul_(lam), contractive_loss.mul_(lam)


def vae_loss_fn(recon_x, x, mu, logvar):

    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    #BCE = torch.nn.MSELoss()(recon_x, x)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD