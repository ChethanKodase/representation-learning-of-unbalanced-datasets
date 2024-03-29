import sys
#sys.path.append('/home/ramana44/autoencoder-regularisation-')

import math
import numpy as np
from models import AE
import matplotlib.pyplot as plt
import torch.nn as nn
from activations import Sin
import torch


from torchvision import datasets, transforms

transform = transforms.ToTensor()

from models_for_circle import ConvoAE, Autoencoder_linear, VAE_mlp_circle_new, ConvVAE_circle
from loss_functions import contractive_loss_function, loss_fn_mlp_vae, loss_fn_cnn_vae
from torch.autograd import Variable

pi = math.pi


def PointsInCircum(r,n=100):
    return [(math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r) for x in range(0,n)]

def ImbalancingPointsInCircum(r,n=100):
    a = [n/2+0.1*k for k in range(5)]
    b = [n/2-0.1*j for j in range(5)]
    ab = a+b
    return [(math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r) for x in ab]

def PointsInCircumNDim(points, transform_to_nD):
    circle_nD = np.matmul(points, transform_to_nD)
    return circle_nD

##################################################################################################################
from random import seed
from random import randint
import random
import os


seed(1)
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(2342)
#previous seeds 2342
##################################################################################################################

##################################################################################################################
I = torch.eye(5)
#print(I)
##################################################################################################################

n = 100
points = PointsInCircum(1.,100)
arr_points = np.array(points)
plt.scatter(arr_points[:,0], arr_points[:,1])
plt.grid(True)
plt.show()
#plt.savefig('/home/ramana44/autoencoder-regularisation-/all_results/cycle_experimnets/imagesaves/orig_circle.png')
plt.close()
##################################################################################################################
transform_to_3D = np.random.rand(2, 3)
#print(transform_to_3D)
circle_3D = np.matmul(arr_points, transform_to_3D, )
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(circle_3D[:,0], circle_3D[:,1], circle_3D[:,2])


plt.show()
#plt.savefig('/home/ramana44/autoencoder-regularisation-/all_results/cycle_experimnets/imagesaves/circleIn3DSpace.png')
plt.close()
##################################################################################################################
import torch
from torch.utils.data import DataLoader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        return self.data[:, index]

def get_loader(data):
    dataset = Dataset(data)
    sampler = torch.utils.data.SubsetRandomSampler(list(range(data.shape[0])))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler)
    return loader

dim = 15

num_training_points = 15

#rng = np.random.default_rng(seed=1)
transform_to_nD = 4*np.random.rand(2, dim)-2
#transform_to_nD = 4*rng.random((2, dim))-2
print(transform_to_nD)



tr_points_2d = np.array(PointsInCircum(1.,num_training_points))

tr_imbalancing_points = np.array(ImbalancingPointsInCircum(1.0,num_training_points))

#tr_points_2d = np.concatenate((tr_points_2d, tr_imbalancing_points))

print('len(tr_points_2d)', len(tr_points_2d))

#print('tr_points_2d', tr_points_2d)

plt.scatter(tr_points_2d[:,0], tr_points_2d[:,1], color='blue')
plt.show()
plt.savefig('./results_plotting/synthetic_data_exp_results/cycle_results/training_points2d.png')



data_tr = torch.from_numpy(PointsInCircumNDim(tr_points_2d, transform_to_nD)).float()
data_val = torch.from_numpy(PointsInCircumNDim(PointsInCircum(1.,200), transform_to_nD)).float()

loader_tr = get_loader(data_tr)
loader_val = get_loader(data_val)
##################################################################################################################
model = AE(dim, 6, 2, 2, Sin()).to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print(pytorch_total_params)
##################################################################################################################

#print(model_conv)
##################################################################################################################
#del model_reg
hidden_size = 6
no_layers = 2
lr = 5e-3

no_filters = 5
kernel_size = 3
no_layers_conv = 2


latent_dim = 2


model = AE(dim, hidden_size, 2, no_layers, Sin()).to(device)

model_reg_tr = AE(dim, hidden_size, latent_dim, no_layers, Sin()).to(device)
model_reg_ran = AE(dim, hidden_size, latent_dim, no_layers, Sin()).to(device)
model_reg_cheb = AE(dim, hidden_size, latent_dim, no_layers, Sin()).to(device)
model_reg_leg = AE(dim, hidden_size, latent_dim, no_layers, Sin()).to(device)

#
model_conv = ConvoAE(latent_dim).to(device)

model_contra = Autoencoder_linear(latent_dim, dim).to(device)

model_cnn_vae = ConvVAE_circle(image_channels=1, h_dim=5*4, z_dim=latent_dim).to(device)


model_mlp_vae = VAE_mlp_circle_new(image_size=dim, h_dim=6, z_dim=latent_dim).to(device)

no_epochs = 550
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer_tr = torch.optim.Adam(model_reg_tr.parameters(), lr=lr)
optimizer_ran = torch.optim.Adam(model_reg_ran.parameters(), lr=lr)
optimizer_cheb = torch.optim.Adam(model_reg_cheb.parameters(), lr=lr)
optimizer_leg = torch.optim.Adam(model_reg_leg.parameters(), lr=lr)


optimizer_mlp_vae = torch.optim.Adam(model_mlp_vae.parameters(), lr=1e-3) 

optimizer_conv = torch.optim.Adam(model_conv.parameters(), lr =0.002, weight_decay = 1e-5)
optimizer_contra = torch.optim.Adam(model_contra.parameters(), lr =0.002, weight_decay = 1e-5)
optimizer_cnn_vae = torch.optim.Adam(model_cnn_vae.parameters(), lr=1e-3) 

mod_loss = []
mod_loss_tr = []
mod_loss_ran = []
mod_loss_cheb = []
mod_loss_leg = []
mod_loss_conv = []
mod_loss_vae = []
mod_loss_mlp_vae = []

from regularisers_without_vegas_fmnist import computeC1Loss, sampleChebyshevNodes, sampleLegendreNodes


                    
regNodesSamplings = (["mlp_ae", "trainingData", "random", "chebyshev", "legendre",
                    "conv", "contra", "mlp_vae", "cnn_vae"])



models = ([model, model_reg_tr, model_reg_ran, model_reg_cheb, model_reg_leg,
        model_conv, model_contra, model_mlp_vae, model_cnn_vae])


optimizers = ([optimizer, optimizer_tr, optimizer_ran, optimizer_cheb, optimizer_leg, 
            optimizer_conv, optimizer_contra, optimizer_mlp_vae, optimizer_cnn_vae])




szSample = 10
#latent_dim = 2
weightJac = False
degPoly=21
alpha = 0.1

###########################################################
# Legendre
###########################################################
points = np.polynomial.legendre.leggauss(degPoly)[0][::-1]

weights = np.polynomial.legendre.leggauss(degPoly)[1][::-1]
###########################################################


for ind, model_reg in enumerate(models):
    mod_loss_reg = []
    regNodesSampling = regNodesSamplings[ind]
    print(regNodesSampling)
    optimizer = optimizers[ind]
    #print(mod_loss_reg)
    for epoch in range(no_epochs):

        if (regNodesSampling != "conv") and (regNodesSampling != "vae") and (regNodesSampling != "mlp_vae") and (regNodesSampling != "contra") and (regNodesSampling != "mlp_ae") and (regNodesSampling != "mlp_vae") and (regNodesSampling != "cnn_vae"):
                        
            model_output = model_reg(data_tr.to(device))
            loss = torch.nn.MSELoss()(model_output, data_tr.to(device))
            mod_loss_reg.append(float(loss.item()))

            if(regNodesSampling == 'chebyshev'):
                nodes_subsample_np, weights_subsample_np = sampleChebyshevNodes(szSample, latent_dim, weightJac, n=degPoly)
                nodes_subsample = torch.FloatTensor(nodes_subsample_np).to(device)
                weights_subsample = torch.FloatTensor(weights_subsample_np).to(device)
            elif(regNodesSampling == 'legendre'): 
                nodes_subsample_np, weights_subsample_np = sampleLegendreNodes(szSample, latent_dim, weightJac, points, weights,  n=degPoly)
                nodes_subsample = torch.FloatTensor(nodes_subsample_np).to(device)

                weights_subsample = torch.FloatTensor(weights_subsample_np).to(device)
            elif(regNodesSampling == 'random'):
                nodes_subsample = torch.FloatTensor(szSample, latent_dim).uniform_(-1, 1)
            elif(regNodesSampling == 'trainingData'):
                nodes_subsample = model_reg.encoder(data_tr[0:szSample, :].to(device))

            loss_C1, Jac = computeC1Loss(nodes_subsample, model_reg, device, guidanceTerm = False) #

            loss = (1.-alpha)*loss + alpha*loss_C1
        
        if regNodesSampling == "mlp_ae":
            model_output = model_reg(data_tr.to(device))
            loss = torch.nn.MSELoss()(model_output, data_tr.to(device))
            mod_loss_reg.append(float(loss.item()))

        if regNodesSampling == "conv":
            model_output = model_reg(data_tr.unsqueeze(1).to(device))
            loss = torch.nn.MSELoss()(model_output.squeeze(1), data_tr.to(device))
            mod_loss_reg.append(float(loss.item()))

        if regNodesSampling == "contra":
            lam = 1e-2
            img = data_tr.unsqueeze(1).to(device)
            img = data_tr.to(device)
            img = Variable(img)
            recon = model_reg(img)
            W = list(model_reg.parameters())[6]
            hidden_representation = model_reg.encoder(img)
            loss, testcontraLoss = contractive_loss_function(W, img, recon,
                                hidden_representation, lam)
            mod_loss_reg.append(float(loss.item()))
        
        if regNodesSampling == "mlp_vae":
            #print('data_tr.shape', data_tr.shape)
            #images = data_tr.reshape(-1, 15)
            images = data_tr
            recon_images, mu, logvar = model_reg(images.float().to(device))
            #print('recon_images.shape', recon_images.shape)
            loss, bce, kld = loss_fn_mlp_vae(recon_images.to(device), images.to(device), mu.to(device), logvar.to(device))
            
            mod_loss_reg.append(float(loss.item()))

        if regNodesSampling == "cnn_vae":

            recon_images, mu, logvar = model_reg(data_tr.unsqueeze(1).float().to(device))
            #print('recon_images.shape', recon_images.shape)
            #print('data_tr.shape', data_tr.shape)
            loss, bce, kld = loss_fn_cnn_vae(recon_images.to(device), data_tr.unsqueeze(1).to(device), mu.to(device), logvar.to(device))
            mod_loss_reg.append(float(loss.item()))
            #print("cnn vae loss", loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.plot(list(range(0,no_epochs)), mod_loss_reg, label=regNodesSampling+', '+str(mod_loss_reg[-1]))
    plt.xlabel("$epoch$")
    plt.ylabel("$loss$")
    plt.legend()
    plt.grid(True)
plt.show()
plt.savefig('/home/ramana44/autoencoder-regularisation-/all_results/cycle_experimnets/imagesaves/allLosses.png')
plt.close()
for opt in optimizers:
    del opt
##################################################################################################################

##################################################################################################################
import matplotlib.pyplot as plt
points_tr = (model.encoder(data_tr.to(device))).detach().cpu().numpy()
points_val = (model.encoder(data_val.to(device))).detach().cpu().numpy()
plt.scatter(points_val[:,0], points_val[:,1], color="orange")
plt.scatter(points_tr[:,0], points_tr[:,1], color="blue")

#plt.scatter(arr_points[:,0], arr_points[:,1], color='gray', alpha=0.05)
plt.scatter(arr_points[:,0], arr_points[:,1], color='gray')
plt.grid(False)
#plt.title("Baseline: training data")
plt.show()


labels = ["mlp_ae", "Reg on training data", "Reg on random points", "Reg on chebyshev nodes", "Reg on legendre nodes","conv", "contra", "mlp_vae", "cnn_vae"]

for ind, model_reg in enumerate(models):
    if ind < 5:
        points_tr = (model_reg.encoder(data_tr.to(device))).detach().cpu().numpy()
        points_val = (model_reg.encoder(data_val.to(device))).detach().cpu().numpy()
        plt.scatter(points_val[:,0], points_val[:,1], label='validation samples', color="orange")
        plt.scatter(points_tr[:,0], points_tr[:,1], label='training samples', color="blue")
        #plt.scatter(arr_points[:,0], arr_points[:,1], color='gray', alpha=0.05)
        plt.scatter(arr_points[:,0], arr_points[:,1], color='gray')
        plt.grid(False)
        #plt.title(labels[ind])
        plt.show()
        plt.savefig('./results_plotting/synthetic_data_exp_results/cycle_results/'+labels[ind]+'.png')
        plt.close()

    elif (labels[ind] == "conv" or labels[ind] == "contra" ):
        #print('ind', ind)
        points_tr = (model_reg.encoder(data_tr.unsqueeze(1).to(device))).detach().cpu().numpy()
        points_val = (model_reg.encoder(data_val.unsqueeze(1).to(device))).detach().cpu().numpy()
        points_tr = points_tr.reshape(-1,latent_dim)
        points_val = points_val.reshape(-1,latent_dim)

        plt.scatter(points_val[:,0], points_val[:,1], label='validation samples', color="orange")
        plt.scatter(points_tr[:,0], points_tr[:,1], label='training samples', color="blue")
        #plt.scatter(arr_points[:,0], arr_points[:,1], color='gray', alpha=0.05)
        plt.scatter(arr_points[:,0], arr_points[:,1], color='gray')
        plt.grid(False)
        #plt.title(labels[ind])
        plt.show()
        plt.savefig('./results_plotting/synthetic_data_exp_results/cycle_results/'+labels[ind]+'.png')
        plt.close()
    elif labels[ind] == "mlp_vae":
        #points_tr, _, _ = (model_reg.encode(data_tr.to(device), False))
        points_tr = model_reg.fc1(model_reg.encoder(data_tr.float().to(device)))
        points_tr = points_tr.detach().cpu().numpy()

        #points_val,_ ,_ = (model_reg.encode(data_val.to(device), False))
        points_val = model_reg.fc1(model_reg.encoder(data_val.float().to(device)))
        points_val = points_val.detach().cpu().numpy()

        points_tr = points_tr.reshape(-1,latent_dim)
        points_val = points_val.reshape(-1,latent_dim)

        plt.scatter(points_val[:,0], points_val[:,1], label='validation samples', color="orange")
        plt.scatter(points_tr[:,0], points_tr[:,1], label='training samples', color="blue")
        #plt.scatter(arr_points[:,0], arr_points[:,1], color='gray', alpha=0.05)
        plt.scatter(arr_points[:,0], arr_points[:,1], color='gray')
        plt.grid(False)
        #plt.title(labels[ind])
        plt.show()
        plt.savefig('./results_plotting/synthetic_data_exp_results/cycle_results/'+labels[ind]+'.png')
        plt.close()
    elif labels[ind] == "cnn_vae":
        #points_tr, _, _ = (model_reg.encode(data_tr.to(device), False))
        points_tr = model_reg.fc1(model_reg.encoder(data_tr.unsqueeze(1).float().to(device)))
        points_tr = points_tr.detach().cpu().numpy()

        #points_val,_ ,_ = (model_reg.encode(data_val.to(device), False))
        points_val = model_reg.fc1(model_reg.encoder(data_val.unsqueeze(1).float().to(device)))
        points_val = points_val.detach().cpu().numpy()

        points_tr = points_tr.reshape(-1,latent_dim)
        points_val = points_val.reshape(-1,latent_dim)

        plt.scatter(points_val[:,0], points_val[:,1], label='validation samples', color="orange")
        plt.scatter(points_tr[:,0], points_tr[:,1], label='training samples', color="blue")
        #plt.scatter(arr_points[:,0], arr_points[:,1], color='gray', alpha=0.05)
        plt.scatter(arr_points[:,0], arr_points[:,1], color='gray')
        plt.grid(False)
        #plt.title(labels[ind])
        plt.show()
        plt.savefig('./results_plotting/synthetic_data_exp_results/cycle_results/'+labels[ind]+'.png')
        plt.close()
    del model_reg
##################################################################################################################


##################################################################################################################


##################################################################################################################


##################################################################################################################


##################################################################################################################


##################################################################################################################


##################################################################################################################


##################################################################################################################
