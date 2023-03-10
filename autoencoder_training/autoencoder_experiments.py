from datasets import getDataset
import torch

import torchvision
from torchvision import transforms, datasets
from torchvision.datasets import FashionMNIST
'''train = FashionMNIST(root='.', download=True, train=True, transform = transforms.Compose([transforms.Resize(32),
                                                                                 transforms.ToTensor(), 
                                                                                 transforms.Lambda(lambda x: x.repeat(1, 1, 1))
                                                                                 ]))'''


'''fashionmnist_data = torchvision.datasets.FashionMNIST(download=True, root = 'data/fashionmnist', transform = 
                                                                                transforms.Compose([transforms.Resize(32),
                                                                                transforms.ToTensor(), 
                                                                                transforms.Lambda(lambda x: x.repeat(1, 1, 1))
                                                                                ]))'''

train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=
                                                transforms.Compose([transforms.Resize(64),transforms.ToTensor()]))


'''fashionmnist_data_test = torchvision.datasets.FashionMNIST(download=True, root = 'data/fashionmnist', train=False, transform = 
                                                                                transforms.Compose([transforms.Resize(32),
                                                                                transforms.ToTensor(), 
                                                                                transforms.Lambda(lambda x: x.repeat(1, 1, 1))
                                                                                ]))'''


training_data, training_labels = train_set.data, train_set.targets

print('X.shape', training_data.shape)

print('y.shape', training_labels.shape) 

print('training_labels[:10]', training_labels[:10])

check = torch.where(training_labels==0)[0]

print('check', check)

print('check.shape', check.shape)

seggregate = training_data[check]

print('seggregate.shape', seggregate.shape)

print("till here")


train_loader, test_loader, noChannels, dx, dy = getDataset("FashionMNIST")  # FashionMNIST , MNIST

label0 = torch.tensor([])
label1 = torch.tensor([])
label2 = torch.tensor([])
label3 = torch.tensor([])
label4 = torch.tensor([])
label5 = torch.tensor([])
label6 = torch.tensor([])
label7 = torch.tensor([])
label8 = torch.tensor([])
label9 = torch.tensor([])

print(test_loader.shape)

for inum, (batch_x, label) in enumerate(test_loader):
    
    if(label==0):
        label0 = torch.cat((label0, batch_x))
    if(label==1):
        label1 = torch.cat((label1, batch_x))
    if(label==2):
        label2 = torch.cat((label2, batch_x))
    if(label==3):
        label3 = torch.cat((label3, batch_x))
    if(label==4):
        label4 = torch.cat((label4, batch_x))
    if(label==5):
        label5 = torch.cat((label5, batch_x))
    if(label==6):
        label6 = torch.cat((label6, batch_x))
    if(label==7):
        label7 = torch.cat((label7, batch_x))
    if(label==8):
        label8 = torch.cat((label8, batch_x))
    if(label==9):
        label9 = torch.cat((label9, batch_x))
    if(inum==100):
        break

