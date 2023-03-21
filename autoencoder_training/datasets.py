import numpy as np

import torch
import torchvision
from torchvision import transforms, datasets
from dataset_imbalancing import create_data_imbalance

import numpy as np
import matplotlib.pyplot as plt

import os
from operator import itemgetter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from operator import itemgetter


import numpy as np

import matplotlib.pyplot

def getFashionMNIST(batch_size = 1, drop_last=False):
    fashionmnist_data = torchvision.datasets.FashionMNIST(download=True, root = 'data/fashionmnist', transform = 
                                                                                 transforms.Compose([transforms.Resize(32),
                                                                                 transforms.ToTensor(), 
                                                                                 transforms.Lambda(lambda x: x.repeat(1, 1, 1))
                                                                                 ]))

    fashionmnist_data_test = torchvision.datasets.FashionMNIST(download=True, root = 'data/fashionmnist', train=False, transform = 
                                                                                 transforms.Compose([transforms.Resize(32),
                                                                                 transforms.ToTensor(), 
                                                                                 transforms.Lambda(lambda x: x.repeat(1, 1, 1))
                                                                                 ]))

    train_loader = torch.utils.data.DataLoader(fashionmnist_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=16,
                                              drop_last=drop_last)

    test_loader = torch.utils.data.DataLoader(fashionmnist_data_test,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=16,
                                              drop_last=drop_last)

    return train_loader, test_loader



def getDataset(dataset = "MNIST", batch_size = 1):
    if(dataset == "MNIST"):
        train_loader, test_loader = getMNIST(batch_size)
        noChannels,dx, dy = train_loader.dataset.__getitem__(1)[0].shape
    elif(dataset == "FashionMNIST"):
        train_loader, test_loader = getFashionMNIST(batch_size)
        noChannels, dx, dy = train_loader.dataset.__getitem__(1)[0].shape
    elif(dataset == "Cifar10"):
        train_loader, test_loader = getCifar10(batch_size)
        noChannels, dx, dy = train_loader.dataset.__getitem__(1)[0].shape
    else:
        return None, None, None, None, None    
        
    return train_loader, test_loader, noChannels, dx, dy


def get_train_test_datasets_and_data_in_batches(unbalancing_train_fractions, unbalancing_test_fractions, set_batch_size, dataset):

    train_loader, test_loader, no_channels, dx, dy = getDataset(dataset, batch_size = 60000)  # FashionMNIST , MNIST
    train_data, train_labels = next(iter(train_loader))
    test_data, test_labels = next(iter(test_loader))

    imbalanced_train_dataset, imblcnd_shffld_train_lbls = create_data_imbalance(train_data, train_labels, unbalancing_train_fractions)
    imbalanced_test_dataset, imblcnd_shffld_test_lbls = create_data_imbalance(test_data, test_labels, unbalancing_test_fractions)

    fix_size_train = set_batch_size* (imbalanced_train_dataset.shape[0]//set_batch_size)
    fix_size_test = set_batch_size* (imbalanced_test_dataset.shape[0]//set_batch_size)
    
    imbalanced_train_dataset = imbalanced_train_dataset[:fix_size_train]
    imbalanced_test_dataset = imbalanced_test_dataset[:fix_size_test]

    batched_imbalanced_train_dataset = imbalanced_train_dataset.reshape(imbalanced_train_dataset.shape[0]//set_batch_size, set_batch_size, no_channels, dx, dy )
    batched_imbalanced_test_dataset = imbalanced_test_dataset.reshape(imbalanced_test_dataset.shape[0]//set_batch_size, set_batch_size, no_channels, dx, dy )

    return batched_imbalanced_train_dataset, batched_imbalanced_test_dataset, no_channels, dx, dy



def get_shuffeled_labels_after_imbalancing(unbalancing_train_fractions, unbalancing_test_fractions, dataset):

    train_loader, test_loader, _, _, _ = getDataset(dataset, batch_size = 60000)  # FashionMNIST , MNIST
    train_data, train_labels = next(iter(train_loader))
    test_data, test_labels = next(iter(test_loader))

    _, imblcnd_shffld_train_lbls = create_data_imbalance(train_data, train_labels, unbalancing_train_fractions)
    _, imblcnd_shffld_test_lbls = create_data_imbalance(test_data, test_labels, unbalancing_test_fractions)

    return imblcnd_shffld_train_lbls, imblcnd_shffld_test_lbls


def get_dataset_class_stats(train_class_fracs, test_class_fracs, class_labels ,dataset):
    train_lbls, test_lbls = get_shuffeled_labels_after_imbalancing(train_class_fracs, test_class_fracs, dataset)
    print("Checking class populations after shuffling in train data")
    for i in range(len(class_labels)):
        print('Size of class  : '+str(class_labels[i]), torch.where(train_lbls==i)[0].shape)    
    print()
    print("total train data size : ", len(train_lbls))

    print("Checking class populations after shuffling in test data")
    for i in range(len(class_labels)):
        print('Size of class: '+str(class_labels[i]), torch.where(test_lbls==i)[0].shape)    
    print()
    print("total test data size : ", len(test_lbls))
    print()