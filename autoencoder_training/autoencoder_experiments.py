from datasets import getDataset
import torch

import torchvision
from torchvision import transforms, datasets
from torchvision.datasets import FashionMNIST
import matplotlib
import matplotlib.pyplot as plt


train_loader, test_loader, noChannels, dx, dy = getDataset(dataset = "FashionMNIST", batch_size = 60000)  # FashionMNIST , MNIST


training_data, training_labels = next(iter(train_loader))

class_indices_ = []


for i in range(10): class_indices_.append(torch.where(training_labels==i)[0])
    
unbalancing_fractions = [0.5, 0.8, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

imabalanced_class_indices = []


for i in range(len(unbalancing_fractions)): 
    imabalanced_class_indices.append(class_indices_[i][:int( unbalancing_fractions[i]*len(class_indices_[i]))])


print('First this : imabalanced_class_indices', imabalanced_class_indices)

# printing the populations

for i in range(len(imabalanced_class_indices)):
    print('class '+str(i)+' population : ', imabalanced_class_indices[i].shape)  

seggregate = training_data[imabalanced_class_indices[9]]

imbalanced_class_indices_merged = torch.tensor([])
for i in range(len(imabalanced_class_indices)):
    imbalanced_class_indices_merged = torch.cat((imbalanced_class_indices_merged, imabalanced_class_indices[i].int() ))

imbalanced_class_indices_merged = imbalanced_class_indices_merged

print('imbalanced_class_indices_merged', imbalanced_class_indices_merged)

print('imbalanced_class_indices_merged.shape', imbalanced_class_indices_merged.shape)

print('imbalanced_class_indices_merged.max()', imbalanced_class_indices_merged.max())

print('imbalanced_class_indices_merged.min()', imbalanced_class_indices_merged.min())


print(" all indices collected : ", imbalanced_class_indices_merged.shape)

rand_inds = torch.randperm(len(imbalanced_class_indices_merged))

print('rand_inds', rand_inds)

print('rand_inds.max()', rand_inds.max())

print('rand_inds.min()', rand_inds.min())


imbal_class_inds_mrgd_shffld = imbalanced_class_indices_merged[rand_inds]

print('imb_class_inss_mrgd_shffld.shape', imbal_class_inds_mrgd_shffld.shape)

print('imbal_class_inds_mrgd_shffld.max()', imbal_class_inds_mrgd_shffld.max())

print('imbal_class_inds_mrgd_shffld.min()', imbal_class_inds_mrgd_shffld.min())

print("till here")




imbal_class_inds_mrgd_shffld = imbal_class_inds_mrgd_shffld.type(torch.int64)
print("now check : ", imbal_class_inds_mrgd_shffld)

#imbal_class_inds_mrgd_shffld = imbal_class_inds_mrgd_shffld.long()
imblcnd_shffld_trng_lbls = training_labels[imbal_class_inds_mrgd_shffld]

print('torch.where(imbal_class_inds_mrgd_shffld==0)[0].shape', torch.where(imblcnd_shffld_trng_lbls==0)[0].shape)
print('torch.where(imbal_class_inds_mrgd_shffld==1)[0].shape', torch.where(imblcnd_shffld_trng_lbls==1)[0].shape)
print('torch.where(imbal_class_inds_mrgd_shffld==2)[0].shape', torch.where(imblcnd_shffld_trng_lbls==2)[0].shape)
print('torch.where(imbal_class_inds_mrgd_shffld==3)[0].shape', torch.where(imblcnd_shffld_trng_lbls==3)[0].shape)
print('torch.where(imbal_class_inds_mrgd_shffld==4)[0].shape', torch.where(imblcnd_shffld_trng_lbls==4)[0].shape)
print('torch.where(imbal_class_inds_mrgd_shffld==5)[0].shape', torch.where(imblcnd_shffld_trng_lbls==5)[0].shape)
print('torch.where(imbal_class_inds_mrgd_shffld==6)[0].shape', torch.where(imblcnd_shffld_trng_lbls==6)[0].shape)
print('torch.where(imbal_class_inds_mrgd_shffld==7)[0].shape', torch.where(imblcnd_shffld_trng_lbls==7)[0].shape)
print('torch.where(imbal_class_inds_mrgd_shffld==8)[0].shape', torch.where(imblcnd_shffld_trng_lbls==8)[0].shape)
print('torch.where(imbal_class_inds_mrgd_shffld==9)[0].shape', torch.where(imblcnd_shffld_trng_lbls==9)[0].shape)


imbalanced_dataset = training_data[imbal_class_inds_mrgd_shffld]

for i in range(len(imbalanced_dataset) - 13700):
    print("did this ? execute")
    plt.imshow(seggregate[i][0])
    plt.savefig('/home/ramana44/representation-learning-of-unbalanced-datasets/experiments/im_no_'+str(i)+'.png')

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
'''print('fixed_x.shape', fixed_x.shape)

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
    
    print(batch_x)

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

'''