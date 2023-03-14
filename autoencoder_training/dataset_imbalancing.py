import torch

#from datasets import getDataset

def create_data_imbalance(training_data, training_labels, unbalancing_fractions):
    class_indices_ = []
    imabalanced_class_indices = []
    torch.manual_seed(0)
    for i in range(10): class_indices_.append(torch.where(training_labels==i)[0])
    for i in range(len(unbalancing_fractions)): 
        imabalanced_class_indices.append(class_indices_[i][:int( unbalancing_fractions[i]*len(class_indices_[i]))])

    imbalanced_class_indices_merged = torch.tensor([])
    for i in range(len(imabalanced_class_indices)):
        imbalanced_class_indices_merged = torch.cat((imbalanced_class_indices_merged, imabalanced_class_indices[i].int() ))

    rand_inds = torch.randperm(len(imbalanced_class_indices_merged))
    imbal_class_inds_mrgd_shffld = imbalanced_class_indices_merged[rand_inds].type(torch.int64)
    imblcnd_shffld_trng_lbls = training_labels[imbal_class_inds_mrgd_shffld]
    imbalanced_dataset = training_data[imbal_class_inds_mrgd_shffld]

    return imbalanced_dataset, imblcnd_shffld_trng_lbls


