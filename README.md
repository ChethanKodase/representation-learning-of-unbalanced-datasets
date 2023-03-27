# representation-learning-of-unbalanced-datasets

### Experiment 1 
Autoencoder reconstruction qualities increased with reduction in class imbalance. All autoencoders performed their best(reconstruction qualities in terms of PSNR and SSIM) when all the classes were of equal size. 
In case of reconstruction qualities of unperturbed test samples,  though the Comparative performances of autoencoders remained the same with variations in class imbalance, the individual autoencoders gave better reconstruction qualities. 
Results directory address : `./results_plotting/box_plots/class_imbalance_without_perturbation`

### Experiment 2
Average reconstruction qualities of all AEs against level of data imbalance
Results : `/home/ramana44/representation-learning-of-unbalanced-datasets/results_plotting/Avg_recon_qlty`


Notes: 

When majority calss fraction of 0.998 is used, the majority class population is 5988 and sample size of rest of the classes is 1.

for majority calss fraction of 0.9995, the autoencoders are trained only with majority class