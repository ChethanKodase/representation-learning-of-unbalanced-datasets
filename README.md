# representation-learning-of-unbalanced-datasets

### Experiment 1 
Autoencoder reconstruction qualities increased with reduction in class imbalance. All autoencoders performed their best(reconstruction qualities in terms of PSNR and SSIM) when all the classes were of equal size. 
In case of reconstruction qualities of unperturbed test samples,  though the Comparative performances of autoencoders remained the same with variations in class imbalance, the individual autoencoders gave better reconstruction qualities. 
Results directory address : `./results_plotting/box_plots/class_imbalance_without_perturbation`

### Experiment 2
Average reconstruction qualities of all AEs against level of data imbalance
Results : `/home/ramana44/representation-learning-of-unbalanced-datasets/results_plotting/Avg_recon_qlty`


### MLP-VAE experiments

Trial 1:

Analysis of class flips under perturbation

equal training fractions: 0.1

Test data perturbed with 50 percent Gaussian noise

class 0 -->"T-shirt/top" : No clear label flipping, output with noise 

class 1 -->"Trouser" : Reconstructions seem to flip to a T - shirt

class 2 -->"Pullover" : Reconstructions seem to flip to a noised T - shirt and Shirt

class 3 -->"Dress" : Reconstructions seem to flip to a noised T - shirt

class 4 -->"Coat" : Reconstructions seem to flip to a noised shirt

class 5 -->"Sandal" : Reconstructions seem to flip to a noised bag

class 6 -->"Shirt" : Reconstructions seem to flip to a noised t shirt or noised shirt

class 7 -->"Sneaker" : Reconstructions seem to flip to a noised bag

class 8 -->"Bag" : Reconstructions show no flipping 

class 9 -->"Ankle boot" : Reconstructions show just noise, sometimes boot

Notes: 

When majority calss fraction of 0.998 is used, the majority class population is 5988 and sample size of rest of the classes is 1.

for majority calss fraction of 0.9995, the autoencoders are trained only with majority class
 