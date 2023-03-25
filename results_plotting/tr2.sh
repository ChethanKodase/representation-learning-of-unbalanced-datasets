#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH -c 16  # number of processor cores (i.e. threads)
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00   # walltime
#SBATCH -J "tr2"   # job name

#module purge                                 # purge if you already have modules loaded
/home/ramana44/.conda/envs/myenv/bin/python3.9 /home/ramana44/representation-learning-of-unbalanced-datasets/results_plotting/class_specific_average_psnr_ssim_against_class_imbalances_training2.py

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
OUTFILE=""

exit 0