#!/bin/bash

#SBATCH --job-name=DistillBERT_all1epoch_head_20epochs_batch8_lr20e-5_(10)         # Job name
#SBATCH --output=log/%x_%j.out          # Output file
#SBATCH --time=24:00:00                 # Time limit (HH:MM:SS)
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --gres=gpu:1                    # Number of GPUs
#SBATCH --mem=14G                       # Memory per node
#SBATCH --partition=gpu                 # Partition
#SBATCH --cpus-per-task=1               # Number of CPU cores&


module load anaconda3/2022.10/gcc-11.2.0
module load cuda/10.2.89/intel-19.0.3.199
module load parmetis/4.0.3/intel-19.0.3.199-intel-mpi-int32-real64

source deactivate NLP
source activate NLP

nvidia-smi
python tester_all1epoch_head.py -g 0
