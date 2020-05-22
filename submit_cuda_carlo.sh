#!/bin/bash
#SBATCH -J CudaMonteCarlo
#SBATCH -A cs475-575
#SBATCH -p class
#SBATCH --gres=gpu:1
#SBATCH -o cuda_carlo.out
#SBATCH -e cuda_carlo.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shillinm@oregonstate.edu

for t in 1 2 4 8 16 32
do
make
./cuda_carlo
done