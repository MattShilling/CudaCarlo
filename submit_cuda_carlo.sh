#!/bin/bash
#SBATCH -J CudaMonteCarlo
#SBATCH -A cs475-575
#SBATCH -p class
#SBATCH --gres=gpu:1
#SBATCH -o cuda_carlo.out
#SBATCH -e cuda_carlo.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shillinm@oregonstate.edu

K16=16384
K32=32768
K64=65536
K128=131072
K256=262144
K512=524288
K1M=1048576

for BLOCKSIZE in 16 32 64 128
do
    for NUMTRIAL in ${K16} ${K32} ${K64} ${K128} ${K256} ${K512} ${K1M}
    do 
        make
        ./cuda_carlo ${BLOCKSIZE} ${NUMTRIAL}
    done
done