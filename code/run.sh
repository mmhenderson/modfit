#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=48:00:00

source ~/cudaenv/bin/activate

subj=1
roi=None

n_ori=2
n_sf=2
up_to_sess=1

# n_ori=36
# n_sf=12
# up_to_sess=20

sample_batch_size=20
voxel_batch_size=100
debug=True

python3 fit_model.py $subj $roi $up_to_sess $n_ori $n_sf $sample_batch_size $voxel_batch_size $debug
