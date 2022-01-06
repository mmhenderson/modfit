#!/bin/bash
#SBATCH --partition=tarrq
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=30-00:00:00

debug=0
use_node_storage=0
n_ori=4
n_sf=4
batch_size=500
which_prf_grid=5

source ~/myenv/bin/activate

CWD=$(pwd)
cd ../
ROOT=$(pwd)
cd $ROOT/feature_extraction

sublist=(4 5 6 7 8)
for subj in ${sublist[@]}
do

    python3 extract_pyramid_texture_features.py --subject $subj --use_node_storage $use_node_storage --n_ori=$n_ori --n_sf=$n_sf --batch_size $batch_size --debug $debug --which_prf_grid $which_prf_grid 

done