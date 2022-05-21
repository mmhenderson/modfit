#!/bin/bash
#SBATCH --partition=tarrq
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

debug=0
use_node_storage=0
n_ori=12
n_sf=8
# n_ori=4;
# n_sf=4;

sample_batch_size=100
which_prf_grid=3
gabor_solo=1
nonlin_fn=1

source ~/myenv/bin/activate

cd /user_data/mmhender/imStat/code/feature_extraction

# sublist=(999)

sublist=(1 2 3 4 5 6 7 8 999)
for subject in ${sublist[@]}
do
    python3 extract_gabor_features.py --subject $subject --n_ori=$n_ori --n_sf=$n_sf --sample_batch_size $sample_batch_size --use_node_storage $use_node_storage --which_prf_grid $which_prf_grid --nonlin_fn $nonlin_fn --debug $debug
    
done

