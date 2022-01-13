#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

debug=0
type=texture_pyramid
zscore=0
min_pct_var=95
max_pc_to_retain=150
which_prf_grid=5

source ~/myenv/bin/activate
cd ../
cd feature_extraction

# subjects=(2 3 4 5 6 7 8)
subjects=(8)
for subject in ${subjects[@]}
do

    python3 pca_feats.py --subject $subject --debug $debug --type $type --zscore $zscore --max_pc_to_retain $max_pc_to_retain --min_pct_var $min_pct_var --which_prf_grid $which_prf_grid

done