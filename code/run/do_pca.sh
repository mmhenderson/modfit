#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

subj=1
debug=1
# type=alexnet
type=clip
min_pct_var=95
zscore=0
max_pc_to_retain=100
which_prf_grid=5

source ~/myenv/bin/activate
cd ../
cd feature_extraction

python3 pca_feats.py --subject $subj --debug $debug --type $type --zscore $zscore --max_pc_to_retain $max_pc_to_retain --min_pct_var $min_pct_var --which_prf_grid $which_prf_grid
