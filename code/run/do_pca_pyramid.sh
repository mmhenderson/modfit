#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --gres=gpu:0
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

debug=0
min_pct_var=95
max_pc_to_retain=100
which_prf_grid=5

source ~/myenv/bin/activate

cd /user_data/mmhender/modfit/code/feature_extraction

subjects=(1)

for subject in ${subjects[@]}
do
    
    pca_type=pcaHL

    python3 pca_texture_feats.py --subject $subject --debug $debug --pca_type $pca_type --max_pc_to_retain $max_pc_to_retain --min_pct_var $min_pct_var --which_prf_grid $which_prf_grid

    pca_type=pcaHL_simple

    python3 pca_texture_feats.py --subject $subject --debug $debug --pca_type $pca_type --max_pc_to_retain $max_pc_to_retain --min_pct_var $min_pct_var --which_prf_grid $which_prf_grid
    
    pca_type=pcaHL_sepscales

    python3 pca_texture_feats.py --subject $subject --debug $debug --pca_type $pca_type --max_pc_to_retain $max_pc_to_retain --min_pct_var $min_pct_var --which_prf_grid $which_prf_grid

    pca_type=pcaAll

    python3 pca_texture_feats.py --subject $subject --debug $debug --pca_type $pca_type --max_pc_to_retain $max_pc_to_retain --min_pct_var $min_pct_var --which_prf_grid $which_prf_grid

    
done