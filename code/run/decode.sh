#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=mind-1-1
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

debug=0
which_prf_grid=5

subjects=(999)

feature_types=(gabor_solo)

source ~/myenv/bin/activate
cd /user_data/mmhender/imStat/code/analyze_features

for subj in ${subjects[@]}
do
    for feature_type in ${feature_types[@]}
    do
        python3 decode_categ_from_features.py --subject $subj --debug $debug --feature_type $feature_type --which_prf_grid $which_prf_grid
    done
    
done
