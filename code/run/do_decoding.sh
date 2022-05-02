#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

source ~/myenv/bin/activate
cd /user_data/mmhender/imStat/code/analyze_features

subj=999
debug=1
which_prf_grid=5

ft=(gabor_solo)
# ft=(gabor_solo pyramid_texture_ll pyramid_texture_hl_pca sketch_tokens alexnet clip)
# ft=(pyramid_texture_hl_pca)
# ft=(pyramid_texture_hl_pca pyramid_texture_hl)

for feature_type in ${ft[@]}
do

    echo "starting python script"
    python3 decode_categ_from_features.py --subject $subj --debug $debug --feature_type $feature_type --which_prf_grid $which_prf_grid
    
done
