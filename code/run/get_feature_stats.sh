#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

debug=0
which_prf_grid=5

subjects=(1)
# subjects=(999)
# subjects=(all)
feature_types=(sketch_tokens_residuals)
# feature_types=(gabor_solo)
# sketch_tokens pyramid_texture_ll pyramid_texture_hl)
# feature_types=(sketch_tokens pyramid_texture_ll pyramid_texture_hl)

# subjects=(2 3 4 5 6 7 8)
# feature_types=(pyramid_texture_hl_pca alexnet clip)

source ~/myenv/bin/activate
cd /user_data/mmhender/imStat/code/analyze_features

for subj in ${subjects[@]}
do
    for feature_type in ${feature_types[@]}
    do
        python3 get_feature_stats.py --subject $subj --debug $debug --feature_type $feature_type --which_prf_grid $which_prf_grid
    done
    
done
