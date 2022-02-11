#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

subj=1
debug=0
which_prf_grid=5

source ~/myenv/bin/activate
cd /user_data/mmhender/imStat/code/analyze_features

ft=(gabor_solo)
# ft=(gabor_solo pyramid_texture_ll pyramid_texture_hl_pca sketch_tokens alexnet clip)
# ft=(pyramid_texture_hl_pca)
# ft=(pyramid_texture_hl_pca pyramid_texture_hl)

for feature_type in ${ft[@]}
do
    
    balance_downsample=1
    zscore_each=1

    python3 linear_discr.py --subject $subj --debug $debug --feature_type $feature_type --which_prf_grid $which_prf_grid --zscore_each $zscore_each --balance_downsample $balance_downsample
    
    balance_downsample=0
    zscore_each=1

    python3 linear_discr.py --subject $subj --debug $debug --feature_type $feature_type --which_prf_grid $which_prf_grid --zscore_each $zscore_each --balance_downsample $balance_downsample
    
done
