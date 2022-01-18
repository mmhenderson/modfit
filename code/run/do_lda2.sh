#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

subj=2
debug=0
which_prf_grid=5

source ~/myenv/bin/activate
cd ../feature_extraction

# ft=(pyramid_texture_hl_pca)
# dt=(animacy)
ft=(gabor_solo sketch_tokens pyramid_texture_ll pyramid_texture_hl)
dt=(all_supcat animacy indoor_outdoor food vehicle person animal)


for feature_type in ${ft[@]}
do
    for discrim_type in ${dt[@]}
    do
        balance_downsample=1
        zscore_each=1
        zscore_groups=0
        
        python3 linear_discr.py --subject $subj --debug $debug --feature_type $feature_type --discrim_type $discrim_type --which_prf_grid $which_prf_grid --zscore_each $zscore_each --zscore_groups $zscore_groups --balance_downsample $balance_downsample
        
#         balance_downsample=1
#         zscore_each=0
#         zscore_groups=0
        
#         python3 linear_discr.py --subject $subj --debug $debug --feature_type $feature_type --discrim_type $discrim_type --which_prf_grid $which_prf_grid --zscore_each $zscore_each --zscore_groups $zscore_groups --balance_downsample $balance_downsample
        
#         balance_downsample=1
#         zscore_each=0
#         zscore_groups=1
        
#         python3 linear_discr.py --subject $subj --debug $debug --feature_type $feature_type --discrim_type $discrim_type --which_prf_grid $which_prf_grid --zscore_each $zscore_each --zscore_groups $zscore_groups --balance_downsample $balance_downsample
        
        
    done
done
