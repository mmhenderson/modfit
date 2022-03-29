#!/bin/bash
#SBATCH --partition=tarrq
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --exclude=mind-1-23
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=28-00:00:00

source ~/myenv/bin/activate

cd /user_data/mmhender/imStat/code/model_fitting

subjects=(1 2 3 4 5 6 7 8)

debug=0
up_to_sess=40

sample_batch_size=100
voxel_batch_size=100
zscore_features=1
ridge=1
use_precomputed_prfs=1
which_prf_grid=5
from_scratch=1
date_str=0
do_val=1
do_tuning=0
do_sem_disc=0

fitting_type=full_midlevel

n_ori_gabor=12
n_sf_gabor=8
gabor_nonlin_fn=1
use_pca_gabor_feats=0

n_ori_pyr=4
n_sf_pyr=4
group_all_hl_feats=1
use_pca_pyr_feats_hl=1

use_pca_st_feats=0

fitting_type2=semantic
# semantic_feature_set=all_coco_categ
semantic_feature_set=all_coco_categ_pca

do_varpart=0
include_solo_models=0
do_pyr_varpart=0


for subject in ${subjects[@]}
do
    python3 fit_model.py --subject $subject --debug $debug --up_to_sess $up_to_sess --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --zscore_features $zscore_features --ridge $ridge --use_precomputed_prfs $use_precomputed_prfs --which_prf_grid $which_prf_grid --from_scratch $from_scratch --date_str $date_str --do_val $do_val --do_tuning $do_tuning --do_sem_disc $do_sem_disc --fitting_type $fitting_type --n_ori_gabor $n_ori_gabor --n_sf_gabor $n_sf_gabor --gabor_nonlin_fn $gabor_nonlin_fn --use_pca_gabor_feats $use_pca_gabor_feats --n_ori_pyr $n_ori_pyr --n_sf_pyr $n_sf_pyr --group_all_hl_feats $group_all_hl_feats --use_pca_pyr_feats_hl $use_pca_pyr_feats_hl --use_pca_st_feats $use_pca_st_feats --fitting_type2 $fitting_type2 --semantic_feature_set $semantic_feature_set --do_varpart $do_varpart --include_solo_models $include_solo_models --do_pyr_varpart $do_pyr_varpart

done