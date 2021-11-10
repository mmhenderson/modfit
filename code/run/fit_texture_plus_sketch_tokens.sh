#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --exclude=mind-1-13
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

source ~/myenv/bin/activate

CWD=$(pwd)
cd ../../
ROOT=$(pwd)

debug=0
up_to_sess=20
which_prf_grid=4

subj=1
volume_space=1
ridge=1
sample_batch_size=100
voxel_batch_size=100
zscore_features=1
zscore_in_groups=0

fitting_type=pyramid_texture
use_pca_pyr_feats_ll=0
use_pca_pyr_feats_hl=1
fitting_type2=sketch_tokens

min_pct_var=95
max_pc_to_retain_pyr_ll=100
max_pc_to_retain_pyr_hl=100
group_all_hl_feats=1

do_stack=0
do_roi_recons=0
do_voxel_recons=0

do_fitting=1
date_str=0
do_val=1
do_tuning=1
do_sem_disc=1

cd $ROOT/code/model_fitting

python3 fit_model.py --subject $subj --debug $debug --fitting_type $fitting_type --fitting_type2 $fitting_type2 --volume_space $volume_space --up_to_sess $up_to_sess --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --zscore_features $zscore_features --zscore_in_groups $zscore_in_groups --ridge $ridge --use_pca_pyr_feats_ll $use_pca_pyr_feats_ll --use_pca_pyr_feats_hl $use_pca_pyr_feats_hl --min_pct_var $min_pct_var --max_pc_to_retain_pyr_ll $max_pc_to_retain_pyr_ll --max_pc_to_retain_pyr_hl $max_pc_to_retain_pyr_hl --group_all_hl_feats $group_all_hl_feats --do_stack $do_stack --do_roi_recons $do_roi_recons --do_voxel_recons $do_voxel_recons --which_prf_grid $which_prf_grid --do_fitting $do_fitting --date_str $date_str --do_val $do_val --do_tuning $do_tuning --do_sem_disc $do_sem_disc
