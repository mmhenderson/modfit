#!/bin/bash
#SBATCH --partition=gpu
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

subj=2
debug=0
volume_space=1
up_to_sess=40
sample_batch_size=100
voxel_batch_size=100
zscore_features=1
ridge=1
which_prf_grid=5
use_precomputed_prfs=1

do_stack=0
do_roi_recons=0
do_voxel_recons=0
do_tuning=1
do_sem_disc=1

fitting_type=pyramid_texture

n_ori_pyr=4
n_sf_pyr=4
group_all_hl_feats=1
use_pca_pyr_feats_hl=1

cd $ROOT/code/model_fitting

python3 fit_model.py --subject $subj --debug $debug --volume_space $volume_space --up_to_sess $up_to_sess --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --zscore_features $zscore_features --ridge $ridge --which_prf_grid $which_prf_grid --use_precomputed_prfs $use_precomputed_prfs --do_stack $do_stack --do_roi_recons $do_roi_recons --do_voxel_recons $do_voxel_recons --do_tuning $do_tuning --do_sem_disc $do_sem_disc --fitting_type $fitting_type --n_ori_pyr $n_ori_pyr --n_sf_pyr $n_sf_pyr --group_all_hl_feats $group_all_hl_feats --use_pca_pyr_feats_hl $use_pca_pyr_feats_hl 

# use_pca_pyr_feats_ll=0
# use_pca_pyr_feats_hl=0

# python3 fit_model.py --subject $subj --debug $debug --volume_space $volume_space --up_to_sess $up_to_sess --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --zscore_features $zscore_features --ridge $ridge --which_prf_grid $which_prf_grid --use_precomputed_prfs $use_precomputed_prfs --do_stack $do_stack --do_roi_recons $do_roi_recons --do_voxel_recons $do_voxel_recons --do_tuning $do_tuning --do_sem_disc $do_sem_disc --fitting_type $fitting_type --n_ori_pyr $n_ori_pyr --n_sf_pyr $n_sf_pyr --group_all_hl_feats $group_all_hl_feats --use_pca_pyr_feats_ll $use_pca_pyr_feats_ll --use_pca_pyr_feats_hl $use_pca_pyr_feats_hl 
