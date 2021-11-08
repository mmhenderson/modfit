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

subj=1
volume_space=1
ridge=1
sample_batch_size=100
voxel_batch_size=100
zscore_features=1
zscore_in_groups=0
use_precomputed_prfs=0

fitting_type=sketch_tokens
use_pca_st_feats=0
use_lda_st_feats=0
do_stack=0
lda_discrim_type=None

do_roi_recons=0
do_voxel_recons=0

which_prf_grid=4
do_fitting=0
date_str=Nov-04-2021_2314_17
do_val=1
do_tuning=1
do_sem_disc=1

cd $ROOT/code/model_fitting

python3 fit_model.py --subject $subj --volume_space $volume_space --up_to_sess $up_to_sess --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --zscore_features $zscore_features --zscore_in_groups $zscore_in_groups --ridge $ridge --use_pca_st_feats $use_pca_st_feats --use_lda_st_feats $use_lda_st_feats --debug $debug --fitting_type $fitting_type --do_stack $do_stack --do_roi_recons $do_roi_recons --do_voxel_recons $do_voxel_recons --lda_discrim_type $lda_discrim_type --use_precomputed_prfs $use_precomputed_prfs --which_prf_grid $which_prf_grid --do_fitting $do_fitting --do_val $do_val --date_str $date_str --do_tuning $do_tuning --do_sem_disc $do_sem_disc
