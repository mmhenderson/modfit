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
up_to_sess=10

subj=1
volume_space=1
ridge=1
sample_batch_size=100
voxel_batch_size=100
zscore_features=1
zscore_in_groups=1

fitting_type=pyramid_texture
use_pca_pyr_feats_ll=0
use_pca_pyr_feats_hl=0

min_pct_var=95
max_pc_to_retain_pyr_ll=100
max_pc_to_retain_pyr_hl=100

do_stack=0
do_roi_recons=0
do_voxel_recons=0

cd $ROOT/code/model_fitting

python3 fit_model.py --subject $subj --debug $debug --fitting_type $fitting_type --volume_space $volume_space --up_to_sess $up_to_sess --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --zscore_features $zscore_features --zscore_in_groups $zscore_in_groups --ridge $ridge --use_pca_pyr_feats_ll $use_pca_pyr_feats_ll --use_pca_pyr_feats_hl $use_pca_pyr_feats_hl --min_pct_var $min_pct_var --max_pc_to_retain_pyr_ll $max_pc_to_retain_pyr_ll --max_pc_to_retain_pyr_hl $max_pc_to_retain_pyr_hl --do_stack $do_stack --do_roi_recons $do_roi_recons --do_voxel_recons $do_voxel_recons 
