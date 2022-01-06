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

debug=0
volume_space=1
up_to_sess=40
sample_batch_size=100
voxel_batch_size=100
zscore_features=1
zscore_in_groups=0
ridge=1
use_precomputed_prfs=0
which_prf_grid=5

do_fitting=1
date_str=0
do_val=1
do_tuning=1
do_sem_disc=1
do_stack=0
do_roi_recons=0
do_voxel_recons=0
save_pred_data=0

fitting_type=alexnet
alexnet_layer_name=all_conv
alexnet_padding_mode=reflect
use_pca_alexnet_feats=1

cd $ROOT/code/model_fitting

subjects=(2 3 4 5 6 7 8)
for subject in ${subjects[@]}
do

    python3 fit_model.py --subject $subject --debug $debug --volume_space $volume_space --up_to_sess $up_to_sess --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --zscore_features $zscore_features --zscore_in_groups $zscore_in_groups --ridge $ridge --use_precomputed_prfs $use_precomputed_prfs --which_prf_grid $which_prf_grid --do_fitting $do_fitting --date_str $date_str --do_val $do_val --do_tuning $do_tuning --do_sem_disc $do_sem_disc --do_stack $do_stack --do_roi_recons $do_roi_recons --do_voxel_recons $do_voxel_recons --save_pred_data $save_pred_data --fitting_type $fitting_type --alexnet_layer_name $alexnet_layer_name --alexnet_padding_mode $alexnet_padding_mode --use_pca_alexnet_feats $use_pca_alexnet_feats   

done