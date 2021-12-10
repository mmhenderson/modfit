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
up_to_sess=40

subj=1
volume_space=1
ridge=1
sample_batch_size=100
voxel_batch_size=100
zscore_features=1

which_prf_grid=5
use_precomputed_prfs=1

do_stack=0
do_roi_recons=0
do_voxel_recons=0
do_tuning=1
do_sem_disc=1

fitting_type=alexnet
alexnet_layer_name=best_layer
alexnet_padding_mode=reflect
use_pca_alexnet_feats=1

fitting_type2=semantic
semantic_discrim_type=animacy

cd $ROOT/code/model_fitting

python3 fit_model.py --subject $subj --debug $debug --volume_space $volume_space --up_to_sess $up_to_sess --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --zscore_features $zscore_features --ridge $ridge --which_prf_grid $which_prf_grid --use_precomputed_prfs $use_precomputed_prfs --do_stack $do_stack --do_roi_recons $do_roi_recons --do_voxel_recons $do_voxel_recons --do_tuning $do_tuning --do_sem_disc $do_sem_disc --fitting_type $fitting_type --alexnet_layer_name $alexnet_layer_name  --alexnet_padding_mode $alexnet_padding_mode  --use_pca_alexnet_feats $use_pca_alexnet_feats --fitting_type2 $fitting_type2 --semantic_discrim_type $semantic_discrim_type 
