#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=12-00:00:00

source ~/myenv/bin/activate

CWD=$(pwd)
cd ../../
ROOT=$(pwd)

debug=0
up_to_sess=40

volume_space=1
ridge=1
sample_batch_size=100
voxel_batch_size=100
zscore_features=1

which_prf_grid=5
use_precomputed_prfs=1

from_scratch=1
do_val=1
date_str=0
do_tuning=1
do_sem_disc=1

fitting_type=clip
clip_layer_name='all_resblocks'
clip_model_architecture='RN50'
use_pca_clip_feats=1

cd $ROOT/code/model_fitting

subjects=(8)

# subjects=(3 4 5 6 7 8)
for subj in ${subjects[@]}
do

    python3 fit_model.py --subject $subj --debug $debug --volume_space $volume_space --up_to_sess $up_to_sess --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --zscore_features $zscore_features --ridge $ridge --which_prf_grid $which_prf_grid --use_precomputed_prfs $use_precomputed_prfs --from_scratch $from_scratch --do_val $do_val --date_str $date_str --do_tuning $do_tuning --do_sem_disc $do_sem_disc --fitting_type $fitting_type --clip_layer_name $clip_layer_name --clip_model_architecture $clip_model_architecture --use_pca_clip_feats $use_pca_clip_feats 

done