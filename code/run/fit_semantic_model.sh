#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

source ~/myenv/bin/activate

cd /user_data/mmhender/imStat/code/model_fitting

debug=0
up_to_sess=40
which_prf_grid=5

subj=1
volume_space=1
ridge=0
sample_batch_size=100
voxel_batch_size=100

fitting_type=semantic
semantic_discrim_type=animacy
use_precomputed_prfs=1

do_fitting=1
date_str=0
do_val=1
do_tuning=1
do_sem_disc=1
save_pred_data=1

python3 fit_model.py --subject $subj --debug $debug --fitting_type $fitting_type --volume_space $volume_space --up_to_sess $up_to_sess --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --ridge $ridge --which_prf_grid $which_prf_grid --do_fitting $do_fitting --date_str $date_str --do_val $do_val --do_tuning $do_tuning --do_sem_disc $do_sem_disc --semantic_discrim_type $semantic_discrim_type --use_precomputed_prfs $use_precomputed_prfs --save_pred_data $save_pred_data

semantic_discrim_type=indoor_outdoor

python3 fit_model.py --subject $subj --debug $debug --fitting_type $fitting_type --volume_space $volume_space --up_to_sess $up_to_sess --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --ridge $ridge --which_prf_grid $which_prf_grid --do_fitting $do_fitting --date_str $date_str --do_val $do_val --do_tuning $do_tuning --do_sem_disc $do_sem_disc --semantic_discrim_type $semantic_discrim_type --use_precomputed_prfs $use_precomputed_prfs --save_pred_data $save_pred_data

