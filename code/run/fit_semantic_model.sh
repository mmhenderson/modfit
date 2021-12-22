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

subj=1
debug=0
fitting_type=semantic
volume_space=1
up_to_sess=40
sample_batch_size=100
voxel_batch_size=100
ridge=0
which_prf_grid=5
do_fitting=1
date_str=0
do_val=1
do_tuning=1
do_sem_disc=1
use_precomputed_prfs=1

semantic_discrim_type=natural_humanmade
# semantic_discrim_type=food
# semantic_discrim_type=vehicle
# semantic_discrim_type=person
# semantic_discrim_type=indoor_outdoor
# semantic_discrim_type=all_supcat

python3 fit_model.py --subject $subj --debug $debug --fitting_type $fitting_type --volume_space $volume_space --up_to_sess $up_to_sess --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --ridge $ridge --which_prf_grid $which_prf_grid --do_fitting $do_fitting --date_str $date_str --do_val $do_val --do_tuning $do_tuning --do_sem_disc $do_sem_disc --use_precomputed_prfs $use_precomputed_prfs --semantic_discrim_type $semantic_discrim_type

