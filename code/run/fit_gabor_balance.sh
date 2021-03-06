#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

echo $SLURM_JOBID
echo $SLURM_NODELIST

source ~/myenv/bin/activate

cd /user_data/mmhender/imStat/code/model_fitting

# subjects=(1)
subjects=(1 2 3 4 5 6 7 8)
# subjects=(1)

# debug=1
# up_to_sess=1
debug=0
up_to_sess=40

average_image_reps=1
compute_sessionwise_r2=0

# trial_subset_list=(balance_orient_indoor_outdoor balance_orient_animacy balance_orient_real_world_size_binary)
trial_subset_list=(balance_freq_indoor_outdoor balance_freq_animacy balance_freq_real_world_size_binary)

sample_batch_size=100
voxel_batch_size=100
zscore_features=1
ridge=1
use_precomputed_prfs=1
which_prf_grid=5
from_scratch=1
date_str=0
overwrite_sem_disc=0
do_val=1
do_tuning=1
do_sem_disc=0

fitting_type=gabor_solo

n_ori_gabor=12
n_sf_gabor=8

use_pca_gabor_feats=0

gabor_nonlin_fn=1

for subject in ${subjects[@]}
do

    for trial_subset in ${trial_subset_list[@]}
    do

        python3 fit_model.py --subject $subject --debug $debug --up_to_sess $up_to_sess --average_image_reps $average_image_reps --compute_sessionwise_r2 $compute_sessionwise_r2 --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --zscore_features $zscore_features --ridge $ridge --use_precomputed_prfs $use_precomputed_prfs --which_prf_grid $which_prf_grid --from_scratch $from_scratch --date_str $date_str --do_val $do_val --do_tuning $do_tuning --do_sem_disc $do_sem_disc --overwrite_sem_disc $overwrite_sem_disc --fitting_type $fitting_type --n_ori_gabor $n_ori_gabor --n_sf_gabor $n_sf_gabor --gabor_nonlin_fn $gabor_nonlin_fn --use_pca_gabor_feats $use_pca_gabor_feats --trial_subset $trial_subset 
        
    done
    
done