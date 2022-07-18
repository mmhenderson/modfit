#!/bin/bash
#SBATCH --partition=tarrq
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --exclude=mind-1-13
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

echo $SLURM_JOBID
echo $SLURM_NODELIST

source ~/myenv/bin/activate

cd /user_data/mmhender/modfit/code/model_fitting

# subjects=(1 2 3 4 5 6 7 8)
subjects=(1)
up_to_sess=20

debug=0
# debug=1
# up_to_sess=1

average_image_reps=1
# compute_sessionwise_r2=0
save_model_residuals=0

sample_batch_size=100
voxel_batch_size=100
zscore_features=0
ridge=1
use_precomputed_prfs=1
which_prf_grid=5

date_str=0
overwrite_sem_disc=0

do_val=1
from_scratch=1
do_tuning=0
do_sem_disc=0

fitting_type=gabor_solo

n_ori_gabor=4
n_sf_gabor=4
# n_ori_gabor=12
# n_sf_gabor=8

use_pca_gabor_feats=0

gabor_nonlin_fn=1

for subject in ${subjects[@]}
do
  
   
    python3 fit_model.py --subject $subject --debug $debug --up_to_sess $up_to_sess --average_image_reps $average_image_reps --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --zscore_features $zscore_features --ridge $ridge --use_precomputed_prfs $use_precomputed_prfs --which_prf_grid $which_prf_grid --from_scratch $from_scratch --date_str $date_str --do_val $do_val --do_tuning $do_tuning --do_sem_disc $do_sem_disc --overwrite_sem_disc $overwrite_sem_disc --fitting_type $fitting_type --n_ori_gabor $n_ori_gabor --n_sf_gabor $n_sf_gabor --gabor_nonlin_fn $gabor_nonlin_fn --use_pca_gabor_feats $use_pca_gabor_feats --save_model_residuals $save_model_residuals 
    
done