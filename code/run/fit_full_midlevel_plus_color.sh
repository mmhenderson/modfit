#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=mind-1-26
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

echo $SLURM_JOBID
echo $SLURM_NODELIST

source ~/myenv/bin/activate

# change this path
ROOT=/user_data/mmhender/modfit/

# put the code directory on your python path
PYTHONPATH=:${ROOT}code/${PYTHONPATH}

cd ${ROOT}code/model_fitting/

subjects=(1 2 3 4 5 6 7 8)
# subjects=(1)

debug=0
# debug=1
# up_to_sess=1
up_to_sess=40

use_precomputed_prfs=1
which_prf_grid=5

from_scratch=1
date_str=0
do_val=1
do_tuning=0
do_sem_disc=1

fitting_type=full_midlevel

n_ori_gabor=12
n_sf_gabor=8

n_ori_pyr=4
n_sf_pyr=4
group_all_hl_feats=1
pyr_pca_type=pcaHL

use_pca_st_feats=0

do_varpart=0
include_solo_models=0
do_pyr_varpart=0
set_lambda_per_group=0

fitting_type2=color

for subject in ${subjects[@]}
do
    
    python3 fit_model.py --subject $subject --debug $debug --up_to_sess $up_to_sess --use_precomputed_prfs $use_precomputed_prfs --which_prf_grid $which_prf_grid --from_scratch $from_scratch --date_str $date_str --do_val $do_val --do_tuning $do_tuning --do_sem_disc $do_sem_disc --fitting_type $fitting_type --n_ori_gabor $n_ori_gabor --n_sf_gabor $n_sf_gabor --n_ori_pyr $n_ori_pyr --n_sf_pyr $n_sf_pyr --group_all_hl_feats $group_all_hl_feats --pyr_pca_type $pyr_pca_type --use_pca_st_feats $use_pca_st_feats --do_varpart $do_varpart --include_solo_models $include_solo_models --do_pyr_varpart $do_pyr_varpart --set_lambda_per_group $set_lambda_per_group --fitting_type2 $fitting_type2
    
done