#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --exclude=mind-1-7
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

debug=1
up_to_sess=1
single_sess=0
which_prf_grid=5

subj=1
volume_space=1
ridge=1
sample_batch_size=100
voxel_batch_size=100
zscore_features=1
use_precomputed_prfs=1
shuff_rnd_seed=0

fitting_type=gabor_solo
gabor_nonlin_fn=1

do_tuning=1
do_sem_disc=1

n_ori_gabor=12
n_sf_gabor=8

do_stack=0
do_roi_recons=0
do_voxel_recons=0
shuffle_images=0

cd $ROOT/code/model_fitting

python3 fit_model.py --subject $subj --debug $debug --fitting_type $fitting_type --volume_space $volume_space --up_to_sess $up_to_sess --single_sess $single_sess --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --zscore_features $zscore_features --ridge $ridge --n_ori_gabor $n_ori_gabor --n_sf_gabor $n_sf_gabor --gabor_nonlin_fn $gabor_nonlin_fn --do_stack $do_stack --do_roi_recons $do_roi_recons --do_voxel_recons $do_voxel_recons --which_prf_grid $which_prf_grid --do_tuning $do_tuning --do_sem_disc $do_sem_disc --shuffle_images $shuffle_images --use_precomputed_prfs $use_precomputed_prfs --shuff_rnd_seed $shuff_rnd_seed
