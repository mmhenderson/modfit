#!/bin/bash
#SBATCH --partition=tarrq
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=96:00:00

source ~/myenv/bin/activate

CWD=$(pwd)
cd ../../
ROOT=$(pwd)

subj=1
roi=None

ridge=1

shuffle_images=0
random_images=0
random_voxel_data=0

sample_batch_size=20
voxel_batch_size=100
zscore_features=1
nonlin_fn=0
padding_mode=circular



# n_ori=8
# n_sf=4
# up_to_sess=1
# debug=1

n_ori=8
n_sf=4
up_to_sess=10
debug=0


fitting_type=texture
include_crosscorrs=1
include_autocorrs=1

cd $ROOT/code/model_fitting



python3 fit_model.py --subject $subj --roi $roi --up_to_sess $up_to_sess --n_ori $n_ori --n_sf $n_sf --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --zscore_features $zscore_features --nonlin_fn $nonlin_fn --padding_mode $padding_mode --debug $debug --fitting_type $fitting_type --shuffle_images $shuffle_images --random_images $random_images --random_voxel_data $random_voxel_data --ridge $ridge --include_autocorrs $include_autocorrs --include_crosscorrs $include_crosscorrs
