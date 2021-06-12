#!/bin/bash
#SBATCH --partition=tarrq
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=48:00:00

source ~/myenv/bin/activate

CWD=$(pwd)
cd ../../
ROOT=$(pwd)

subj=1
roi=None

n_ori=2
n_sf=2
up_to_sess=1
debug=1

# n_ori=8
# n_sf=6
# up_to_sess=20
# debug=False

sample_batch_size=20
voxel_batch_size=100
zscore_features=0
nonlin_fn=1
padding_mode=circular
fitting_type=gabor_combs
shuffle_images=0
random_images=0
random_voxel_data=0
ridge=1

cd code/model_fitting
python3 fit_model.py --subject $subj --roi $roi --up_to_sess $up_to_sess --n_ori $n_ori --n_sf $n_sf --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --zscore_features $zscore_features --nonlin_fn $nonlin_fn --padding_mode $padding_mode --debug $debug --fitting_type $fitting_type --shuffle_images $shuffle_images --random_images $random_images --random_voxel_data $random_voxel_data --ridge $ridge
