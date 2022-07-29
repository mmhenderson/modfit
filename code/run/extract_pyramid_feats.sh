#!/bin/bash
#SBATCH --partition=tarrq
#SBATCH --gres=gpu:1
#SBATCH --exclude=mind-1-23
#SBATCH --mem=32G
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

cd ${ROOT}code/feature_extraction/

debug=1
use_node_storage=0
n_ori=4
n_sf=4
batch_size=50
which_prf_grid=5

sublist=(1)

# for subj in ${sublist[@]}
# do

#     python3 extract_pyramid_texture_features.py --subject $subj --use_node_storage $use_node_storage --n_ori=$n_ori --n_sf=$n_sf --batch_size $batch_size --debug $debug --which_prf_grid $which_prf_grid 

# done

image_set=floc

python3 extract_pyramid_texture_features.py --image_set $image_set --use_node_storage $use_node_storage --n_ori=$n_ori --n_sf=$n_sf --batch_size $batch_size --debug $debug --which_prf_grid $which_prf_grid 
