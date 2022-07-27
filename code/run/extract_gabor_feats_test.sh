#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

source ~/myenv/bin/activate

# change this path
ROOT=/user_data/mmhender/

# put the code directory on your python path
PYTHONPATH=:${ROOT}modfit/code/${PYTHONPATH}
echo $PYTHONPATH

# go to folder where script is located
cd ${ROOT}modfit/code/feature_extraction

# to test the code, use debug=1
debug=1
use_node_storage=0
n_ori=12
n_sf=8

sample_batch_size=100
which_prf_grid=5
gabor_solo=1
nonlin_fn=1

sublist=(1)
# sublist=(1 2 3 4 5 6 7 8 999)

for subject in ${sublist[@]}
do
    python3 extract_gabor_features.py --subject $subject --n_ori=$n_ori --n_sf=$n_sf --sample_batch_size $sample_batch_size --use_node_storage $use_node_storage --which_prf_grid $which_prf_grid --nonlin_fn $nonlin_fn --debug $debug
    
done

