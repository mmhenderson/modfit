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

# change this path
ROOT=/user_data/mmhender/modfit/

# put the code directory on your python path
PYTHONPATH=:${ROOT}code/${PYTHONPATH}

cd ${ROOT}code/model_fitting/

subjects=(1 2 3 4 5 6 7 8)
# subjects=(1)

debug=0
# debug=1

which_prf_grid=5

fitting_type=gabor_solo
use_precomputed_prfs=1

n_ori_gabor=12
n_sf_gabor=8
use_pca_gabor_feats=0

image_set=floc

for subject in ${subjects[@]}
do
    
    python3 predict_other_ims.py --subject $subject --image_set $image_set --use_precomputed_prfs $use_precomputed_prfs --debug $debug --which_prf_grid $which_prf_grid --fitting_type $fitting_type --n_ori_gabor $n_ori_gabor --n_sf_gabor $n_sf_gabor --use_pca_gabor_feats $use_pca_gabor_feats
    
done