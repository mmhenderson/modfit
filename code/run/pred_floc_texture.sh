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

# subjects=(1 2 3 4 5 6 7 8)
subjects=(1)

debug=0
# debug=1

which_prf_grid=5

fitting_type=texture_pyramid

n_ori_pyr=4
n_sf_pyr=4
group_all_hl_feats=1

pyr_pca_type=pcaHL


image_set=floc

for subject in ${subjects[@]}
do
    
    python3 predict_other_ims.py --subject $subject --image_set $image_set --debug $debug --which_prf_grid $which_prf_grid --fitting_type $fitting_type --n_ori_pyr $n_ori_pyr --n_sf_pyr $n_sf_pyr --group_all_hl_feats $group_all_hl_feats --pyr_pca_type $pyr_pca_type
    
done