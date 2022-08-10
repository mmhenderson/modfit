#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=mind-1-1
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

cd ${ROOT}code/analyze_features/

debug=0
which_prf_grid=5

subjects=(999)
# subjects=(1)

# feature_types=(gabor_solo)
# feature_types=(pyramid_texture)
# feature_types=(sketch_tokens) 
# feature_types=(alexnet)
# feature_types=(clip)
feature_types=(color)

balance_downsample=1
# balance_downsample=0

for subj in ${subjects[@]}
do
    for feature_type in ${feature_types[@]}
    do
        python3 decode_categ_from_features.py --subject $subj --debug $debug --feature_type $feature_type --which_prf_grid $which_prf_grid --balance_downsample $balance_downsample
    done
    
done
