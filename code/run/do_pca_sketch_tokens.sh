#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
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

debug=0
type_list=(sketch_tokens)
subjects=(1 2 3 4 5 6 7 8)
# subjects=(1)
max_pc_to_retain=150
min_pct_var=95

which_prf_grid=5

for subject in ${subjects[@]}
do
    for type in ${type_list[@]}
    do

        python3 pca_feats.py --subject $subject --debug $debug --type $type --max_pc_to_retain $max_pc_to_retain --min_pct_var $min_pct_var --which_prf_grid $which_prf_grid
        
    done
    
done