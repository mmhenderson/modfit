#!/bin/bash
#SBATCH --partition=tarrq
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

debug=0
which_prf_grid=5

source ~/myenv/bin/activate

cd /user_data/mmhender/imStat/code/feature_extraction

sublist=(1)

for subject in ${sublist[@]}
do
    python3 get_sketch_token_residuals.py --subject $subject --which_prf_grid $which_prf_grid --debug $debug
    
done

