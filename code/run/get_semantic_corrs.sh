#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

debug=0
which_prf_grid=5

source ~/myenv/bin/activate
cd /user_data/mmhender/imStat/code/analyze_features

python3 semantic_corrs.py --debug $debug --which_prf_grid $which_prf_grid
