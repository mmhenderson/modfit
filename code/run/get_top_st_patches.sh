#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --gres=gpu:0
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

source ~/myenv/bin/activate
cd /user_data/mmhender/imStat/code/analyze_features

debug=0
which_prf_grid=5
top_n_images=96

python3 get_top_patches.py --top_n_images $top_n_images --which_prf_grid $which_prf_grid --debug $debug