#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

source ~/myenv/bin/activate
cd /user_data/mmhender/imStat/code/analyze_features

debug=0

python3 analyze_st_cluster_ims.py --debug $debug

# python3 analyze_im_curvature.py --debug $debug