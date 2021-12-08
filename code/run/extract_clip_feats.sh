#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

subj=1
debug=0
use_node_storage=0
which_prf_grid=5
min_pct_var=95
max_pc_to_retain=100


source ~/myenv/bin/activate

cd /user_data/mmhender/imStat/code/feature_extraction/

python3 extract_clip_features.py --subject $subj --use_node_storage $use_node_storage --debug $debug --which_prf_grid $which_prf_grid --min_pct_var $min_pct_var --max_pc_to_retain $max_pc_to_retain