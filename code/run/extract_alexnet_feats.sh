#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --exclude=mind-1-13
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

subj=1
debug=0
use_node_storage=0

source ~/myenv/bin/activate

cd /user_data/mmhender/imStat/code/feature_extraction/

python3 extract_alexnet_features.py --subject $subj --use_node_storage $use_node_storage --debug $debug
