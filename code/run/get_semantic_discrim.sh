#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

source ~/myenv/bin/activate

cd /user_data/mmhender/imStat/code/model_fitting/

debug=0
up_to_sess=40

subj=1
which_prf_grid=5

use_all_data=0

python3 semantic_discrim_raw.py --subject $subj --up_to_sess $up_to_sess --debug $debug --which_prf_grid $which_prf_grid --use_all_data $use_all_data

use_all_data=1

python3 semantic_discrim_raw.py --subject $subj --up_to_sess $up_to_sess --debug $debug --which_prf_grid $which_prf_grid --use_all_data $use_all_data
