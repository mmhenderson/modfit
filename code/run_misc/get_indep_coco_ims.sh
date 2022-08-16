#!/bin/bash
#SBATCH --partition=tarrq
#SBATCH --mem=24G
#SBATCH --cpus-per-task=1
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

source ~/myenv/bin/activate

cd /user_data/mmhender/modfit/code/utils/run/

debug=0

python3 prep_indep_coco_ims.py --debug $debug
