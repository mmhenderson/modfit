#!/bin/bash
#SBATCH --partition=tarrq
#SBATCH --mem=48G
#SBATCH --cpus-per-task=1
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

source ~/myenv/bin/activate

cd /user_data/mmhender/imStat/code/utils/

debug=0
which_prf_grid=5

python3 count_labels.py --debug $debug --which_prf_grid $which_prf_grid
