#!/bin/bash
#SBATCH --partition=tarrq
#SBATCH --mem=24G
#SBATCH --cpus-per-task=1
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

source ~/myenv/bin/activate

cd /user_data/mmhender/imStat/code/utils/

subject=1
debug=0
which_prf_grid=3

python3 get_prf_labels.py --subject $subject --debug $debug --which_prf_grid $which_prf_grid

which_prf_grid=4

python3 get_prf_labels.py --subject $subject --debug $debug --which_prf_grid $which_prf_grid


