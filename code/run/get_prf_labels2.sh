#!/bin/bash
#SBATCH --partition=tarrq
#SBATCH --mem=24G
#SBATCH --cpus-per-task=1
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

source ~/myenv/bin/activate

cd /user_data/mmhender/imStat/code/utils/run/

debug=0
which_prf_grid=5

# subjects=(1)
subjects=(6 7 8)
for subject in ${subjects[@]}
do
    echo $subject
    python3 get_binary_prf_labels.py --subject $subject --debug $debug --which_prf_grid $which_prf_grid
done
