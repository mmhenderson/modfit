#!/bin/bash
#SBATCH --partition=tarrq
#SBATCH --mem=24G
#SBATCH --cpus-per-task=1
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

source ~/myenv/bin/activate

cd /user_data/mmhender/imStat/code/utils/run/

which_prf_grid=5
do_indoor=1
do_natural=1
do_size=1

# subjects=(1 2 3 4)
subjects=(999)

for subject in ${subjects[@]}
do
    echo $subject
    python3 get_highlevel_prf_labels.py --subject $subject --which_prf_grid $which_prf_grid --do_indoor $do_indoor --do_size $do_size --do_natural $do_natural

done