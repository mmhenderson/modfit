#!/bin/bash
#SBATCH --partition=tarrq
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

source ~/myenv/bin/activate

cd /user_data/mmhender/imStat/code/utils/

debug=0
sublist=(3 4 6 8)
for subject in ${sublist[@]}
do
    python3 get_ranked_images.py --subject $subject --debug $debug
done