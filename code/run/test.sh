#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --nodelist=mind-1-9
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

source ~/myenv/bin/activate

echo $SLURM_JOBID
echo $SLURM_NODELIST

cd /user_data/mmhender/imStat/code/utils/

rndval=0

python3 test_saving.py --rndval $rndval