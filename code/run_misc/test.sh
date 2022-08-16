#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --nodelist=mind-1-9
#SBATCH --mem=1G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

source ~/myenv/bin/activate

echo $SLURM_JOBID
echo $SLURM_NODELIST

cd /user_data/mmhender/imStat/code/model_fitting

# prf_fixed_sigma=0

# python3 test_arg.py --prf_fixed_sigma $prf_fixed_sigma

python3 test_arg.py

prf_fixed_sigma=0

python3 test_arg.py --prf_fixed_sigma $prf_fixed_sigma