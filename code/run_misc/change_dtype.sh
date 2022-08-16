#!/bin/bash
#SBATCH --partition=tarrq
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

source ~/myenv/bin/activate

cd ../utils
python3 change_dtype.py
