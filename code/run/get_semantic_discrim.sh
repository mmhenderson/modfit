#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

source ~/myenv/bin/activate

CWD=$(pwd)
cd ../../
ROOT=$(pwd)

debug=0
up_to_sess=40
single_sess=0
subj=1
volume_space=1
which_prf_grid=5

cd $ROOT/code/model_fitting

python3 semantic_discrim_raw.py --subject $subj --volume_space $volume_space --up_to_sess $up_to_sess --single_sess $single_sess --debug $debug --which_prf_grid $which_prf_grid 
