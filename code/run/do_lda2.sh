#!/bin/bash
#SBATCH --partition=tarrq
#SBATCH --exclude=mind-1-13
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

subj=1
debug=0

feature_type=sketch_tokens
do_features=1

source ~/myenv/bin/activate
cd ../
cd feature_extraction

discrim_type=all_supcat
python3 linear_discr.py --subject $subj --debug $debug --feature_type $feature_type --discrim_type $discrim_type --do_features $do_features
