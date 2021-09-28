#!/bin/bash
#SBATCH --partition=tarrq
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

subj=1
debug=0
type=texture_pyramid
zscore=0

source ~/myenv/bin/activate
cd ../
cd feature_extraction

python3 pca_feats.py --subject $subj --debug $debug --type $type --zscore $zscore

# type=sketch_tokens
# python3 pca_feats.py --subject $subj --debug $debug --type $type --zscore $zscore