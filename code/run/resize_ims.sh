#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

source ~/myenv/bin/activate

CWD=$(pwd)
cd ../
ROOT=$(pwd)

cd $ROOT/utils
python3 resize_ims.py 

subj=1
debug=0
use_node_storage=0
n_ori=4
n_sf=4
batch_size=100

cd $ROOT/feature_extraction

python3 extract_pyramid_texture_features.py --subject $subj --use_node_storage $use_node_storage --n_ori=$n_ori --n_sf=$n_sf --batch_size $batch_size --debug $debug

