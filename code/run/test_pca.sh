#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

debug=1
# type_list=(gabor)
type_list=(sketch_tokens)
# subjects=(1 2 3 4 5 6 7 8)
subjects=(1)
max_pc_to_retain=96
min_pct_var=95

which_prf_grid=5

source ~/myenv/bin/activate
cd /user_data/mmhender/modfit/code/feature_extraction

for subject in ${subjects[@]}
do
    for type in ${type_list[@]}
    do

        python3 pca_feats_new.py --subject $subject --debug $debug --type $type --max_pc_to_retain $max_pc_to_retain --min_pct_var $min_pct_var --which_prf_grid $which_prf_grid
        
    done
    
done