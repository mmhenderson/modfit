#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

source ~/myenv/bin/activate

cd /user_data/mmhender/imStat/code/model_fitting/

# debug=0
# up_to_sess=40
debug=1
up_to_sess=1

subjects=(1)
# subjects=(1 2 3 4 5 6 7 8)

trial_subset_list=(all)
# trial_subset_list=(balance_orient_indoor_outdoor balance_orient_animacy balance_orient_real_world_size_binary)

which_prf_grid=5

for subject in ${subjects[@]}
do
    for trial_subset in ${trial_subset_list[@]}
    do

        python3 feature_corrs_raw.py --subject $subject --up_to_sess $up_to_sess --debug $debug --which_prf_grid $which_prf_grid --trial_subset $trial_subset
        
    done
    
done