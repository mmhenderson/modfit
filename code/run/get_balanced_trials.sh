#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

echo $SLURM_JOBID
echo $SLURM_NODELIST

source ~/myenv/bin/activate

cd /user_data/mmhender/imStat/code/model_fitting

# subject_list=(1)
# subject_list=()
# subject_list=(1 2 3 4 5 6 7 8)
subject_list=(999)
debug=0
n_samp_iters=10
balance_orient_categ=0
balance_freq_categ=0
balance_for_decoding=1

for subject in ${subject_list[@]}
do
    python3 subsample_trials.py --subject $subject --debug $debug --n_samp_iters $n_samp_iters --balance_orient_categ $balance_orient_categ --balance_freq_categ $balance_freq_categ --balance_for_decoding $balance_for_decoding
    
done