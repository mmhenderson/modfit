#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

echo $SLURM_JOBID
echo $SLURM_NODELIST

source ~/myenv/bin/activate

# change this path
ROOT=/user_data/mmhender/modfit/

# put the code directory on your python path
PYTHONPATH=:${ROOT}code/${PYTHONPATH}

cd ${ROOT}code/model_fitting/

subjects=(1 2 3 4 5 6 7 8)
# subjects=(1)

debug=0
# debug=1
# up_to_sess=1
up_to_sess=40

use_precomputed_prfs=1
which_prf_grid=5

from_scratch=1
date_str=0
overwrite_sem_disc=0
do_val=1
do_tuning=0
do_sem_disc=1

fitting_type=clip
resnet_layer_name=best_layer
resnet_model_architecture=RN50
use_pca_resnet_feats=1
# n_resnet_blocks_include=8
n_resnet_blocks_include=4

for subject in ${subjects[@]}
do
    python3 fit_model.py --subject $subject --debug $debug --up_to_sess $up_to_sess --use_precomputed_prfs $use_precomputed_prfs --which_prf_grid $which_prf_grid --from_scratch $from_scratch --date_str $date_str --do_val $do_val --do_tuning $do_tuning --do_sem_disc $do_sem_disc --overwrite_sem_disc $overwrite_sem_disc --fitting_type $fitting_type --resnet_layer_name $resnet_layer_name --resnet_model_architecture $resnet_model_architecture --use_pca_resnet_feats $use_pca_resnet_feats --n_resnet_blocks_include $n_resnet_blocks_include
    
done
