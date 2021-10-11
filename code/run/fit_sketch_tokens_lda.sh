#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --exclude=mind-1-13
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

subj=1
volume_space=1
up_to_sess=40
ridge=1
sample_batch_size=100
voxel_batch_size=100
zscore_features=1

fitting_type=sketch_tokens
use_pca_st_feats=0
use_lda_st_feats=0
use_lda_animacy_st_feats=1
min_pct_var=100
max_pc_to_retain=11

do_stack=0

debug=0



cd $ROOT/code/model_fitting

python3 fit_model.py --subject $subj --volume_space $volume_space --up_to_sess $up_to_sess --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --zscore_features $zscore_features --ridge $ridge --use_pca_st_feats $use_pca_st_feats --use_lda_st_feats $use_lda_st_feats --use_lda_animacy_st_feats $use_lda_animacy_st_feats --min_pct_var $min_pct_var --max_pc_to_retain $max_pc_to_retain --debug $debug --fitting_type $fitting_type --do_stack $do_stack
