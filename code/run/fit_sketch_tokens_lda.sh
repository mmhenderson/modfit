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

debug=0

subj=1
volume_space=1
up_to_sess=40
ridge=1
sample_batch_size=100
voxel_batch_size=100
zscore_features=1

fitting_type=sketch_tokens
use_pca_st_feats=0
use_lda_st_feats=1
# lda_discrim_type=indoor_outdoor
lda_discrim_type=animal

use_precomputed_prfs=1
do_stack=0


cd $ROOT/code/model_fitting

python3 fit_model.py --subject $subj --volume_space $volume_space --up_to_sess $up_to_sess --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --zscore_features $zscore_features --ridge $ridge --use_pca_st_feats $use_pca_st_feats --use_lda_st_feats $use_lda_st_feats --lda_discrim_type $lda_discrim_type --debug $debug --fitting_type $fitting_type --do_stack $do_stack --use_precomputed_prfs $use_precomputed_prfs

lda_discrim_type=vehicle

python3 fit_model.py --subject $subj --volume_space $volume_space --up_to_sess $up_to_sess --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --zscore_features $zscore_features --ridge $ridge --use_pca_st_feats $use_pca_st_feats --use_lda_st_feats $use_lda_st_feats --lda_discrim_type $lda_discrim_type --debug $debug --fitting_type $fitting_type --do_stack $do_stack --use_precomputed_prfs $use_precomputed_prfs

lda_discrim_type=food

python3 fit_model.py --subject $subj --volume_space $volume_space --up_to_sess $up_to_sess --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --zscore_features $zscore_features --ridge $ridge --use_pca_st_feats $use_pca_st_feats --use_lda_st_feats $use_lda_st_feats --lda_discrim_type $lda_discrim_type --debug $debug --fitting_type $fitting_type --do_stack $do_stack --use_precomputed_prfs $use_precomputed_prfs
