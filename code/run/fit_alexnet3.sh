#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --exclude=mind-1-13
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
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

subj=1
volume_space=1
ridge=1
sample_batch_size=100
voxel_batch_size=100
zscore_features=1
zscore_in_groups=0

fitting_type=alexnet
alexnet_layer_name='Conv5_ReLU'
fitting_type2=sketch_tokens

do_stack=0
do_roi_recons=0
do_voxel_recons=0

cd $ROOT/code/model_fitting

python3 fit_model.py --subject $subj --debug $debug --fitting_type $fitting_type --volume_space $volume_space --up_to_sess $up_to_sess --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --zscore_features $zscore_features --zscore_in_groups $zscore_in_groups --ridge $ridge --do_stack $do_stack --do_roi_recons $do_roi_recons --do_voxel_recons $do_voxel_recons  --alexnet_layer_name $alexnet_layer_name --fitting_type2 $fitting_type2

alexnet_layer_name='Conv4_ReLU'
python3 fit_model.py --subject $subj --debug $debug --fitting_type $fitting_type --volume_space $volume_space --up_to_sess $up_to_sess --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --zscore_features $zscore_features --zscore_in_groups $zscore_in_groups --ridge $ridge --do_stack $do_stack --do_roi_recons $do_roi_recons --do_voxel_recons $do_voxel_recons  --alexnet_layer_name $alexnet_layer_name --fitting_type2 $fitting_type2

alexnet_layer_name='Conv3_ReLU'
python3 fit_model.py --subject $subj --debug $debug --fitting_type $fitting_type --volume_space $volume_space --up_to_sess $up_to_sess --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --zscore_features $zscore_features --zscore_in_groups $zscore_in_groups --ridge $ridge --do_stack $do_stack --do_roi_recons $do_roi_recons --do_voxel_recons $do_voxel_recons  --alexnet_layer_name $alexnet_layer_name --fitting_type2 $fitting_type2

alexnet_layer_name='Conv2_ReLU'
python3 fit_model.py --subject $subj --debug $debug --fitting_type $fitting_type --volume_space $volume_space --up_to_sess $up_to_sess --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --zscore_features $zscore_features --zscore_in_groups $zscore_in_groups --ridge $ridge --do_stack $do_stack --do_roi_recons $do_roi_recons --do_voxel_recons $do_voxel_recons  --alexnet_layer_name $alexnet_layer_name --fitting_type2 $fitting_type2

alexnet_layer_name='Conv1_ReLU'
python3 fit_model.py --subject $subj --debug $debug --fitting_type $fitting_type --volume_space $volume_space --up_to_sess $up_to_sess --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --zscore_features $zscore_features --zscore_in_groups $zscore_in_groups --ridge $ridge --do_stack $do_stack --do_roi_recons $do_roi_recons --do_voxel_recons $do_voxel_recons  --alexnet_layer_name $alexnet_layer_name --fitting_type2 $fitting_type2
