#!/bin/bash
#SBATCH --partition=tarrq
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=12-00:00:00


echo $SLURM_JOBID
echo $SLURM_NODELIST

source ~/myenv/bin/activate

module load matlab-9.7

sketchtokens_dir='"/user_data/mmhender/toolboxes/SketchTokens/"'
toolbox_dir='"/user_data/mmhender/toolboxes/toolbox/"'

code_path=/user_data/mmhender/modfit/code/feature_extraction/
save_dir=/user_data/mmhender/features/sketch_tokens/

image_dir=/lab_data/tarrlab/maggie/fLoc_stimuli/
image_set_list=(floc)

debug=0
use_node_storage=0
which_prf_grid=5
batch_size=100

for image_set in ${image_set_list[@]}
do

    image_filename='"'${image_dir}${image_set}_stimuli_240.h5py'"'
    
    cd ${code_path}/matlab/
    image_set_str='"'${image_set}'"'
    save_dir_str='"'${save_dir}'"'
    echo $save_dir_str
    
    matlab -nodisplay -nodesktop -nosplash -r "get_st_features_wrapper($image_set_str,$image_filename,$save_dir_str,$sketchtokens_dir,$toolbox_dir,$batch_size,$debug); exit"

    cd ${code_path}
    
    python3 extract_sketch_token_features.py --image_set $image_set --use_node_storage $use_node_storage --debug $debug --which_prf_grid $which_prf_grid --batch_size $batch_size


    fn_edges=${save_dir}${image_set}_features_240.h5py
    rm $fn_edges   
    fn_features_big=${save_dir}${image_set}_edges_240.h5py
    rm $fn_features
    
done