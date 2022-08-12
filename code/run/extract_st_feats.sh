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
save_dir='"/user_data/mmhender/features/sketch_tokens/"'
# save_dir='"/scratch/mmhender/features/sketch_tokens/"'

image_dir=/user_data/mmhender/nsd/stimuli/
# subject_list=(1 2 3 4 5 6 7 8 999)
subject_list=(1)

debug=0
use_node_storage=0
which_prf_grid=5
batch_size=100
grayscale=1


for subject in ${subject_list[@]}
do

    
    image_filename='"'${image_dir}S${subject}_stimuli_240.h5py'"'
    
    cd ${code_path}/matlab/
    image_set='"'S${subject}'"'
    
    matlab -nodisplay -nodesktop -nosplash -r "get_st_features_wrapper($image_set,$image_filename,$save_dir,$sketchtokens_dir,$toolbox_dir,$batch_size,$grayscale,$debug); exit"

    cd ${code_path}
    
    python3 extract_sketch_token_features.py --subject $subject --use_node_storage $use_node_storage --debug $debug --which_prf_grid $which_prf_grid --batch_size $batch_size --grayscale $grayscale

#     fn_features_node=/scratch/mmhender/features/sketch_tokens/S${subject}_features_each_prf_grid5.h5py
#     fn_features_local=/user_data/mmhender/features/sketch_tokens/S${subject}_features_each_prf_grid5.h5py
    
#     echo Copying file back from $fn_features_node to $fn_features_local
#     scp $fn_features_node $fn_features_local
#     echo Done copying file!
#     rm $fn_features_node

#     fn_edges_node=/scratch/mmhender/features/sketch_tokens/S${subject}_edges_240.h5py
#     rm $fn_edges_node   
#     fn_features_big_node=/scratch/mmhender/features/sketch_tokens/S${subject}_features_240.h5py
    # rm $fn_features_big_node
    
done