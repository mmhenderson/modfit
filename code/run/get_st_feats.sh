#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodelist=mind-0-12
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

subject=1
debug=0
module load matlab-9.7
codepath=/user_data/mmhender/toolboxes/SketchTokens/
cd $codepath

subjects=(2 3 4 5 6 7 8)
for subject in ${subjects[@]}
do

    matlab -nodisplay -nodesktop -nosplash -r "get_st_features_wrapper($subject,$debug); exit"

    fn_edges_node=/scratch/mmhender/features/sketch_tokens/S${subject}_edges_240.h5py
    fn_edges_local=/user_data/mmhender/features/sketch_tokens/S${subject}_edges_240.h5py
    echo Copying file back from $fn_edges_node to $fn_edges_local
    scp $fn_edges_node $fn_edges_local
    echo Done copying file!
    rm $fn_edges_node
    
    fn_features_node=/scratch/mmhender/features/sketch_tokens/S${subject}_features_240.h5py
#     fn_features_local=/user_data/mmhender/features/sketch_tokens/S${subject}_features_240.h5py
#     echo Copying file back from $fn_features_node to $fn_features_local
#     scp $fn_features_node $fn_features_local
#     echo Done copying file!
    rm $fn_features_node
done