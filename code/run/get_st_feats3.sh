#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodelist=mind-0-33-1
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

debug=0
source ~/myenv/bin/activate
module load matlab-9.7
codepath1=/user_data/mmhender/toolboxes/SketchTokens/
codepath2=/user_data/mmhender/imStat/code/feature_extraction/
use_node_storage=0
which_prf_grid=5

subjects=(2)
for subject in ${subjects[@]}
do

#     fn_features_big_node=/scratch/mmhender/features/sketch_tokens/S${subject}_features_240.h5py
#     fn_features_big_local=/user_data/mmhender/features/sketch_tokens/S${subject}_features_240.h5py
#     scp $fn_features_big_node $fn_features_big_local
#     echo Done copying file!
    
#     cd $codepath1

#     matlab -nodisplay -nodesktop -nosplash -r "get_st_features_wrapper($subject,$debug); exit"

#     cd $codepath2
    
#     python3 extract_sketch_token_features.py --subject $subject --use_node_storage $use_node_storage --debug $debug --which_prf_grid $which_prf_grid

#     fn_features_node=/scratch/mmhender/features/sketch_tokens/S${subject}_features_each_prf_grid5.h5py
#     fn_features_local=/user_data/mmhender/features/sketch_tokens/S${subject}_features_each_prf_grid5.h5py
    
#     echo Copying file back from $fn_features_node to $fn_features_local
#     scp $fn_features_node $fn_features_local
#     echo Done copying file!
#     rm $fn_features_node

    fn_edges_node=/scratch/mmhender/features/sketch_tokens/S${subject}_edges_240.h5py
    rm $fn_edges_node   
    fn_features_big_node=/scratch/mmhender/features/sketch_tokens/S${subject}_features_240.h5py
    rm $fn_features_big_node
done