#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=mind-1-3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

subj=1
debug=0
use_node_storage=1

source ~/myenv/bin/activate

CWD=$(pwd)
cd ../
ROOT=$(pwd)
cd $ROOT/model_src


if [ $use_node_storage == 1 ]
then

    remote_feature_path=/user_data/mmhender/features/sketch_tokens
    local_feature_path=/scratch/mmhender/features/sketch_tokens
    remote_file="${remote_feature_path}""/S""${subj}""_features_227.h5py"
    local_file="${local_feature_path}""/S""${subj}""_features_227.h5py"
    if [ -f ${local_file} ]
    then
        echo File already exists in the node scratch folder
    else
        if [ ! -d $local_feature_path ]
        then
            echo Making new directory at $local_feature_path
            mkdir -p $local_feature_path
        fi
        echo Copying big file from $remote_file to $local_file

        if [ $debug == 0 ]
        then            
            scp $remote_file $local_file
        fi

        echo Done copying big file, starting script...
    fi
fi

python3 extract_sketch_token_features.py --subject $subj --use_node_storage $use_node_storage --debug $debug

if [ $use_node_storage == 1 ]
then

    echo Copying output file from $local_feature_path to $remote_feature_path
   
    if [ $debug == 0 ]
    then
        scp "${local_feature_path}""/S""${subj}""_features_each_prf.h5py" "${remote_feature_path}""/S""${subj}""_features_each_prf.h5py"
    fi
    
    echo Done copying file back!  
    
fi