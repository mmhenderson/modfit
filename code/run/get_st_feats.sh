#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodelist=mind-0-33-2
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

matlab -nodisplay -nodesktop -nosplash -r "get_st_features_wrapper($subject,$debug); exit"
