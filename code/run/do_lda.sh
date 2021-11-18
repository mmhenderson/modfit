#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

subj=1
debug=0
which_prf_grid=5

feature_type=sketch_tokens

source ~/myenv/bin/activate
cd ../
cd feature_extraction

discrim_type=animacy
python3 linear_discr.py --subject $subj --debug $debug --feature_type $feature_type --discrim_type $discrim_type --which_prf_grid $which_prf_grid

discrim_type=all_supcat
python3 linear_discr.py --subject $subj --debug $debug --feature_type $feature_type --discrim_type $discrim_type --which_prf_grid $which_prf_grid

discrim_type=indoor_outdoor
python3 linear_discr.py --subject $subj --debug $debug --feature_type $feature_type --discrim_type $discrim_type --which_prf_grid $which_prf_grid

discrim_type=person
python3 linear_discr.py --subject $subj --debug $debug --feature_type $feature_type --discrim_type $discrim_type --which_prf_grid $which_prf_grid

discrim_type=animal
python3 linear_discr.py --subject $subj --debug $debug --feature_type $feature_type --discrim_type $discrim_type --which_prf_grid $which_prf_grid

discrim_type=food
python3 linear_discr.py --subject $subj --debug $debug --feature_type $feature_type --discrim_type $discrim_type --which_prf_grid $which_prf_grid

discrim_type=vehicle
python3 linear_discr.py --subject $subj --debug $debug --feature_type $feature_type --discrim_type $discrim_type --which_prf_grid $which_prf_grid
