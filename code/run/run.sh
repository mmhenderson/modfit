#!/bin/bash
#SBATCH --partition=tarrq
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

source ~/myenv/bin/activate

CWD=$(pwd)
cd ../../
ROOT=$(pwd)

subj=1
roi=None

ridge=1

shuffle_images=0
random_images=0
random_voxel_data=0

sample_batch_size=200
voxel_batch_size=100
zscore_features=1
nonlin_fn=0

n_ori=4
n_sf=4
up_to_sess=1
debug=0
shuff_rnd_seed=0
# shuff_rnd_seed=251709

fitting_type=pyramid_texture

do_fitting=1
do_val=1
do_partial=1

date_str='None'
# date_str='Jul-21-2021_0533'
# date_str='Jul-09-2021_1804'
# date_str='Jul-09-2021_1725'
# date_str='Jul-09-2021_1736'
# date_str='Jul-09-2021_1653'
# date_str='Jul-06-2021_0356'

cd $ROOT/code/model_fitting

python3 fit_model.py --subject $subj --roi $roi --up_to_sess $up_to_sess --n_ori $n_ori --n_sf $n_sf --sample_batch_size $sample_batch_size --voxel_batch_size $voxel_batch_size --zscore_features $zscore_features --nonlin_fn $nonlin_fn --debug $debug --fitting_type $fitting_type --shuffle_images $shuffle_images --random_images $random_images --random_voxel_data $random_voxel_data --ridge $ridge --do_fitting $do_fitting --do_val $do_val --do_partial $do_partial --date_str $date_str --shuff_rnd_seed $shuff_rnd_seed
