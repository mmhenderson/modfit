import os, sys, argparse

from utils import nsd_utils
from utils import default_paths
from utils import coco_utils

nsd_root = default_paths.nsd_root
path_to_save = default_paths.stim_root
labels_path = default_paths.stim_labels_root
features_path = os.path.join(default_paths.root, 'features')

print('nsd_root: %s'%nsd_root)
print('path_to_save: %s'%path_to_save)
print('features_path: %s'%features_path)
print('labels_path: %s'%labels_path)

if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

if not os.path.exists(features_path):
    os.makedirs(features_path)
    
if not os.path.exists(labels_path):
    os.makedirs(labels_path)
    
sys.stdout.flush()

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
   
    args = parser.parse_args()

    print('debug=%d'%args.debug)
    
    coco_utils.get_coco_ids_indep_big(n_images=50000)
    
    coco_utils.prep_indep_coco_images_big(n_pix=240, debug=args.debug==1)
    
    coco_utils.prep_indep_coco_images_big(n_pix=100, debug=args.debug==1)
    

