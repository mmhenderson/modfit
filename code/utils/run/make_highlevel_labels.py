import os, sys, argparse
import numpy as np

from utils import label_utils
from utils import default_paths

nsd_root = default_paths.nsd_root
labels_path = default_paths.stim_labels_root

print('nsd_root: %s'%nsd_root)
print('labels_path: %s'%labels_path)
 
if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
   
   
    args = parser.parse_args()
    debug=args.debug==1
    
    print('debug=%d'%debug)
    
    which_prf_grid=5
    
    if debug:
        subjects = [1, 999]
    else:
        subjects = list(np.arange(1,9))+[999]
        
    for subject in subjects:

        label_utils.concat_highlevel_labels_each_prf(subject=subject, \
                                      which_prf_grid=which_prf_grid, verbose=True, debug=debug)
    
    label_utils.count_highlevel_labels(which_prf_grid=which_prf_grid, axes_to_do=[0,1,2,3,4], \
                           debug=debug)