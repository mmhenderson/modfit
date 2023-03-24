import os, sys, argparse
import numpy as np

from utils import label_utils
from utils import default_paths

nsd_root = default_paths.nsd_root
labels_path = default_paths.stim_labels_root

print('nsd_root: %s'%nsd_root)
print('labels_path: %s'%labels_path)
 
if __name__ == '__main__':
    
    which_prf_grid=5
    
    subjects = list(np.arange(1,9))+[998]
        
    for subject in subjects:

        # higher-level semantic dimensions
        label_utils.make_indoor_outdoor_labels(subject=subject)
        label_utils.make_realworldsize_labels(subject=subject, which_prf_grid=which_prf_grid)
        label_utils.make_animacy_labels(subject=subject, which_prf_grid=which_prf_grid)
        label_utils.make_buildings_labels(subject=subject, which_prf_grid=which_prf_grid)
        label_utils.make_face_labels(subject=subject, which_prf_grid=which_prf_grid)
   
    label_utils.count_highlevel_labels(which_prf_grid=which_prf_grid)