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
    
    label_utils.count_total_coco_labels(which_prf_grid=which_prf_grid)