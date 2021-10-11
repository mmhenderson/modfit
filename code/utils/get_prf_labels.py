import sys
code_dir = '/user_data/mmhender/imStat/code/'
sys.path.append(code_dir)
from utils import coco_utils
import numpy as np

if __name__ == '__main__':
    
    for subject in np.arange(1, 9, 1):
        
        debug=False
        coco_utils.write_binary_labels_csv_within_prf(subject, min_overlap_pix=10, debug=debug, stuff=False)
        coco_utils.write_binary_labels_csv_within_prf(subject, min_overlap_pix=10, debug=debug, stuff=True)

