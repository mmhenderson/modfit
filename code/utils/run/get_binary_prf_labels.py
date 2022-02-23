import sys
from utils import label_utils
import numpy as np
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=1,
                    help="number of the subject, 1-8")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which prf grid to use")
   
    args = parser.parse_args()

    sys.stdout.flush()
    if args.debug:
        print('DEBUG MODE\n')
    
   
    # doing these labels separately for coco labels (objects) and coco-stuff
    label_utils.write_binary_labels_csv_within_prf(subject=args.subject, min_pix=10, debug=args.debug, stuff=False, which_prf_grid=args.which_prf_grid)

    label_utils.write_binary_labels_csv_within_prf(subject=args.subject, min_pix=10, debug=args.debug, stuff=True, which_prf_grid=args.which_prf_grid)
