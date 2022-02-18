import sys
from utils import label_utils
import numpy as np
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=1,
                    help="number of the subject, 1-8")
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which prf grid to use")
   
    args = parser.parse_args()

    sys.stdout.flush()
   
    print('concat labels')
    label_utils.concat_labels_each_prf(subject=args.subject, \
                                      which_prf_grid=args.which_prf_grid, verbose=True)