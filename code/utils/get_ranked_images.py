import sys
from utils import nsd_utils
import numpy as np
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=1,
                    help="number of the subject, 1-8")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    
    args = parser.parse_args()

    if args.debug:
        print('DEBUG MODE\n')
    
    # doing these labels separately for coco labels (objects) and coco-stuff
    nsd_utils.get_image_ranks(subject=args.subject, debug=args.debug==1)

