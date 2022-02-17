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
    parser.add_argument("--do_indoor", type=int,default=1,
                    help="do indoor vs outdoor labels?")
    parser.add_argument("--do_natural", type=int,default=1,
                    help="do natural vs human-made labels?")
    parser.add_argument("--do_size", type=int,default=1,
                    help="do real-world-size labels?")
   
    args = parser.parse_args()
    sys.stdout.flush()
    
    if args.do_indoor==1:
        label_utils.write_indoor_outdoor_csv(subject=args.subject)
    if args.do_natural==1:
        label_utils.write_natural_humanmade_csv(subject=args.subject, which_prf_grid=args.which_prf_grid)
    if args.do_size==1:
        label_utils.write_realworldsize_csv(subject=args.subject, which_prf_grid=args.which_prf_grid)
