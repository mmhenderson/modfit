import sys
code_dir = '/user_data/mmhender/imStat/code/'
sys.path.append(code_dir)
from utils import coco_utils
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
    parser.add_argument("--concat", type=int,default=1,
                    help="do you want to concatenate the different sets of labels?")
   
    args = parser.parse_args()

    if args.debug:
        print('DEBUG MODE\n')
    
    if args.concat==0:
    #     doing these labels separately for coco labels (objects) and coco-stuff
        coco_utils.write_binary_labels_csv_within_prf(subject=args.subject, min_overlap_pix=10, debug=args.debug, stuff=False, which_prf_grid=args.which_prf_grid)

        coco_utils.write_binary_labels_csv_within_prf(subject=args.subject, min_overlap_pix=10, debug=args.debug, stuff=True, which_prf_grid=args.which_prf_grid)

        coco_utils.write_indoor_outdoor_csv(subject=args.subject)
        coco_utils.write_natural_humanmade_csv(subject=args.subject, which_prf_grid=args.which_prf_grid)
        
    elif args.concat==1:
        coco_utils.concat_labels_each_prf(subject=args.subject, which_prf_grid=args.which_prf_grid, verbose=True)