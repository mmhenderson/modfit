from utils import label_utils
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which prf grid to use")
   
    args = parser.parse_args()

    label_utils.count_labels_each_prf(which_prf_grid=args.which_prf_grid, debug=args.debug)
    label_utils.get_top_two_subcateg(which_prf_grid=args.which_prf_grid)