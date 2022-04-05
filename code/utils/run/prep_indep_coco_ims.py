from utils import coco_utils
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--debug", type=int,default=0)
   
    args = parser.parse_args()
    
    coco_utils.prep_indep_coco_images(n_pix=240, debug=args.debug==1)

