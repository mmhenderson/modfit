import argparse
import numpy as np
import sys, os
import torch
import time
import h5py
import torch.nn as nn
from sklearn import decomposition

#import custom modules
code_dir = '/user_data/mmhender/imStat/code/'
sys.path.append(code_dir)
from utils import prf_utils, torch_utils, texture_utils, default_paths
from model_fitting import initialize_fitting
from feature_extraction import sketch_token_features

device = initialize_fitting.init_cuda()

    
def extract_features(subject, use_node_storage=False, debug=False, which_prf_grid=1):
    
    if use_node_storage:
        sketch_token_feat_path = default_paths.sketch_token_feat_path_localnode
    else:
        sketch_token_feat_path = default_paths.sketch_token_feat_path

    # Params for the spatial aspect of the model (possible pRFs)
    aperture = 1.0
    aperture_rf_range = 1.1
    aperture, models = initialize_fitting.get_prf_models(aperture_rf_range=aperture_rf_range, \
                                                             which_grid=which_prf_grid)    
    
    # Fix these params
    map_resolution = 227  
    n_prf_sd_out = 2
    batch_size = 100
    mult_patch_by_prf = True
    do_avg_pool = True
      
    features_file = os.path.join(sketch_token_feat_path, 'S%d_features_%d.h5py'%(subject, map_resolution))
    if not os.path.exists(features_file):
        raise RuntimeError('Looking at %s for precomputed features, not found.'%features_file)
        
    features_each_prf = sketch_token_features.get_features_each_prf(features_file, models, \
                            mult_patch_by_prf=mult_patch_by_prf, \
                            do_avg_pool=do_avg_pool, batch_size=batch_size, aperture=aperture, \
                                            debug=debug, device=device)

    if which_prf_grid==1:
        fn2save = os.path.join(sketch_token_feat_path, 'S%d_features_each_prf.h5py'%(subject))
    else:
        fn2save = os.path.join(sketch_token_feat_path, 'S%d_features_each_prf_grid%d.h5py'%(subject, which_prf_grid))

    print('Writing prf features to %s\n'%fn2save)
    
    t = time.time()
    with h5py.File(fn2save, 'w') as data_set:
        dset = data_set.create_dataset("features", np.shape(features_each_prf), dtype=np.float64)
        data_set['/features'][:,:,:] = features_each_prf
        data_set.close()  
    elapsed = time.time() - t
    
    print('Took %.5f sec to write file'%elapsed)
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=1,
                    help="number of the subject, 1-8")
    parser.add_argument("--use_node_storage", type=int,default=0,
                    help="want to save and load from scratch dir on current node? 1 for yes, 0 for no")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which version of prf grid to use")
    
    args = parser.parse_args()
    
    extract_features(subject = args.subject, use_node_storage = args.use_node_storage, debug = args.debug==1, which_prf_grid=args.which_prf_grid)
