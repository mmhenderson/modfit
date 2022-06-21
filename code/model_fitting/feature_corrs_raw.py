"""
Compute the feature selectivity (corr coef) for raw single-voxel activations (no model).
Based on the visual features for each voxel's pRF - pRFs are computed previously with FWRF
encoding model, and loaded here to get labels.
"""

# import basic modules
import sys
import os
import time
import numpy as np
import argparse
import distutils.util

# import custom modules
from utils import nsd_utils, roi_utils, default_paths

import initialize_fitting

from analyze_fits import feature_selectivity

from feature_extraction import fwrf_features

try:
    device = initialize_fitting.init_cuda()
except:
    device = 'cpu:0'
    
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

#################################################################################################
        
    
def get_corrs(args):

    model_name = 'corr_%s_raw'%args.feature_set
    model_name += '_%s'%args.trial_subset
    
    output_dir, fn2save = initialize_fitting.get_save_path(model_name, args)
    sys.stdout.flush()
    
    def save_all(fn2save):
    
        """
        Define all the important parameters that have to be saved
        """
        dict2save = {
        'subject': args.subject,
        'volume_space': args.volume_space,
        'voxel_mask': voxel_mask,
        'brain_nii_shape': brain_nii_shape,
        'image_order': image_order,
        'voxel_index': voxel_index,
        'voxel_roi': voxel_roi,
        'voxel_ncsnr': voxel_ncsnr, 
        'which_prf_grid': args.which_prf_grid,
        'models': prf_models,          
        'debug': args.debug,
        'up_to_sess': args.up_to_sess,
        'single_sess': args.single_sess,
        'best_model_each_voxel': best_model_each_voxel, 
        'saved_prfs_fn': saved_prfs_fn,
        'feature_set': args.feature_set,
        'corr_each_feature': corr_each_feature,
        }
        
        print('\nSaving to %s\n'%fn2save)
        np.save(fn2save, dict2save, allow_pickle=True)

    corr_each_feature = None
    
    
     ########## DEFINE PARAMETERS #############################################################################

    prf_models = initialize_fitting.get_prf_models(which_grid=args.which_prf_grid) 
    n_prfs = prf_models.shape[0]
    
    sys.stdout.flush()
    
    ########## LOADING THE DATA #############################################################################
    # decide what voxels to use  
    voxel_mask, voxel_index, voxel_roi, voxel_ncsnr, brain_nii_shape = \
                                roi_utils.get_voxel_roi_info(args.subject, \
                                args.volume_space)

    if (args.single_sess is not None) and (args.single_sess!=0):
        sessions = np.array([args.single_sess])
    else:
        sessions = np.arange(0,args.up_to_sess)
    # Get all data and corresponding images, in two splits. Always a fixed set that gets left out
    voxel_data, image_order, val_inds, holdout_inds, session_inds = \
                                nsd_utils.get_data_splits(args.subject, \
                                sessions=sessions, \
                                voxel_mask=voxel_mask, volume_space=args.volume_space, \
                                zscore_betas_within_sess=True, \
                                shuffle_images=False, \
                                random_voxel_data=False, \
                                average_image_reps = args.average_image_reps)
    trn_inds = ~val_inds & ~holdout_inds
    
    
    if args.trial_subset!='all':
        print('choosing a subset of trials to work with: %s'%args.trial_subset) 
        trn_trials_use, holdout_trials_use, val_trials_use = \
                initialize_fitting.get_subsampled_trial_order(image_order[trn_inds], \
                                                              image_order[holdout_inds],\
                                                              image_order[val_inds], \
                                                              args=args, \
                                                              index=0, trn_only=True)
        
        # Use trn set trials, because once we have subsampled there 
        # might not be enough val set trials
        image_inds_use = image_order[trn_inds]
        voxel_data_use = voxel_data[trn_inds,:]
        trials_use = trn_trials_use       
        print('min trn trials: %d'%np.min(np.sum(trn_trials_use, axis=0)))
        
    else:
        # default case - using all validation set trials, no sub-sampling.
        image_inds_use = image_order[val_inds]
        voxel_data_use = voxel_data[val_inds,:]    
        trials_use = None
        
    voxel_data_use = voxel_data_use[:,:,np.newaxis]
        
    n_voxels = voxel_data_use.shape[1]   
    
   
   
    ########## LOAD PRECOMPUTED PRFS ##########################################################################
    
    # will need these estimates in order to get the appropriate feature labels for each voxel, based
    # on where its spatial pRF is. 
    best_model_each_voxel, saved_prfs_fn = initialize_fitting.load_precomputed_prfs(args.subject, args)
    assert(len(best_model_each_voxel)==n_voxels)  
    
    ########## CREATE FEATURE LOADERS ###################################################################
    # these help to load sets of pre-computed features in an organized way.
    # first making a list of all the modules of interest (different feature spaces)

    if args.feature_set=='gabor_solo':
    
        feat_loader = fwrf_features.fwrf_feature_loader(subject=args.subject,\
                                                        which_prf_grid=args.which_prf_grid,\
                                                        feature_type='gabor_solo',\
                                                        n_ori=12, n_sf=8,\
                                                        nonlin_fn=True, \
                                                        use_pca_feats=False)
    else:
        raise ValueError("args.feature_set must be gabor solo, or need to update this code")
                         
    ### ESTIMATE FEATURE SELECTIVITY #########################################################################
    sys.stdout.flush()
    
    features_each_prf = fwrf_features.get_features_each_prf(image_inds = image_inds_use, \
                                                        feature_loader=feat_loader, \
                                                        zscore=True, debug=args.debug, \
                                                        dtype=np.float32)
            
    corr_each_feature = feature_selectivity.get_feature_tuning(best_prf_inds=best_model_each_voxel[:,np.newaxis] ,\
                                                        features_each_prf=features_each_prf, \
                                                        val_voxel_data_pred = voxel_data_use, \
                                                        trials_use_each_prf = trials_use, \
                                                        debug=args.debug)
    save_all(fn2save)
    
    # Done!

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    def nice_str2bool(x):
        return bool(distutils.util.strtobool(x))

    parser.add_argument("--subject", type=int,default=1,
                    help="number of the subject, 1-8")
    parser.add_argument("--volume_space", type=nice_str2bool, default=True,
                    help="want to do fitting with volume space or surface space data? 1 for volume, 0 for surface.")
    parser.add_argument("--up_to_sess", type=int,default=1,
                    help="analyze sessions 1-#")
    parser.add_argument("--single_sess", type=int, default=0,
                    help="analyze just this one session (enter integer)")
    parser.add_argument("--average_image_reps",type=nice_str2bool,default=True,
                    help="Want to average over 3 repetitions of same image? 1 for yes, 0 for no")
    
    parser.add_argument("--feature_set", type=str,default='gabor_solo', 
                       help="Which feature set to use?")
    
    parser.add_argument("--trial_subset", type=str,default='all', 
                    help="fit for a subset of trials only? default all trials")
   
    
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which grid of candidate prfs?")
    parser.add_argument("--prfs_model_name", type=str, default='gabor', 
                    help="model the prfs are from?")
    
    parser.add_argument("--debug",type=nice_str2bool,default=False,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    
    
    parser.add_argument("--from_scratch",type=nice_str2bool,default=True,
                    help="Starting from scratch? 1 for yes, 0 for no")
    
    parser.add_argument("--shuffle_images_once", type=nice_str2bool,default=False,
                    help="want to shuffle the images randomly (control analysis)? 1 for yes, 0 for no")
    parser.add_argument("--random_images", type=nice_str2bool,default=False,
                    help="want to use random gaussian values for images (control analysis)? 1 for yes, 0 for no")
    parser.add_argument("--random_voxel_data", type=nice_str2bool,default=False,
                    help="want to use random gaussian values for voxel data (control analysis)? 1 for yes, 0 for no")
    parser.add_argument("--date_str", type=str,default='',
                    help="what date was the model fitting done (only if you're starting from validation step.)")

    
    args = parser.parse_args()
    if args.debug==1:
        print('USING DEBUG MODE...')
    
    get_corrs(args)
   