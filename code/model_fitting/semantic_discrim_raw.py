"""
Compute the semantic discriminability for raw single-voxel activations (no model).
Based on semantic labels for voxel's pRF - pRFs are computed previously with FWRF
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

import initialize_fitting, arg_parser, fwrf_predict

try:
    device = initialize_fitting.init_cuda()
except:
    device = 'cpu:0'
    
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

#################################################################################################
        
    
def get_discrim(args):

    if args.use_all_data:
        model_name = 'semantic_discrim_raw_trnval'
    else:
        model_name = 'semantic_discrim_raw_val'
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
        'sem_discrim_each_axis': sem_discrim_each_axis,
        'sem_corr_each_axis': sem_corr_each_axis,
        'discrim_type_list': discrim_type_list,
        'n_sem_samp_each_axis': n_sem_samp_each_axis, 
        }
        
        print('\nSaving to %s\n'%fn2save)
        np.save(fn2save, dict2save, allow_pickle=True)

    sem_discrim_each_axis = None
    sem_corr_each_axis = None
    discrim_type_list = None
    n_sem_samp_each_axis = None
    
    ########## LOADING THE DATA #############################################################################
    # decide what voxels to use  
    voxel_mask, voxel_index, voxel_roi, voxel_ncsnr, brain_nii_shape = \
                                roi_utils.get_voxel_roi_info(args.subject, \
                                args.volume_space, include_all=True)

    if (args.single_sess is not None) and (args.single_sess!=0):
        sessions = np.array([args.single_sess])
    else:
        sessions = np.arange(0,args.up_to_sess)
    # Get all data and corresponding images, in two splits. Always a fixed set that gets left out
    trn_stim_data, trn_voxel_data, val_stim_data, val_voxel_data, \
    image_order, trn_image_order, val_image_order = \
                                nsd_utils.get_data_splits(args.subject, \
                                sessions=sessions, image_inds_only = True, \
                                voxel_mask=voxel_mask, volume_space=args.volume_space, \
                                zscore_betas_within_sess=True, \
                                shuffle_images=False, random_images=False, \
                                random_voxel_data=False)
    n_voxels = trn_voxel_data.shape[1]   
    
    ########## DEFINE PARAMETERS #############################################################################

    prf_models = initialize_fitting.get_prf_models(which_grid=args.which_prf_grid) 
    n_prfs = prf_models.shape[0]
    
    sys.stdout.flush()
   
    ########## LOAD PRECOMPUTED PRFS ##########################################################################
    
    # will need these estimates in order to get the appropriate semantic labels for each voxel, based
    # on where its spatial pRF is. 
    best_model_each_voxel, saved_prfs_fn = initialize_fitting.load_precomputed_prfs(args.subject)
    assert(len(best_model_each_voxel)==trn_voxel_data.shape[1])  
    best_params_tmp = [None, None, None, None, None, best_model_each_voxel[:,np.newaxis]]
    
    ### ESTIMATE SEMANTIC DISCRIMINABILITY #########################################################################
    sys.stdout.flush()
    
    print('\nStarting semantic discriminability analysis ...\n')
    sys.stdout.flush()
    if args.use_all_data:
        image_inds_use = np.concatenate([val_image_order, trn_image_order], axis=0)
        voxel_data_use = np.concatenate([val_voxel_data[:,:,np.newaxis], \
                                         trn_voxel_data[:,:,np.newaxis]], axis=0)
    else:
        image_inds_use = val_image_order
        voxel_data_use = val_voxel_data[:,:,np.newaxis]
    
    labels_all, discrim_type_list, unique_labs_each = \
            initialize_fitting.load_labels_each_prf(args.subject, args.which_prf_grid,\
                                            image_inds=image_inds_use, \
                                            models=prf_models,verbose=False, \
                                            debug=args.debug)
    print('size of voxel data:')
    print(voxel_data_use.shape)
    print('size of labels:')
    print(labels_all.shape)
    # Plug in the actual raw data here, not encoding model predictions.
    sem_discrim_each_axis, sem_corr_each_axis, n_sem_samp_each_axis = \
            fwrf_predict.get_semantic_discrim(best_params_tmp, \
                                              labels_all, unique_labs_each, \
                                              val_voxel_data_pred=voxel_data_use,\
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
    
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which grid of candidate prfs?")
     
    parser.add_argument("--debug",type=nice_str2bool,default=False,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    
    parser.add_argument("--use_all_data",type=nice_str2bool,default=False,
                    help="want to use both train and validation data? Otherwise just validation. 1 for yes, 0 for no")
    
    parser.add_argument("--shuffle_images", type=nice_str2bool,default=False,
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
    
    get_discrim(args)
   