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
from tqdm import tqdm
import gc
import torch
import argparse

# import custom modules
code_dir = '/user_data/mmhender/imStat/code/'
sys.path.append(code_dir)
from utils import nsd_utils, roi_utils, default_paths, coco_utils

import initialize_fitting as initialize_fitting
import fwrf_predict

fpX = np.float32
device = initialize_fitting.init_cuda()

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

#################################################################################################
        
    
def get_discrim(subject=1, volume_space = True, up_to_sess = 1, single_sess=0, \
             shuffle_images = False, random_images = False, random_voxel_data = False, \
             debug = False, which_prf_grid=1):
    
    def save_all(fn2save):
        """
        Define all the important parameters that have to be saved
        """
        dict2save = {
        'subject': subject,
        'volume_space': volume_space,
        'voxel_mask': voxel_mask,
        'brain_nii_shape': brain_nii_shape,
        'image_order': image_order,
        'voxel_index': voxel_index,
        'voxel_roi': voxel_roi,
        'voxel_ncsnr': voxel_ncsnr, 
        'which_prf_grid': which_prf_grid,
        'models': models,        
        'debug': debug,
        'up_to_sess': up_to_sess,
        'single_sess': single_sess,
        'prf_fit_filename': prf_fit_filename,
        'discrim_each_axis': discrim_each_axis,
        }
        
        print('\nSaving to %s\n'%fn2save)
        torch.save(dict2save, fn2save, pickle_protocol=4)

    if single_sess==0:
        single_sess=None
      
    # First figure out name for where to save results   
    model_name = 'semantic_discrim_raw'
    output_dir, fn2save = initialize_fitting.get_save_path(subject, volume_space, model_name, shuffle_images, \
                                                           random_images, random_voxel_data, debug, date_str=None)
    
    # decide what voxels to use  
    voxel_mask, voxel_index, voxel_roi, voxel_ncsnr, brain_nii_shape = roi_utils.get_voxel_roi_info(subject, \
                                                            volume_space, include_all=True, include_body=True)

    if single_sess is not None:
        sessions = np.array([single_sess])
    else:
        sessions = np.arange(0,up_to_sess)
    zscore_betas_within_sess = True
    image_inds_only = True
    # get all data and corresponding images, in two splits. always fixed set that gets left out
    trn_stim_data, trn_voxel_data, val_stim_data, val_voxel_data, \
            image_order, image_order_trn, image_order_val = nsd_utils.get_data_splits(subject, \
                                      sessions=sessions, image_inds_only = image_inds_only, \
                                      voxel_mask=voxel_mask, volume_space=volume_space, \
                                      zscore_betas_within_sess=zscore_betas_within_sess, \
                                  shuffle_images=shuffle_images, random_images=random_images, \
                                    random_voxel_data=random_voxel_data)

    if image_inds_only==True:
        # For this model, the features are pre-computed, so we will just load them rather than passing in images.
        # Going to pass the image indices (into 10,000 dim array) instead of images to fitting and val functions, 
        # which will tell which features to load.
        trn_stim_data = image_order_trn
        val_stim_data = image_order_val
   
    # Params for the spatial aspect of the model (possible pRFs)
    models = initialize_fitting.get_prf_models(which_grid=which_prf_grid) 

    prf_fit_filename = os.path.join(default_paths.save_fits_path,'S%02d'%subject, 'sketch_tokens/Nov-11-2021_1659_27/all_fit_params')
    print('Loading pre-computed pRF estimates for all voxels from %s'%prf_fit_filename)
    out = torch.load(prf_fit_filename)
    best_params = out['best_params']
    assert(best_params[0].shape[0]==trn_voxel_data.shape[1])
    
    ### ESTIMATE SEMANTIC DISCRIMINABILITY OF EACH VOXEL'S PREDICTED RESPONSES ######
    sys.stdout.flush()
    gc.collect()
    torch.cuda.empty_cache()
    print('about to start semantic discriminability analysis')
    sys.stdout.flush()
#     labels_all = coco_utils.load_labels_each_prf(subject, which_prf_grid, \
#                                              image_inds=val_stim_data, models=models,verbose=False)
#     discrim_each_axis = fwrf_predict.get_semantic_discrim(best_params, labels_all, \
#                                   val_voxel_data_pred=val_voxel_data[:,:,np.newaxis], debug=debug)
    labels_all = coco_utils.load_labels_each_prf(subject, which_prf_grid, \
             image_inds=np.concatenate([trn_stim_data, val_stim_data], axis=0), models=models,verbose=False)
    discrim_each_axis = fwrf_predict.get_semantic_discrim(best_params, labels_all, \
              val_voxel_data_pred=np.concatenate([trn_voxel_data[:,:,np.newaxis], \
                                                  val_voxel_data[:,:,np.newaxis]],axis=0),\
                                                          debug=debug)

    save_all(fn2save)

            
    ########## SUPPORT FUNCTIONS HERE ###############

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=1,
                    help="number of the subject, 1-8")
    parser.add_argument("--volume_space", type=int, default=1,
                    help="want to do fitting with volume space or surface space data? 1 for volume, 0 for surface.")
    parser.add_argument("--up_to_sess", type=int,default=1,
                    help="analyze sessions 1-#")
    parser.add_argument("--single_sess", type=int,default=0,
                    help="analyze just this one session (enter integer)")
    
    parser.add_argument("--shuffle_images", type=int,default=0,
                    help="want to shuffle the images randomly (control analysis)? 1 for yes, 0 for no")
    parser.add_argument("--random_images", type=int,default=0,
                    help="want to use random gaussian values for images (control analysis)? 1 for yes, 0 for no")
    parser.add_argument("--random_voxel_data", type=int,default=0,
                    help="want to use random gaussian values for voxel data (control analysis)? 1 for yes, 0 for no")
    
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which grid of candidate prfs?")
   
    args = parser.parse_args()
    
    # print values of a few key things to the command line...
    if args.debug==1:
        print('USING DEBUG MODE...')
    if args.shuffle_images==1:
        print('\nWILL RANDOMLY SHUFFLE IMAGES\n')
    if args.random_images==1:
        print('\nWILL USE RANDOM NOISE IMAGES\n')
    if args.random_voxel_data==1:
        print('\nWILL USE RANDOM DATA\n')

    # now actually call the function to execute fitting...
 
    get_discrim(subject=args.subject, volume_space = args.volume_space==1, up_to_sess = args.up_to_sess, single_sess=args.single_sess, shuffle_images = args.shuffle_images==1, random_images = args.random_images==1, random_voxel_data=args.random_voxel_data==1, debug = args.debug==1, which_prf_grid=args.which_prf_grid)
             
