"""
Run the model fitting for Gabor FWRF model. 
Can also fit versions with extra features, or with pca applied to feature space.
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
root_dir   = os.path.dirname(os.getcwd())
sys.path.append(os.path.join(root_dir))
from model_src import fwrf_fit, fwrf_predict
import initialize_fitting

fpX = np.float32

#################################################################################################
  
def fit_gabor_fwrf(subject=1, roi='V1', up_to_sess=1, n_ori=36, n_sf=12, sample_batch_size=50, voxel_batch_size=100, zscore_features=True, nonlin_fn=False, ridge=True, padding_mode = 'circular', 
                   debug=False, shuffle_images=False, random_images=False, random_voxel_data=False):
    
    """ 
    Fit linear mapping directly from gabor feature space (within specified pRF) to voxel response.
    """
    
    
    device = initialize_fitting.init_cuda()
    nsd_root, stim_root, beta_root, mask_root = initialize_fitting.get_paths()

    
    if ridge==True:
        # ridge regression, testing several positive lambda values (default)
        model_name = 'gabor_ridge_%dori_%dsf'%(n_ori, n_sf)
    else:        
        # fixing lambda at zero, so it turns into ordinary least squares
        model_name = 'gabor_OLS_%dori_%dsf'%(n_ori, n_sf)
    output_dir, fn2save = initialize_fitting.get_save_path(root_dir, subject, model_name, shuffle_images, random_images, random_voxel_data, debug)
       
    # decide what voxels to use  
    voxel_mask, voxel_index, voxel_roi, voxel_ncsnr, brain_nii_shape = initialize_fitting.get_voxel_info(mask_root, beta_root, subject, roi)

    # get all data and corresponding images, in two splits. always fixed set that gets left out
    trn_stim_data, trn_voxel_data, val_stim_single_trial_data, val_voxel_single_trial_data, \
        n_voxels, n_trials_val, image_order = initialize_fitting.get_data_splits(nsd_root, beta_root, stim_root, subject, voxel_mask, up_to_sess, 
                                                                                 shuffle_images=shuffle_images, random_images=random_images, random_voxel_data=random_voxel_data)

    # Set up the filters
    _gaborizer, _fmaps_fn = initialize_fitting.get_feature_map_fn(n_ori, n_sf, padding_mode, device, nonlin_fn)

    # Params for the spatial aspect of the model (possible pRFs)
    aperture, models = initialize_fitting.get_prf_models()    
    
    # More params for fitting
    holdout_size, lambdas = initialize_fitting.get_fitting_pars(trn_voxel_data, zscore_features, ridge=ridge)

    #### DO THE ACTUAL MODEL FITTING HERE ####
    gc.collect()
    torch.cuda.empty_cache()
    best_losses, best_lambdas, best_params, covar_each_model_training = fwrf_fit.learn_params_ridge_regression(
        trn_stim_data, trn_voxel_data, _fmaps_fn, models, lambdas, \
        aperture=aperture, zscore=zscore_features, sample_batch_size=sample_batch_size, \
        voxel_batch_size=voxel_batch_size, holdout_size=holdout_size, shuffle=True, add_bias=True, debug=debug)
    # note there's also a shuffle param in the above fn call, that determines the nested heldout data for lambda and param selection. always using true.
    print('\nDone with training\n')

    # Validate model on held-out test set
    val_cc, val_r2 = fwrf_predict.validate_model(best_params, val_voxel_single_trial_data, val_stim_single_trial_data, _fmaps_fn, sample_batch_size, voxel_batch_size, aperture, dtype=fpX)
    
    # As a less model-sensitive way of assessing tuning, directly measure each voxel's correlation with each feature channel.
    # Using validation set data. 
    features_each_model_val, voxel_feature_correlations_val, ignore1, ignore2 =  fwrf_predict.get_voxel_feature_corrs(best_params, models, _fmaps_fn, val_voxel_single_trial_data, 
                                                                                                    val_stim_single_trial_data, sample_batch_size, aperture, device, debug=False, dtype=fpX)
 
    ### SAVE THE RESULTS TO DISK #########
    print('\nSaving result to %s'%fn2save)
    
    torch.save({
    'feature_table': _gaborizer.feature_table,
    'sf_tuning_masks': _gaborizer.sf_tuning_masks, 
    'ori_tuning_masks': _gaborizer.ori_tuning_masks,
    'cyc_per_stim': _gaborizer.cyc_per_stim,
    'orients_deg': _gaborizer.orients_deg,
    'orient_filters': _gaborizer.orient_filters,  
    'aperture': aperture,
    'models': models,
    'voxel_mask': voxel_mask,
    'brain_nii_shape': brain_nii_shape,
    'image_order': image_order,
    'voxel_index': voxel_index,
    'voxel_roi': voxel_roi,
    'voxel_ncsnr': voxel_ncsnr, 
    'best_params': best_params,
    'lambdas': lambdas, 
    'best_lambdas': best_lambdas,
    'best_losses': best_losses,
    'val_cc': val_cc,
    'val_r2': val_r2,   
    'features_each_model_val': features_each_model_val,
    'covar_each_model_training': covar_each_model_training,
    'voxel_feature_correlations_val': voxel_feature_correlations_val,
    'zscore_features': zscore_features,
    'nonlin_fn': nonlin_fn,
    'padding_mode': padding_mode,
    'debug': debug
    }, fn2save)
  



def fit_gabor_pca(subject=1, roi='V1', up_to_sess=1, n_ori=36, n_sf=12, sample_batch_size=50, voxel_batch_size=100, zscore_features=True, nonlin_fn=False, padding_mode = 'circular', 
                   debug=False, shuffle_images=False, random_images=False, random_voxel_data=False):
    
    """ 
    Apply PCA to gabor feature space (within specified pRF), then fit linear mapping from PC scores to voxel response.
    """
        
    device = initialize_fitting.init_cuda()
    nsd_root, stim_root, beta_root, mask_root = initialize_fitting.get_paths()

    model_name = 'gabor_PCA_%dori_%dsf'%(n_ori, n_sf)      
    output_dir, fn2save = initialize_fitting.get_save_path(root_dir, subject, model_name, shuffle_images, random_images, random_voxel_data, debug)
       
    # decide what voxels to use  
    voxel_mask, voxel_index, voxel_roi, voxel_ncsnr, brain_nii_shape = initialize_fitting.get_voxel_info(mask_root, beta_root, subject, roi)

    # get all data and corresponding images, in two splits. always fixed set that gets left out
    trn_stim_data, trn_voxel_data, val_stim_single_trial_data, val_voxel_single_trial_data, \
        n_voxels, n_trials_val, image_order = initialize_fitting.get_data_splits(nsd_root, beta_root, stim_root, subject, voxel_mask, up_to_sess, 
                                                                                 shuffle_images=shuffle_images, random_images=random_images, random_voxel_data=random_voxel_data)

    # Set up the filters
    _gaborizer, _fmaps_fn = initialize_fitting.get_feature_map_fn(n_ori, n_sf, padding_mode, device, nonlin_fn)

    # Params for the spatial aspect of the model (possible pRFs)
    aperture, models = initialize_fitting.get_prf_models()    
    
    # More params for fitting
    # note that these lambda values never get used in my pca fitting code (since pca already should reduce overfitting)
    holdout_size, lambdas = initialize_fitting.get_fitting_pars(trn_voxel_data, zscore_features, ridge=False)    

    #### DO THE ACTUAL MODEL FITTING HERE ####
    gc.collect()
    torch.cuda.empty_cache()
    best_losses,  pc,  best_params = fwrf_fit.learn_params_pca(
        trn_stim_data, trn_voxel_data, _fmaps_fn, models, min_pct_var=99,
        aperture=aperture, zscore=zscore_features, sample_batch_size=sample_batch_size, \
        voxel_batch_size=voxel_batch_size, holdout_size=holdout_size, shuffle=True, add_bias=True, debug=debug)
    # note there's also a shuffle param in the above fn call, that determines the nested heldout data for lambda and param selection. always using true.
    print('\nDone with training\n')

    # Validate model on held-out test set
    val_cc, val_r2 = fwrf_predict.validate_model(best_params, val_voxel_single_trial_data, val_stim_single_trial_data, _fmaps_fn, sample_batch_size, voxel_batch_size, aperture, dtype=fpX, pc=pc)
    
    # As a less model-sensitive way of assessing tuning, directly measure each voxel's correlation with each feature channel.
    # Using validation set data. 
    features_each_model_val, voxel_feature_correlations_val, features_pca_each_model_val, voxel_pca_feature_correlations_val =  fwrf_predict.get_voxel_feature_corrs(best_params, models, _fmaps_fn, val_voxel_single_trial_data, 
                                                                                                    val_stim_single_trial_data, sample_batch_size, aperture, device, debug=False, dtype=fpX, pc=pc)
 
    ### SAVE THE RESULTS TO DISK #########
    print('\nSaving result to %s'%fn2save)
    
    torch.save({
    'feature_table': _gaborizer.feature_table,
    'sf_tuning_masks': _gaborizer.sf_tuning_masks, 
    'ori_tuning_masks': _gaborizer.ori_tuning_masks,
    'cyc_per_stim': _gaborizer.cyc_per_stim,
    'orients_deg': _gaborizer.orients_deg,
    'orient_filters': _gaborizer.orient_filters,  
    'aperture': aperture,
    'models': models,
    'voxel_mask': voxel_mask,
    'brain_nii_shape': brain_nii_shape,
    'image_order': image_order,
    'voxel_index': voxel_index,
    'voxel_roi': voxel_roi,
    'voxel_ncsnr': voxel_ncsnr, 
    'best_params': best_params,
    'best_losses': best_losses,
    'val_cc': val_cc,
    'val_r2': val_r2,   
    'features_each_model_val': features_each_model_val,
    'features_pca_each_model_val': features_pca_each_model_val,    
    'voxel_feature_correlations_val': voxel_feature_correlations_val,
    'voxel_pca_feature_correlations_val': voxel_pca_feature_correlations_val,
    'pc':pc,
    'zscore_features': zscore_features,
    'nonlin_fn': nonlin_fn,
    'padding_mode': padding_mode,
    'debug': debug
    }, fn2save)

 
def fit_gabor_combinations(subject=1, roi='V1', up_to_sess=1, n_ori=36, n_sf=12, sample_batch_size=50, voxel_batch_size=100, zscore_features=True, nonlin_fn=False,  ridge=True, padding_mode = 'circular', 
                   debug=False, shuffle_images=False, random_images=False, random_voxel_data=False):
    
    """ 
    Use model that includes second order "combinations" consisting of multiplying features from gabor model.
    """
        
    device = initialize_fitting.init_cuda()
    nsd_root, stim_root, beta_root, mask_root = initialize_fitting.get_paths()

    if ridge==True:
        # ridge regression, testing several positive lambda values (default)
        model_name = 'gabor_combinations_ridge_%dori_%dsf'%(n_ori, n_sf)
    else:        
        # fixing lambda at zero, so it turns into ordinary least squares
        model_name = 'gabor_combinations_OLS_%dori_%dsf'%(n_ori, n_sf)
    output_dir, fn2save = initialize_fitting.get_save_path(root_dir, subject, model_name, shuffle_images, random_images, random_voxel_data, debug)
       
    # decide what voxels to use  
    voxel_mask, voxel_index, voxel_roi, voxel_ncsnr, brain_nii_shape = initialize_fitting.get_voxel_info(mask_root, beta_root, subject, roi)

    # get all data and corresponding images, in two splits. always fixed set that gets left out
    trn_stim_data, trn_voxel_data, val_stim_single_trial_data, val_voxel_single_trial_data, \
        n_voxels, n_trials_val, image_order = initialize_fitting.get_data_splits(nsd_root, beta_root, stim_root, subject, voxel_mask, up_to_sess, 
                                                                                 shuffle_images=shuffle_images, random_images=random_images, random_voxel_data=random_voxel_data)

    # Set up the filters
    _gaborizer, _fmaps_fn = initialize_fitting.get_feature_map_fn(n_ori, n_sf, padding_mode, device, nonlin_fn)

    # Params for the spatial aspect of the model (possible pRFs)
    aperture, models = initialize_fitting.get_prf_models()    
    
    # More params for fitting
    # note that these lambda values never get used in my pca fitting code (since pca already should reduce overfitting)
    holdout_size, lambdas = initialize_fitting.get_fitting_pars(trn_voxel_data, zscore_features, ridge=ridge)    

    #### DO THE ACTUAL MODEL FITTING HERE ####
    gc.collect()
    torch.cuda.empty_cache()
    best_losses, best_lambdas, best_params, covar_each_model_training, combs_zstats = fwrf_fit.learn_params_combinations_ridge_regression(
        trn_stim_data, trn_voxel_data, _fmaps_fn, models, lambdas, \
        aperture=aperture, zscore=zscore_features, sample_batch_size=sample_batch_size, \
        voxel_batch_size=voxel_batch_size, holdout_size=holdout_size, shuffle=True, add_bias=True, debug=debug)
    # note there's also a shuffle param in the above fn call, that determines the nested heldout data for lambda and param selection. always using true.
    print('\nDone with training\n')
    print('size of final weight params matrix is:')
    print(np.shape(best_params[1]))
    # Validate model on held-out test set
    val_cc, val_r2 = fwrf_predict.validate_model(best_params, val_voxel_single_trial_data, val_stim_single_trial_data, _fmaps_fn, sample_batch_size, voxel_batch_size, aperture, dtype=fpX, combs_zstats = combs_zstats)
    
    # As a less model-sensitive way of assessing tuning, directly measure each voxel's correlation with each feature channel.
    # Using validation set data. 
    features_each_model_val, voxel_feature_correlations_val, features_pca_each_model_val, voxel_pca_feature_correlations_val =  fwrf_predict.get_voxel_feature_corrs(best_params, models, _fmaps_fn, val_voxel_single_trial_data, 
                                                                                                    val_stim_single_trial_data, sample_batch_size, aperture, device, debug=False, dtype=fpX, combs_zstats = combs_zstats)
 
    ### SAVE THE RESULTS TO DISK #########
    print('\nSaving result to %s'%fn2save)
    
    torch.save({
    'feature_table': _gaborizer.feature_table,
    'sf_tuning_masks': _gaborizer.sf_tuning_masks, 
    'ori_tuning_masks': _gaborizer.ori_tuning_masks,
    'cyc_per_stim': _gaborizer.cyc_per_stim,
    'orients_deg': _gaborizer.orients_deg,
    'orient_filters': _gaborizer.orient_filters,  
    'aperture': aperture,
    'models': models,
    'voxel_mask': voxel_mask,
    'brain_nii_shape': brain_nii_shape,
    'image_order': image_order,
    'voxel_index': voxel_index,
    'voxel_roi': voxel_roi,
    'voxel_ncsnr': voxel_ncsnr, 
    'best_params': best_params,
    'best_losses': best_losses,
    'val_cc': val_cc,
    'val_r2': val_r2,   
    'lambdas': lambdas, 
    'best_lambdas': best_lambdas,
    'covar_each_model_training': covar_each_model_training,
    'features_each_model_val': features_each_model_val,   
    'voxel_feature_correlations_val': voxel_feature_correlations_val,   
    'combs_zstats':combs_zstats,
    'zscore_features': zscore_features,
    'nonlin_fn': nonlin_fn,
    'padding_mode': padding_mode,
    'debug': debug
    }, fn2save)
    
 
def fit_gabor_combinations_pca(subject=1, roi='V1', up_to_sess=1, n_ori=36, n_sf=12, sample_batch_size=50, voxel_batch_size=100, zscore_features=True, nonlin_fn=False, padding_mode = 'circular', 
                   debug=False, shuffle_images=False, random_images=False, random_voxel_data=False):
    
    """ 
    Use model that includes second order "combinations" consisting of multiplying features from gabor model.
    Apply PCA to gabor feature space (within specified pRF), then fit linear mapping from PC scores to voxel response.
    """
        
    device = initialize_fitting.init_cuda()
    nsd_root, stim_root, beta_root, mask_root = initialize_fitting.get_paths()

    model_name = 'gabor_combinations_PCA_%dori_%dsf'%(n_ori, n_sf)      
    output_dir, fn2save = initialize_fitting.get_save_path(root_dir, subject, model_name, shuffle_images, random_images, random_voxel_data, debug)
       
    # decide what voxels to use  
    voxel_mask, voxel_index, voxel_roi, voxel_ncsnr, brain_nii_shape = initialize_fitting.get_voxel_info(mask_root, beta_root, subject, roi)

    # get all data and corresponding images, in two splits. always fixed set that gets left out
    trn_stim_data, trn_voxel_data, val_stim_single_trial_data, val_voxel_single_trial_data, \
        n_voxels, n_trials_val, image_order = initialize_fitting.get_data_splits(nsd_root, beta_root, stim_root, subject, voxel_mask, up_to_sess, 
                                                                                 shuffle_images=shuffle_images, random_images=random_images, random_voxel_data=random_voxel_data)

    # Set up the filters
    _gaborizer, _fmaps_fn = initialize_fitting.get_feature_map_fn(n_ori, n_sf, padding_mode, device, nonlin_fn)

    # Params for the spatial aspect of the model (possible pRFs)
    aperture, models = initialize_fitting.get_prf_models()    
    
    # More params for fitting
    # note that these lambda values never get used in my pca fitting code (since pca already should reduce overfitting)
    holdout_size, lambdas = initialize_fitting.get_fitting_pars(trn_voxel_data, zscore_features, ridge=False)    

    #### DO THE ACTUAL MODEL FITTING HERE ####
    gc.collect()
    torch.cuda.empty_cache()
    best_losses,  pc,  best_params, combs_zstats = fwrf_fit.learn_params_combinations_pca(
        trn_stim_data, trn_voxel_data, _fmaps_fn, models, min_pct_var=99,
        aperture=aperture, zscore=zscore_features, sample_batch_size=sample_batch_size, \
        voxel_batch_size=voxel_batch_size, holdout_size=holdout_size, shuffle=True, add_bias=True, debug=debug)
    # note there's also a shuffle param in the above fn call, that determines the nested heldout data for lambda and param selection. always using true.
    print('\nDone with training\n')
    print('size of final weight params matrix is:')
    print(np.shape(best_params[1]))
    # Validate model on held-out test set
    val_cc, val_r2 = fwrf_predict.validate_model(best_params, val_voxel_single_trial_data, val_stim_single_trial_data, _fmaps_fn, sample_batch_size, voxel_batch_size, aperture, dtype=fpX, combs_zstats = combs_zstats, pc=pc)
    
    # As a less model-sensitive way of assessing tuning, directly measure each voxel's correlation with each feature channel.
    # Using validation set data. 
    features_each_model_val, voxel_feature_correlations_val, features_pca_each_model_val, voxel_pca_feature_correlations_val =  fwrf_predict.get_voxel_feature_corrs(best_params, models, _fmaps_fn, val_voxel_single_trial_data, 
                                                                                                    val_stim_single_trial_data, sample_batch_size, aperture, device, debug=False, dtype=fpX, combs_zstats = combs_zstats,  pc=pc)
 
    ### SAVE THE RESULTS TO DISK #########
    print('\nSaving result to %s'%fn2save)
    
    torch.save({
    'feature_table': _gaborizer.feature_table,
    'sf_tuning_masks': _gaborizer.sf_tuning_masks, 
    'ori_tuning_masks': _gaborizer.ori_tuning_masks,
    'cyc_per_stim': _gaborizer.cyc_per_stim,
    'orients_deg': _gaborizer.orients_deg,
    'orient_filters': _gaborizer.orient_filters,  
    'aperture': aperture,
    'models': models,
    'voxel_mask': voxel_mask,
    'brain_nii_shape': brain_nii_shape,
    'image_order': image_order,
    'voxel_index': voxel_index,
    'voxel_roi': voxel_roi,
    'voxel_ncsnr': voxel_ncsnr, 
    'best_params': best_params,
    'best_losses': best_losses,
    'val_cc': val_cc,
    'val_r2': val_r2,   
    'features_each_model_val': features_each_model_val,
    'features_pca_each_model_val': features_pca_each_model_val,    
    'voxel_feature_correlations_val': voxel_feature_correlations_val,
    'voxel_pca_feature_correlations_val': voxel_pca_feature_correlations_val,
    'pc':pc,
    'combs_zstats':combs_zstats,
    'zscore_features': zscore_features,
    'nonlin_fn': nonlin_fn,
    'padding_mode': padding_mode,
    'debug': debug
    }, fn2save)

           

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int,default=1,
                    help="number of the subject, 1-8")
    parser.add_argument("--roi", type=str, default='None',
                    help="ROI name, in ['V1', 'V2', 'V3', 'hV4', 'V3ab', 'LO', 'IPS', 'VO', 'PHC', 'MT', 'MST'] or 'None' for all vis areas")
    
    parser.add_argument("--fitting_type", type=str,default='gabor_ridge',
                    help="what kind of fitting are we doing? opts are 'gabor','gabor_pca','gabor_combs','gabor_combs_pca'")
    parser.add_argument("--ridge", type=int,default=1,
                    help="want to do ridge regression (lambda>0)? 1 for yes, 0 for no")
    parser.add_argument("--shuffle_images", type=int,default=0,
                    help="want to shuffle the images randomly (control analysis)? 1 for yes, 0 for no")
    parser.add_argument("--random_images", type=int,default=0,
                    help="want to use random gaussian values for images (control analysis)? 1 for yes, 0 for no")
    parser.add_argument("--random_voxel_data", type=int,default=0,
                    help="want to use random gaussian values for voxel data (control analysis)? 1 for yes, 0 for no")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    
    parser.add_argument("--up_to_sess", type=int,default=1,
                    help="analyze sessions 1-#")
    parser.add_argument("--n_ori", type=int,default=36,
                    help="number of orientation channels to use")
    parser.add_argument("--n_sf", type=int,default=12,
                    help="number of spatial frequency channels to use")
    parser.add_argument("--sample_batch_size", type=int,default=50,
                    help="number of trials to analyze at once when making features (smaller will help with out-of-memory errors)")
    parser.add_argument("--voxel_batch_size", type=int,default=100,
                    help="number of voxels to analyze at once when fitting weights (smaller will help with out-of-memory errors)")
    parser.add_argument("--zscore_features", type=int,default=1,
                    help="want to z-score each feature right before fitting encoding model? 1 for yes, 0 for no")
    parser.add_argument("--nonlin_fn", type=int,default=0,
                    help="want to apply a nonlinearity to each feature before fitting encoding model? 1 for yes, 0 for no")
    parser.add_argument("--padding_mode", type=str,default='circular',
                    help="how to pad when doing convolution during gabor feature generation? opts are 'circular','reflect','constant','replicate'; default is circular.")
    
    
    args = parser.parse_args()
    if args.roi=='None':
        roi = None
    else:
        roi = args.roi
        
    # print values of a few key things to the command line...
    if args.debug==1:
        print('USING DEBUG MODE...')
    if args.ridge==1 and 'pca' not in args.fitting_type:
        print('will perform ridge regression for a range of positive lambdas.')
    else:
        print('will fix ridge parameter at 0.0')    
        
    if args.zscore_features==1:
        print('will perform z-scoring of features')
    else:
        print('skipping z-scoring of features')
    if args.nonlin_fn==1:
        print('will use log(1+sqrt(x)) as nonlinearity fn')
    else:
        print('skipping nonlinearity fn')
    print('padding mode is %s'%args.padding_mode)   
    
    if args.shuffle_images==1:
        print('\nWILL RANDOMLY SHUFFLE IMAGES\n')
    if args.random_images==1:
        print('\nWILL USE RANDOM NOISE IMAGES\n')
    if args.random_voxel_data==1:
        print('\nWILL USE RANDOM DATA\n')

    # now actually call the function to execute fitting...
    if args.fitting_type=='gabor':       
        fit_gabor_fwrf(args.subject, roi, args.up_to_sess, args.n_ori, args.n_sf, args.sample_batch_size, args.voxel_batch_size,
                       args.zscore_features==1, args.nonlin_fn==1, args.ridge==1, args.padding_mode, args.debug==1, args.shuffle_images==1, args.random_images==1, args.random_voxel_data==1)
        
    elif args.fitting_type=='gabor_pca':       
        fit_gabor_pca(args.subject, roi, args.up_to_sess, args.n_ori, args.n_sf, args.sample_batch_size, args.voxel_batch_size,
                       args.zscore_features==1, args.nonlin_fn==1, args.padding_mode, args.debug==1, args.shuffle_images==1, args.random_images==1, args.random_voxel_data==1)
        
    elif args.fitting_type=='gabor_combs':       
        fit_gabor_combinations(args.subject, roi, args.up_to_sess, args.n_ori, args.n_sf, args.sample_batch_size, args.voxel_batch_size,
                       args.zscore_features==1, args.nonlin_fn==1, args.ridge==1, args.padding_mode, args.debug==1, args.shuffle_images==1, args.random_images==1, args.random_voxel_data==1)
        
    elif args.fitting_type=='gabor_combs_pca':       
        fit_gabor_combinations_pca(args.subject, roi, args.up_to_sess, args.n_ori, args.n_sf, args.sample_batch_size, args.voxel_batch_size,
                       args.zscore_features==1, args.nonlin_fn==1, args.padding_mode, args.debug==1, args.shuffle_images==1, args.random_images==1, args.random_voxel_data==1)
    else:
        print('fitting for %s not implemented yet!!'%args.fitting_type)
        exit()
