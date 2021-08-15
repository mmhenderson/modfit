"""
Run the model fitting for FWRF model. 
There are a few different versions of fitting in this script, the input arguments tell which kind of fitting to do.
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
import skimage.transform

# import custom modules
root_dir   = os.path.dirname(os.getcwd())
sys.path.append(os.path.join(root_dir))
from model_src import fwrf_fit as fwrf_fit
from model_src import fwrf_predict as fwrf_predict
from model_src import texture_statistics_gabor, texture_statistics_pyramid

import initialize_fitting

fpX = np.float32

#################################################################################################

def fit_pyramid_texture_fwrf(subject=1, roi='V1', up_to_sess=1, n_ori=36, n_sf=12, sample_batch_size=50, voxel_batch_size=100, \
                    zscore_features=True, nonlin_fn=False, ridge=True, \
                   debug=False, shuffle_images=False, random_images=False, random_voxel_data=False, do_fitting=True, do_val=True, do_partial=True, date_str=None, shuff_rnd_seed=0):
    
    """ 
    Fit linear mapping from various textural features (within specified pRF) to voxel response.
    """
    
    
    device = initialize_fitting.init_cuda()
    nsd_root, stim_root, beta_root, mask_root = initialize_fitting.get_paths()
    model_name, feature_types_exclude = initialize_fitting.get_pyramid_model_name(ridge, n_ori, n_sf)
    
    if do_fitting==False and date_str is None:
        raise ValueError('if you want to start midway through the process (--do_fitting=False), then specify the date when training result was saved (--date_str).')

    if do_fitting==True and date_str is not None:
        raise ValueError('if you want to do fitting from scratch (--do_fitting=True), specify --date_str=None (rather than entering a date)')

    output_dir, fn2save = initialize_fitting.get_save_path(root_dir, subject, model_name, shuffle_images, random_images, random_voxel_data, debug, date_str)
    
    def save_all():
        print('\nSaving to %s\n'%fn2save)
        torch.save({
        'aperture': aperture,
        'aperture_rf_range': aperture_rf_range,
        'models': models,
        'feature_info':feature_info,
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
        'val_cc_partial': val_cc_partial,
        'val_r2_partial': val_r2_partial,   
        'features_each_model_val': features_each_model_val,
        'voxel_feature_correlations_val': voxel_feature_correlations_val,
        'zscore_features': zscore_features,
        'nonlin_fn': nonlin_fn,
        'n_prf_sd_out': n_prf_sd_out,
        'debug': debug,
        'up_to_sess': up_to_sess,
        'shuff_rnd_seed': shuff_rnd_seed
        }, fn2save, pickle_protocol=4)
            
            
    # decide what voxels to use  
    voxel_mask, voxel_index, voxel_roi, voxel_ncsnr, brain_nii_shape = initialize_fitting.get_voxel_info(mask_root, beta_root, subject, roi)

    # get all data and corresponding images, in two splits. always fixed set that gets left out
    trn_stim_data, trn_voxel_data, val_stim_single_trial_data, val_voxel_single_trial_data, \
        n_voxels, n_trials_val, image_order = initialize_fitting.get_data_splits(nsd_root, beta_root, stim_root, subject, voxel_mask, up_to_sess, 
                                                                                 shuffle_images=shuffle_images, random_images=random_images, random_voxel_data=random_voxel_data)

    # Need a multiple of 8
    process_at_size=240
    trn_stim_data = skimage.transform.resize(trn_stim_data, output_shape=(trn_stim_data.shape[0],1,process_at_size, process_at_size))
    val_stim_single_trial_data = skimage.transform.resize(val_stim_single_trial_data, output_shape=(val_stim_single_trial_data.shape[0],1,process_at_size, process_at_size))
    
    # Set up the pyramid
    _fmaps_fn = texture_statistics_pyramid.steerable_pyramid_extractor(pyr_height=n_sf, n_ori = n_ori)
    # Params for the spatial aspect of the model (possible pRFs)
#     aperture_rf_range=0.8 # using smaller range here because not sure what to do with RFs at edges...
    aperture_rf_range = 1.1
    aperture, models = initialize_fitting.get_prf_models(aperture_rf_range=aperture_rf_range)    
    
    # Initialize the "texture" model which builds on first level feature maps
    n_prf_sd_out=2
    _texture_fn = texture_statistics_pyramid.texture_feature_extractor(_fmaps_fn,sample_batch_size=sample_batch_size, feature_types_exclude=feature_types_exclude, n_prf_sd_out=n_prf_sd_out, aperture=aperture, device=device)
    
    # More params for fitting
    holdout_size, lambdas = initialize_fitting.get_fitting_pars(trn_voxel_data, zscore_features, ridge=ridge)

    
    #### DO THE ACTUAL MODEL FITTING HERE ####
    
    if do_fitting:
        gc.collect()
        torch.cuda.empty_cache()
        print('\nStarting training...\n')
        if shuff_rnd_seed==0:
            shuff_rnd_seed = int(time.strftime('%M%H%d', time.localtime()))
        best_losses, best_lambdas, best_params, feature_info = fwrf_fit.fit_texture_model_ridge(trn_stim_data, trn_voxel_data, _texture_fn, models, lambdas, \
            zscore=zscore_features, voxel_batch_size=voxel_batch_size, holdout_size=holdout_size, shuffle=True, add_bias=True, debug=debug, shuff_rnd_seed=shuff_rnd_seed,device=device)
        # note there's also a shuffle param in the above fn call, that determines the nested heldout data for lambda and param selection. always using true.
        print('\nDone with training\n')
        
        val_cc=None
        val_r2=None
        val_cc_partial=None
        val_r2_partial=None
        features_each_model_val=None;
        voxel_feature_correlations_val=None;
        
        save_all()
    
    else:
        print('\nLoading the results of training from %s\n'%fn2save)
        out = torch.load(fn2save)
        best_losses = out['best_losses']
        best_lambdas = out['best_lambdas']
        best_params = out['best_params']
        feature_info = out['feature_info']
        val_cc = out['val_cc']
        val_r2 = out['val_r2']
        val_cc_partial = out['val_cc_partial']
        val_r2_partial = out['val_r2_partial']
        features_each_model_val=out['features_each_model_val'];
        voxel_feature_correlations_val=out['voxel_feature_correlations_val'];
        if 'shuff_rnd_seed' in list(out.keys()):
            shuff_rnd_seed=out['shuff_rnd_seed']
        else:
            shuff_rnd_seed=0
            
        # some checks to make sure we're resuming same process...        
        assert(np.all(models==out['models']))
        assert(out['n_prf_sd_out']== n_prf_sd_out)
        assert(out['autocorr_output_pix']== autocorr_output_pix)
        assert(_texture_fn.n_ori==n_ori)
        assert(_texture_fn.n_sf==n_sf)
        assert(out['zscore_features']==zscore_features)
        assert(out['nonlin_fn']==nonlin_fn)
        assert(out['up_to_sess']==up_to_sess)
        assert(out['padding_mode']==padding_mode)
        assert(out['best_params'][1].shape[1]==feature_info[0].shape[0])
        assert(out['best_params'][1].shape[0]==voxel_index[0].shape[0])
        assert(not np.any([ff in feature_info[1] for ff in feature_types_exclude]))
    
    if val_cc is not None:
        do_val=False
        print('\nValidation is already done! not going to run it again.')       
    if val_cc_partial is not None:
        do_partial=False
        print('\nVariance partition is already done! not going to run it again.')

    ## Validate model on held-out test set #####
    sys.stdout.flush()
    if do_val: 
        gc.collect()
        torch.cuda.empty_cache()
        val_cc, val_r2 = fwrf_predict.validate_texture_model(best_params, models, val_voxel_single_trial_data, val_stim_single_trial_data, _texture_fn, sample_batch_size=sample_batch_size, voxel_batch_size=voxel_batch_size, debug=debug, dtype=fpX)
                                                       
        save_all()
  
    ## Validate model, including subsets of features  #####

    sys.stdout.flush()
    if do_partial:
        if len(_texture_fn.feature_types_include)>1:
            gc.collect()
            torch.cuda.empty_cache()
            val_cc_partial, val_r2_partial = fwrf_predict.validate_texture_model_partial(best_params, models, val_voxel_single_trial_data, val_stim_single_trial_data, _texture_fn, sample_batch_size=sample_batch_size, voxel_batch_size=voxel_batch_size, debug=debug, dtype=fpX)
        else:
            print('Model only has one feature type, so will not do variance partition.')
            
        save_all()
      
    

def fit_gabor_texture_fwrf(subject=1, roi='V1', up_to_sess=1, n_ori=36, n_sf=12, sample_batch_size=50, voxel_batch_size=100, \
                     include_pixel=True, include_simple=True, include_complex=True, include_autocorrs=True, include_crosscorrs=True, \
                     zscore_features=True, nonlin_fn=False, ridge=True, padding_mode = 'circular', 
                   debug=False, shuffle_images=False, random_images=False, random_voxel_data=False, do_fitting=True, do_val=True, do_partial=True, date_str=None, shuff_rnd_seed=0):
    
    """ 
    Fit linear mapping from various textural features (within specified pRF) to voxel response.
    """
    
    
    device = initialize_fitting.init_cuda()
    nsd_root, stim_root, beta_root, mask_root = initialize_fitting.get_paths()
    model_name, feature_types_exclude = initialize_fitting.get_model_name(ridge, n_ori, n_sf, include_pixel, include_simple, include_complex, include_autocorrs, include_crosscorrs)
    
    if do_fitting==False and date_str is None:
        raise ValueError('if you want to start midway through the process (--do_fitting=False), then specify the date when training result was saved (--date_str).')

    if do_fitting==True and date_str is not None:
        raise ValueError('if you want to do fitting from scratch (--do_fitting=True), specify --date_str=None (rather than entering a date)')

    output_dir, fn2save = initialize_fitting.get_save_path(root_dir, subject, model_name, shuffle_images, random_images, random_voxel_data, debug, date_str)
    
    def save_all():
        print('\nSaving to %s\n'%fn2save)
        torch.save({
        'feature_table_simple': _gaborizer_simple.feature_table,
        'sf_tuning_masks_simple': _gaborizer_simple.sf_tuning_masks, 
        'ori_tuning_masks_simple': _gaborizer_simple.ori_tuning_masks,
        'cyc_per_stim_simple': _gaborizer_simple.cyc_per_stim,
        'orients_deg_simple': _gaborizer_simple.orients_deg,
        'orient_filters_simple': _gaborizer_simple.orient_filters,  
        'feature_table_complex': _gaborizer_complex.feature_table,
        'sf_tuning_masks_complex': _gaborizer_complex.sf_tuning_masks, 
        'ori_tuning_masks_complex': _gaborizer_complex.ori_tuning_masks,
        'cyc_per_stim_complex': _gaborizer_complex.cyc_per_stim,
        'orients_deg_complex': _gaborizer_complex.orients_deg,
        'orient_filters_complex': _gaborizer_complex.orient_filters,  
        'aperture': aperture,
        'aperture_rf_range': aperture_rf_range,
        'models': models,
        'include_autocorrs': include_autocorrs,
        'feature_info':feature_info,
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
        'val_cc_partial': val_cc_partial,
        'val_r2_partial': val_r2_partial,   
        'features_each_model_val': features_each_model_val,
        'voxel_feature_correlations_val': voxel_feature_correlations_val,
        'zscore_features': zscore_features,
        'nonlin_fn': nonlin_fn,
        'padding_mode': padding_mode,
        'n_prf_sd_out': n_prf_sd_out,
        'autocorr_output_pix': autocorr_output_pix,
        'debug': debug,
        'up_to_sess': up_to_sess,
        'shuff_rnd_seed': shuff_rnd_seed
        }, fn2save, pickle_protocol=4)
            
            
    # decide what voxels to use  
    voxel_mask, voxel_index, voxel_roi, voxel_ncsnr, brain_nii_shape = initialize_fitting.get_voxel_info(mask_root, beta_root, subject, roi)

    # get all data and corresponding images, in two splits. always fixed set that gets left out
    trn_stim_data, trn_voxel_data, val_stim_single_trial_data, val_voxel_single_trial_data, \
        n_voxels, n_trials_val, image_order = initialize_fitting.get_data_splits(nsd_root, beta_root, stim_root, subject, voxel_mask, up_to_sess, 
                                                                                 shuffle_images=shuffle_images, random_images=random_images, random_voxel_data=random_voxel_data)

    # Set up the filters
    _gaborizer_complex, _gaborizer_simple, _fmaps_fn_complex, _fmaps_fn_simple = initialize_fitting.get_feature_map_simple_complex_fn(n_ori, n_sf, padding_mode=padding_mode, device=device, nonlin_fn=nonlin_fn)

    # Params for the spatial aspect of the model (possible pRFs)
#     aperture_rf_range=0.8 # using smaller range here because not sure what to do with RFs at edges...
    aperture_rf_range = 1.1
    aperture, models = initialize_fitting.get_prf_models(aperture_rf_range=aperture_rf_range)    
    
     # Initialize the "texture" model which builds on first level feature maps
    autocorr_output_pix=5
    n_prf_sd_out=2
    _texture_fn = texture_statistics_gabor.texture_feature_extractor(_fmaps_fn_complex, _fmaps_fn_simple, sample_batch_size=sample_batch_size, feature_types_exclude=feature_types_exclude, autocorr_output_pix=autocorr_output_pix, n_prf_sd_out=n_prf_sd_out, aperture=aperture, device=device)

    # More params for fitting
    holdout_size, lambdas = initialize_fitting.get_fitting_pars(trn_voxel_data, zscore_features, ridge=ridge)

    
    #### DO THE ACTUAL MODEL FITTING HERE ####
    
    if do_fitting:
        gc.collect()
        torch.cuda.empty_cache()
        print('\nStarting training...\n')
        if shuff_rnd_seed==0:
            shuff_rnd_seed = int(time.strftime('%M%H%d', time.localtime()))
        best_losses, best_lambdas, best_params, feature_info = fwrf_fit.fit_texture_model_ridge(trn_stim_data, trn_voxel_data, _texture_fn, models, lambdas, \
            zscore=zscore_features, voxel_batch_size=voxel_batch_size, holdout_size=holdout_size, shuffle=True, add_bias=True, debug=debug, shuff_rnd_seed=shuff_rnd_seed, device=device)
        # note there's also a shuffle param in the above fn call, that determines the nested heldout data for lambda and param selection. always using true.
        print('\nDone with training\n')
        
        val_cc=None
        val_r2=None
        val_cc_partial=None
        val_r2_partial=None
        features_each_model_val=None;
        voxel_feature_correlations_val=None;
        
        save_all()
    
    else:
        print('\nLoading the results of training from %s\n'%fn2save)
        out = torch.load(fn2save)
        best_losses = out['best_losses']
        best_lambdas = out['best_lambdas']
        best_params = out['best_params']
        feature_info = out['feature_info']
        val_cc = out['val_cc']
        val_r2 = out['val_r2']
        val_cc_partial = out['val_cc_partial']
        val_r2_partial = out['val_r2_partial']
        features_each_model_val=out['features_each_model_val'];
        voxel_feature_correlations_val=out['voxel_feature_correlations_val'];
        if 'shuff_rnd_seed' in list(out.keys()):
            shuff_rnd_seed=out['shuff_rnd_seed']
        else:
            shuff_rnd_seed=0
            
        # some checks to make sure we're resuming same process...        
        assert(np.all(models==out['models']))
        assert(out['n_prf_sd_out']== n_prf_sd_out)
        assert(out['autocorr_output_pix']== autocorr_output_pix)
        assert(_texture_fn.n_ori==n_ori)
        assert(_texture_fn.n_sf==n_sf)
        assert(out['zscore_features']==zscore_features)
        assert(out['nonlin_fn']==nonlin_fn)
        assert(out['up_to_sess']==up_to_sess)
        assert(out['padding_mode']==padding_mode)
        assert(out['best_params'][1].shape[1]==feature_info[0].shape[0])
        assert(out['best_params'][1].shape[0]==voxel_index[0].shape[0])
        assert(not np.any([ff in feature_info[1] for ff in feature_types_exclude]))
    
    if val_cc is not None:
        do_val=False
        print('\nValidation is already done! not going to run it again.')       
    if val_cc_partial is not None:
        do_partial=False
        print('\nVariance partition is already done! not going to run it again.')

    ## Validate model on held-out test set #####
    sys.stdout.flush()
    if do_val: 
        gc.collect()
        torch.cuda.empty_cache()
        val_cc, val_r2 = fwrf_predict.validate_texture_model(best_params, models, val_voxel_single_trial_data, val_stim_single_trial_data, _texture_fn, sample_batch_size=sample_batch_size, voxel_batch_size=voxel_batch_size, debug=debug, dtype=fpX)
                                                       
        save_all()
  
    ## Validate model, including subsets of features  #####

    sys.stdout.flush()
    if do_partial:
        if len(_texture_fn.feature_types_include)>1:
            gc.collect()
            torch.cuda.empty_cache()
            val_cc_partial, val_r2_partial = fwrf_predict.validate_texture_model_partial(best_params, models, val_voxel_single_trial_data, val_stim_single_trial_data, _texture_fn, sample_batch_size=sample_batch_size, voxel_batch_size=voxel_batch_size, debug=debug, dtype=fpX)
        else:
            print('Model only has one feature type, so will not do variance partition.')
            
        save_all()
            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int,default=1,
                    help="number of the subject, 1-8")
    parser.add_argument("--roi", type=str, default='None',
                    help="ROI name, in ['V1', 'V2', 'V3', 'hV4', 'V3ab', 'LO', 'IPS', 'VO', 'PHC', 'MT', 'MST'] or 'None' for all vis areas")

    parser.add_argument("--fitting_type", type=str,default='texture',
                    help="what kind of fitting are we doing? opts are 'texture' for now, use '--include_XX' flags for more specific versions")
    parser.add_argument("--ridge", type=int,default=1,
                    help="want to do ridge regression (lambda>0)? 1 for yes, 0 for no")
    parser.add_argument("--include_pixel", type=int,default=1,
                    help="want to include pixel-level stats (only used for texture model)? 1 for yes, 0 for no")
    parser.add_argument("--include_simple", type=int,default=1,
                    help="want to include simple cell-like resp (only used for texture model)? 1 for yes, 0 for no")
    parser.add_argument("--include_complex", type=int,default=1,
                    help="want to include complex cell-like resp (only used for texture model)? 1 for yes, 0 for no")
    parser.add_argument("--include_autocorrs", type=int,default=1,
                    help="want to include autocorrelations (only used for texture model)? 1 for yes, 0 for no")
    parser.add_argument("--include_crosscorrs", type=int,default=1,
                    help="want to include crosscorrelations (only used for texture model)? 1 for yes, 0 for no")
    
    parser.add_argument("--shuffle_images", type=int,default=0,
                    help="want to shuffle the images randomly (control analysis)? 1 for yes, 0 for no")
    parser.add_argument("--random_images", type=int,default=0,
                    help="want to use random gaussian values for images (control analysis)? 1 for yes, 0 for no")
    parser.add_argument("--random_voxel_data", type=int,default=0,
                    help="want to use random gaussian values for voxel data (control analysis)? 1 for yes, 0 for no")
    
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    
    
    parser.add_argument("--do_fitting", type=int,default=1,
                    help="want to do model training? 1 for yes, 0 for no")
    parser.add_argument("--do_val", type=int,default=1,
                    help="want to do model validation? 1 for yes, 0 for no")
    parser.add_argument("--do_partial", type=int,default=1,
                    help="want to do variance partition? 1 for yes, 0 for no")
    parser.add_argument("--date_str", type=str,default='None',
                    help="what date was the model fitting done (only if you're starting from validation step.)")
    
    
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
    parser.add_argument("--shuff_rnd_seed", type=int,default=0,
                    help="random seed to use for shuffling, when holding out part of training set for lambda selection.")
    
    
    args = parser.parse_args()
    if args.roi=='None':
        roi = None
    else:
        roi = args.roi
    if args.date_str=='None':
        date_str=None
    else:
        date_str = args.date_str
        
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

    if args.fitting_type=='texture':       
        fit_gabor_texture_fwrf(args.subject, roi, args.up_to_sess, args.n_ori, args.n_sf, args.sample_batch_size, args.voxel_batch_size, args.include_pixel==1, args.include_simple==1, args.include_complex==1, args.include_autocorrs==1, args.include_crosscorrs==1,
                         args.zscore_features==1, args.nonlin_fn==1, args.ridge==1, args.padding_mode, args.debug==1, args.shuffle_images==1, args.random_images==1, args.random_voxel_data==1,
                         args.do_fitting==1, args.do_val==1, args.do_partial==1, date_str, args.shuff_rnd_seed)   
    elif args.fitting_type=='pyramid_texture':
        fit_pyramid_texture_fwrf(args.subject, roi, args.up_to_sess, args.n_ori, args.n_sf, args.sample_batch_size, args.voxel_batch_size, 
                         args.zscore_features==1, args.nonlin_fn==1, args.ridge==1, args.debug==1, args.shuffle_images==1, args.random_images==1, args.random_voxel_data==1,
                         args.do_fitting==1, args.do_val==1, args.do_partial==1, date_str, args.shuff_rnd_seed)   
  
    else:
        print('fitting for %s not implemented currently!!'%args.fitting_type)
        exit()
