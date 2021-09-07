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
from model_src import texture_statistics_gabor, texture_statistics_pyramid, bdcn_features
from utils import nsd_utils, roi_utils

bdcn_path = '/user_data/mmhender/toolboxes/BDCN/'

import initialize_fitting, merge_features

fpX = np.float32
device = initialize_fitting.init_cuda()

#################################################################################################
        
    
def fit_fwrf(fitting_type, subject=1, volume_space = True, up_to_sess = 1, \
             n_ori = 4, n_sf = 4, nonlin_fn = False,  padding_mode = 'circular', \
             include_simple = True, include_complex = True, \
             include_autocorrs = True, include_crosscorrs = True, group_all_hl_feats = False, \
             sample_batch_size = 50, voxel_batch_size = 100, \
             zscore_features = True, ridge = True, \
             shuffle_images = False, random_images = False, random_voxel_data = False, \
             do_fitting = True, do_val = True, do_varpart = True, 
             shuff_rnd_seed = 0, debug = False, \
             do_pca = True, min_pct_var = 99, max_pc_to_retain = 400, map_ind = -1, \
             n_prf_sd_out = 2, mult_patch_by_prf = True, \
             downsample_factor = 1.0, do_nms = False):
    
    
    def save_fn_bdcn(fn2save):
        print('\nSaving to %s\n'%fn2save)
        torch.save({
        'aperture': aperture,
        'aperture_rf_range': aperture_rf_range,
        'models': models,
        'voxel_mask': voxel_mask,
        'brain_nii_shape': brain_nii_shape,
        'image_order': image_order,
        'voxel_index': voxel_index,
        'voxel_roi': voxel_roi,
        'voxel_ncsnr': voxel_ncsnr, 
        'best_params': best_params,
        'pc': pc2save,
        'lambdas': lambdas, 
        'best_lambdas': best_lambdas,
        'best_losses': best_losses,
        'val_cc': val_cc,
        'val_r2': val_r2,    
        'partial_masks': partial_masks, 
        'partial_version_names': partial_version_names,
        'zscore_features': zscore_features,        
        'n_prf_sd_out': n_prf_sd_out,
        'mult_patch_by_prf': mult_patch_by_prf,
        'do_nms': do_nms, 
        'downsample_factor': downsample_factor,
        'ridge': ridge,
        'do_pca': do_pca,
        'debug': debug,
        'up_to_sess': up_to_sess,
        'shuff_rnd_seed': shuff_rnd_seed
        }, fn2save, pickle_protocol=4)



    def save_fn_pyramid(fn2save):
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
        'partial_masks': partial_masks, 
        'partial_version_names': partial_version_names,
        'zscore_features': zscore_features,
        'n_prf_sd_out': n_prf_sd_out,
        'debug': debug,
        'up_to_sess': up_to_sess,
        'shuff_rnd_seed': shuff_rnd_seed
        }, fn2save, pickle_protocol=4)


    def save_fn_pyramid_bdcn(fn2save):
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
        'pc': pc2save,
        'lambdas': lambdas, 
        'best_lambdas': best_lambdas,
        'best_losses': best_losses,
        'val_cc': val_cc,
        'val_r2': val_r2,    
        'partial_masks': partial_masks, 
        'partial_version_names': partial_version_names,
        'zscore_features': zscore_features,        
        'n_prf_sd_out': n_prf_sd_out,
        'mult_patch_by_prf': mult_patch_by_prf,
        'do_nms': do_nms, 
        'downsample_factor': downsample_factor,
        'ridge': ridge,
        'do_pca': do_pca,
        'debug': debug,
        'up_to_sess': up_to_sess,
        'shuff_rnd_seed': shuff_rnd_seed
        }, fn2save, pickle_protocol=4)


    def save_fn_gabor(fn2save):
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
        'partial_masks': partial_masks, 
        'partial_version_names': partial_version_names,
        'feature_info': feature_info,
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
  
    if do_fitting==False and date_str is None:
        raise ValueError('if you want to start midway through the process (--do_fitting=False), then specify the date when training result was saved (--date_str).')

    if do_fitting==True and date_str is not None:
        raise ValueError('if you want to do fitting from scratch (--do_fitting=True), specify --date_str=None (rather than entering a date)')

    if do_fitting==False and do_pca==True:
        raise ValueError('Cannot start midway through the process (--do_fitting=False) when doing pca, because the pca weight matrix is not saved in between trn/val.')
        
    if 'pyramid' in fitting_type:
        model_name, feature_types_exclude = initialize_fitting.get_pyramid_model_name(ridge, n_ori, n_sf)
        
        if 'plus_bdcn' in fitting_type:
            model_name2 = initialize_fitting.get_bdcn_model_name(do_pca, map_ind)
            model_name = model_name + '_plus_' + model_name2
            save_all = save_fn_pyramid_bdcn
        else:
            save_all = save_fn_pyramid
            
    elif 'bdcn' in fitting_type:
        model_name = initialize_fitting.get_bdcn_model_name(do_pca, map_ind)
        save_all = save_fn_bdcn
        
    elif 'gabor' in fitting_type:
        model_name, feature_types_exclude = initialize_fitting.get_model_name(ridge, n_ori, n_sf, include_pixel, include_simple, include_complex, include_autocorrs, include_crosscorrs)
        save_all = save_fn_gabor
    
        
    output_dir, fn2save = initialize_fitting.get_save_path(root_dir, subject, volume_space, model_name, shuffle_images, random_images, random_voxel_data, debug, date_str)
    
    # decide what voxels to use  
    voxel_mask, voxel_index, voxel_roi, voxel_ncsnr, brain_nii_shape = roi_utils.get_voxel_roi_info(subject, volume_space)

    sessions = np.arange(0,up_to_sess)
    zscore_betas_within_sess = True
    # get all data and corresponding images, in two splits. always fixed set that gets left out
    trn_stim_data, trn_voxel_data, val_stim_data, val_voxel_data, image_order = nsd_utils.get_data_splits(subject, sessions=sessions, \
                                                                         voxel_mask=voxel_mask, volume_space=volume_space, \
                                                                          zscore_betas_within_sess=zscore_betas_within_sess, \
                                                                          shuffle_images=shuffle_images, random_images=random_images, \
                                                                                             random_voxel_data=random_voxel_data)

    if 'pyramid' in fitting_type:
        # Need a multiple of 8
        process_at_size=240
        trn_stim_data = skimage.transform.resize(trn_stim_data, output_shape=(trn_stim_data.shape[0],1,process_at_size, process_at_size))
        val_stim_data = skimage.transform.resize(val_stim_data, output_shape=(val_stim_data.shape[0],1,process_at_size, process_at_size))

    # More params for fitting
    holdout_size, lambdas = initialize_fitting.get_fitting_pars(trn_voxel_data, zscore_features, ridge=ridge)
    # Params for the spatial aspect of the model (possible pRFs)
    aperture_rf_range = 1.1
    aperture, models = initialize_fitting.get_prf_models(aperture_rf_range=aperture_rf_range)    
    
    if 'pyramid' in fitting_type:
        
        # Set up the pyramid
        _fmaps_fn = texture_statistics_pyramid.steerable_pyramid_extractor(pyr_height=n_sf, n_ori = n_ori)
        # Initialize the "texture" model which builds on first level feature maps
        _feature_extractor = texture_statistics_pyramid.texture_feature_extractor(_fmaps_fn,sample_batch_size=sample_batch_size, feature_types_exclude=feature_types_exclude, n_prf_sd_out=n_prf_sd_out, aperture=aperture, device=device, \
                                      group_all_hl_feats = group_all_hl_feats, do_varpart = do_varpart)
        feature_info = [_feature_extractor.feature_column_labels, _feature_extractor.feature_types_include]
        
        if 'plus_bdcn' in fitting_type:
             # Set up the contour feature extractor
            pretrained_model_file = os.path.join(bdcn_path,'params','bdcn_pretrained_on_bsds500.pth')
            _feature_extractor2 = bdcn_features.bdcn_feature_extractor(pretrained_model_file, device, aperture_rf_range, \
                                                                      n_prf_sd_out, \
                                                       batch_size=10, map_ind=map_ind, mult_patch_by_prf=mult_patch_by_prf,
                                                    downsample_factor = downsample_factor, do_nms = do_nms)
            
            _feature_extractor = merge_features.combined_feature_extractor([_feature_extractor, _feature_extractor2], \
                                                                           ['pyramid','bdcn'], do_varpart = do_varpart)
        
        
    elif 'bdcn' in fitting_type:
        
         # Set up the contour feature extractor
        pretrained_model_file = os.path.join(bdcn_path,'params','bdcn_pretrained_on_bsds500.pth')
        _feature_extractor = bdcn_features.bdcn_feature_extractor(pretrained_model_file, device, aperture_rf_range, n_prf_sd_out, \
                                                   batch_size=10, map_ind=map_ind, mult_patch_by_prf=mult_patch_by_prf,
                                                downsample_factor = downsample_factor, do_nms = do_nms)
        
    elif 'gabor' in fitting_type:
        
         # Set up the filters
        _gaborizer_complex, _gaborizer_simple, _fmaps_fn_complex, _fmaps_fn_simple = \
                initialize_fitting.get_feature_map_simple_complex_fn(n_ori, n_sf, padding_mode=padding_mode, device=device, nonlin_fn=nonlin_fn)
        # Params for the spatial aspect of the model (possible pRFs)
         # Initialize the "texture" model which builds on first level feature maps
        autocorr_output_pix=5
        _feature_extractor = texture_statistics_gabor.texture_feature_extractor(_fmaps_fn_complex, _fmaps_fn_simple, sample_batch_size=sample_batch_size, \
                         feature_types_exclude=feature_types_exclude, autocorr_output_pix=autocorr_output_pix, n_prf_sd_out=n_prf_sd_out, \
                                                                         aperture=aperture, device=device)

     
    #### DO THE ACTUAL MODEL FITTING HERE ####
    
    if do_fitting:
        gc.collect()
        torch.cuda.empty_cache()
        print('\nStarting training...\n')
        if shuff_rnd_seed==0:
            shuff_rnd_seed = int(time.strftime('%M%H%d', time.localtime()))       
        if debug:
            print('flipping the models upside down to start w biggest pRFs')
            models = np.flipud(models)

        # add an intercept
        add_bias=True
        # determines whether to shuffle before separating the nested heldout data for lambda and param selection. 
        # always using true.
        shuffle=True 
        best_losses, best_lambdas, best_params = fwrf_fit.fit_fwrf_model(trn_stim_data, trn_voxel_data, _feature_extractor, models, \
                                                       lambdas, zscore=zscore_features, add_bias=add_bias, \
                                                       voxel_batch_size=voxel_batch_size, holdout_size=holdout_size, \
                                                       shuffle=shuffle, shuff_rnd_seed=shuff_rnd_seed, device=device, \
                                                                                       do_varpart=do_varpart, debug=debug)
        
        if do_pca and 'bdcn' in fitting_type:
            if 'pyramid_plus_bdcn' in fitting_type:
                m = _feature_extractor.modules[1]
            else:
                m = _feature_extractor
            pc = [m.pct_var_expl, m.min_pct_var,  m.n_comp_needed]
        else:
            pc = None
            
        partial_masks, partial_version_names = _feature_extractor.get_partial_versions()
            
        sys.stdout.flush()
        val_cc=None
        val_r2=None
       
        save_all(fn2save)       
        print('\nSaved training results\n')        
        sys.stdout.flush()
    
    else:
        
        print('\nLoading the results of training from %s\n'%fn2save)
        out = torch.load(fn2save)
        best_losses = out['best_losses']
        best_lambdas = out['best_lambdas']
        best_params = out['best_params']
        feature_info = out['feature_info']
        val_cc = out['val_cc']
        val_r2 = out['val_r2']
        pc = out['pc']
        pc2save = pc
        if 'shuff_rnd_seed' in list(out.keys()):
            shuff_rnd_seed=out['shuff_rnd_seed']
        else:
            shuff_rnd_seed=0
            
        # some checks to make sure we're resuming same process...        
        assert(np.all(models==out['models']))
        assert(out['n_prf_sd_out']== n_prf_sd_out)
        assert(out['zscore_features']==zscore_features)
        assert(out['up_to_sess']==up_to_sess)
        assert(out['best_params'][1].shape[1]==feature_info[0].shape[0])
        assert(out['best_params'][1].shape[0]==voxel_index[0].shape[0])
         

    if val_cc is not None:
        do_val=False
        print('\nValidation is already done! not going to run it again.')       
   
    ######### VALIDATE MODEL ON HELD-OUT TEST SET ##############################################
    sys.stdout.flush()
    if do_val: 
        gc.collect()
        torch.cuda.empty_cache()
        print('about to start validation')
        sys.stdout.flush()
        
        val_cc, val_r2 = fwrf_predict.validate_fwrf_model(best_params, models, val_voxel_data, val_stim_data, _feature_extractor, \
                                   sample_batch_size=sample_batch_size, voxel_batch_size=voxel_batch_size, debug=debug, dtype=fpX)
                                     
        save_all(fn2save)
   

    ########## SUPPORT FUNCTIONS HERE ###############

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int,default=1,
                    help="number of the subject, 1-8")
    parser.add_argument("--volume_space", type=int, default=1,
                    help="want to do fitting with volume space or surface space data? 1 for volume, 0 for surface.")
    
    
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
    parser.add_argument("--group_all_hl_feats", type=int,default=1,
                    help="want to simplify groups of features in texture model? 1 for yes, 0 for no")
    
    parser.add_argument("--do_pca", type=int, default=1,
                    help="want to do PCA before fitting only works for BDCN model for now. 1 for yes, 0 for no")
    parser.add_argument("--min_pct_var", type=int,default=95,
                    help="minimum percent var to use when choosing num pcs to retain, default 95")
    parser.add_argument("--max_pc_to_retain", type=int,default=100,
                    help="maximum number of pcs to retain, default 100")
    
    parser.add_argument("--map_ind", type=int, default=-1, 
                    help="which map to use in BDCN model? Default is -1 which gives fused map")
    parser.add_argument("--n_prf_sd_out", type=int, default=2, 
                    help="How many pRF stddevs to use in patch for BDCN model? Default is 2")
    parser.add_argument("--mult_patch_by_prf", type=int, default=1,
                    help="In BDCN model, want to multiply the feature map patch by pRF gaussian? 1 for yes, 0 for no")
    parser.add_argument("--do_nms", type=int, default=1,
                    help="In BDCN model, want to apply non-maximal suppression to thin edge maps? 1 for yes, 0 for no")
    parser.add_argument("--downsample_factor", type=np.float32, default=1,
                    help="In BDCN model, downsample edge maps before getting feautures? 1 for yes, 0 for no")
    
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
    parser.add_argument("--do_varpart", type=int,default=1,
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

    if 'gabor' in args.fitting_type:
        raise RuntimeError('need to update this code before running gabor model!')
        
    fit_fwrf(fitting_type = args.fitting_type, subject=args.subject, volume_space = args.volume_space, up_to_sess = args.up_to_sess, \
             n_ori = args.n_ori, n_sf = args.n_sf, nonlin_fn = args.nonlin_fn==1,  padding_mode = args.padding_mode, \
             include_simple = args.include_simple==1, include_complex = args.include_complex==1, \
             include_autocorrs = args.include_autocorrs==1, include_crosscorrs = args.include_crosscorrs==1, \
             group_all_hl_feats = args.group_all_hl_feats, \
             sample_batch_size = args.sample_batch_size, voxel_batch_size = args.voxel_batch_size, \
             zscore_features = args.zscore_features==1, ridge = args.ridge==1, \
             shuffle_images = args.shuffle_images==1, random_images = args.random_images==1, random_voxel_data = args.random_voxel_data==1, \
             do_fitting = args.do_fitting==1, do_val = args.do_val==1, do_varpart = args.do_varpart==1, 
             shuff_rnd_seed = args.shuff_rnd_seed, debug = args.debug, \
             do_pca = args.do_pca==1, min_pct_var = args.min_pct_var, max_pc_to_retain = args.max_pc_to_retain, map_ind = args.map_ind, \
             n_prf_sd_out = args.n_prf_sd_out, mult_patch_by_prf = args.mult_patch_by_prf==1, \
             downsample_factor = args.downsample_factor, do_nms = args.do_nms==1)
