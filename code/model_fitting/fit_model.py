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
code_dir = '/user_data/mmhender/imStat/code/'
sys.path.append(code_dir)
from feature_extraction import texture_statistics_gabor, texture_statistics_pyramid, bdcn_features, sketch_token_features
from utils import nsd_utils, roi_utils, default_paths

import initialize_fitting2 as initialize_fitting
import arg_parser2 as arg_parser
import merge_features, fwrf_fit, fwrf_predict

fpX = np.float32
device = initialize_fitting.init_cuda()

#################################################################################################
        
    
def fit_fwrf(fitting_type, subject=1, volume_space = True, up_to_sess = 1, \
             n_ori = 4, n_sf = 4, nonlin_fn = False,  padding_mode = 'circular', \
             group_all_hl_feats = False, \
             sample_batch_size = 50, voxel_batch_size = 100, \
             zscore_features = True, ridge = True, \
             shuffle_images = False, random_images = False, random_voxel_data = False, \
             do_fitting = True, do_val = True, do_varpart = True, date_str = 0, \
             shuff_rnd_seed = 0, debug = False, \
             do_pca_st = True, do_pca_bdcn = True, do_pca_pyr_hl = False, min_pct_var = 99, max_pc_to_retain = 400, \
             map_ind = -1, n_prf_sd_out = 2, mult_patch_by_prf = True, do_avg_pool = True, \
             downsample_factor = 1.0, do_nms = False):
    
    def save_all(fn2save, fitting_type):
        """
        Define all the important parameters that have to be saved
        """
        dict2save = {
        'subject': subject,
        'volume_space': volume_space,
        'fitting_type': fitting_type,
        'voxel_mask': voxel_mask,
        'brain_nii_shape': brain_nii_shape,
        'image_order': image_order,
        'voxel_index': voxel_index,
        'voxel_roi': voxel_roi,
        'voxel_ncsnr': voxel_ncsnr, 
        'aperture': aperture,
        'aperture_rf_range': aperture_rf_range,
        'models': models,        
        'n_prf_sd_out': n_prf_sd_out,
        'best_losses': best_losses,           
        'best_lambdas': best_lambdas,
        'best_params': best_params,       
        'lambdas': lambdas, 
        'val_cc': val_cc,
        'val_r2': val_r2,    
        'partial_masks': partial_masks, 
        'partial_version_names': partial_version_names,
        'zscore_features': zscore_features,        
        'ridge': ridge,
        'debug': debug,
        'up_to_sess': up_to_sess,
        'shuff_rnd_seed': shuff_rnd_seed
        }
        # Might be some more things to save, depending what kind of fitting this is
        if 'bdcn' in fitting_type:
            dict2save.update({
            'pc': pc,
            'min_pct_var': min_pct_var,
            'max_pc_to_retain': max_pc_to_retain,           
            'mult_patch_by_prf': mult_patch_by_prf,
            'do_nms': do_nms, 
            'downsample_factor': downsample_factor,
            })
            
        if 'sketch_tokens' in fitting_type:
            dict2save.update({
            'pc': pc,
            'min_pct_var': min_pct_var,
            'max_pc_to_retain': max_pc_to_retain,           
            'mult_patch_by_prf': mult_patch_by_prf,
            'map_resolution': map_resolution, 
            'do_avg_pool': do_avg_pool,
            })
            
        if 'pyramid' in fitting_type:
            dict2save.update({
            'pc': pc,
            'min_pct_var': min_pct_var,
            'max_pc_to_retain': max_pc_to_retain,   
            'feature_info':feature_info,
            'group_all_hl_feats': group_all_hl_feats,
            })
            
        if 'gabor' in fitting_type:
            dict2save.update({
            'feature_table_simple': _gabor_ext_simple.feature_table,
            'filter_pars_simple': _gabor_ext_simple.gabor_filter_pars,
            'orient_filters_simple': _gabor_ext_simple.filter_stack,  
            'feature_table_complex': _gabor_ext_complex.feature_table,
            'filter_pars_complex': _gabor_ext_complex.gabor_filter_pars,
            'orient_filters_complex': _gabor_ext_complex.filter_stack, 
            'feature_types_exclude': feature_types_exclude,
            'feature_info':feature_info,
            'nonlin_fn': nonlin_fn,
            'padding_mode': padding_mode,
            'autocorr_output_pix': autocorr_output_pix,
            'group_all_hl_feats': group_all_hl_feats,
            })
            
        print('\nSaving to %s\n'%fn2save)
        torch.save(dict2save, fn2save, pickle_protocol=4)

    if date_str==0:
        date_str = None
        
    if do_fitting==False and date_str is None:
        raise ValueError('if you want to start midway through the process (--do_fitting=False), then specify the date when training result was saved (--date_str).')

    if do_fitting==True and date_str is not None:
        raise ValueError('if you want to do fitting from scratch (--do_fitting=True), specify --date_str=None (rather than entering a date)')

    if do_fitting==False and (do_pca_pyr_hl or do_pca_st or do_pca_bdcn):
        raise ValueError('Cannot start midway through the process (--do_fitting=False) when doing pca, because the pca weight matrix is not saved in between trn/val.')
        
    if 'pyramid' in fitting_type:
        model_name = initialize_fitting.get_pyramid_model_name(ridge, n_ori, n_sf, do_pca_hl = do_pca_pyr_hl)
        feature_types_exclude = []        
        name1 = 'pyramid_texture'
        
    elif 'gabor_texture' in fitting_type:        
        model_name = initialize_fitting.get_gabor_texture_model_name(ridge, n_ori, n_sf)
        feature_types_exclude = []
        name1 = 'gabor_texture'
        
    elif 'gabor_solo' in fitting_type:        
        model_name = initialize_fitting.get_gabor_solo_model_name(ridge, n_ori, n_sf)
        feature_types_exclude = ['pixel', 'simple_feature_means', 'autocorrs', 'crosscorrs']
        name1 = 'gabor_solo'
        
    elif 'bdcn' in fitting_type:
        model_name = initialize_fitting.get_bdcn_model_name(do_pca_bdcn, map_ind)   
        name1 = 'bdcn'
        
    elif 'sketch_tokens' in fitting_type:
        model_name = initialize_fitting.get_sketch_tokens_model_name(do_pca_st)   
        name1 = 'sketch_tokens'
        
    else:
        raise ValueError('your string for fitting_type was not recognized')
        
    if 'plus_sketch_tokens' in fitting_type:
        model_name2 = initialize_fitting.get_sketch_tokens_model_name(do_pca)
        model_name = model_name + '_plus_' + model_name2
    elif 'plus_bdcn' in fitting_type:
        model_name2 = initialize_fitting.get_bdcn_model_name(do_pca, map_ind)
        model_name = model_name + '_plus_' + model_name2
               
        
    output_dir, fn2save = initialize_fitting.get_save_path(subject, volume_space, model_name, shuffle_images, random_images, random_voxel_data, debug, date_str)
    
    # decide what voxels to use  
    voxel_mask, voxel_index, voxel_roi, voxel_ncsnr, brain_nii_shape = roi_utils.get_voxel_roi_info(subject, volume_space)

    sessions = np.arange(0,up_to_sess)
    zscore_betas_within_sess = True
    # get all data and corresponding images, in two splits. always fixed set that gets left out
    trn_stim_data, trn_voxel_data, val_stim_data, val_voxel_data, \
            image_order, image_order_trn, image_order_val = nsd_utils.get_data_splits(subject, sessions=sessions, \
                                                                         voxel_mask=voxel_mask, volume_space=volume_space, \
                                                                          zscore_betas_within_sess=zscore_betas_within_sess, \
                                                                          shuffle_images=shuffle_images, random_images=random_images, \
                                                                                             random_voxel_data=random_voxel_data)

    
    if 'gabor' in fitting_type or 'sketch_tokens' in fitting_type or 'pyramid' in fitting_type:
        # For this model, the features are pre-computed, so we will just load them rather than passing in images.
        # Going to pass the image indices (into 10,000 dim array) instead of images to fitting and val functions, 
        # which will tell which features to load.
        trn_stim_data = image_order_trn
        val_stim_data = image_order_val
        
    # More params for fitting
    holdout_size, lambdas = initialize_fitting.get_fitting_pars(trn_voxel_data, zscore_features, ridge=ridge)
    # Params for the spatial aspect of the model (possible pRFs)
    aperture_rf_range = 1.1
    aperture, models = initialize_fitting.get_prf_models(aperture_rf_range=aperture_rf_range)    
    
    if 'pyramid' in fitting_type:
        
        # Set up the pyramid
        compute_features = False
        _fmaps_fn = texture_statistics_pyramid.steerable_pyramid_extractor(pyr_height = n_sf, n_ori = n_ori)
        # Initialize the "texture" model which builds on first level feature maps
        _feature_extractor = texture_statistics_pyramid.texture_feature_extractor(_fmaps_fn,sample_batch_size=sample_batch_size, \
                                      subject=subject, feature_types_exclude=feature_types_exclude, n_prf_sd_out=n_prf_sd_out,\
                                      aperture=aperture, do_varpart = do_varpart, \
                                      group_all_hl_feats = group_all_hl_feats, compute_features = compute_features, \
                                      do_pca_hl = do_pca_pyr_hl, min_pct_var = min_pct_var, max_pc_to_retain = max_pc_to_retain, \
                                                                                  device=device)
        feature_info = [_feature_extractor.feature_column_labels, _feature_extractor.feature_types_include]
        
    elif 'gabor' in fitting_type:
        
        # Set up the Gabor filtering modules
        _gabor_ext_complex, _gabor_ext_simple, _fmaps_fn_complex, _fmaps_fn_simple = \
                initialize_fitting.get_gabor_feature_map_fn(n_ori, n_sf, padding_mode=padding_mode, device=device, \
                                                                     nonlin_fn=nonlin_fn);    
        # Initialize the "texture" model which builds on first level feature maps
        autocorr_output_pix=5
        compute_features = False
        _feature_extractor = texture_statistics_gabor.texture_feature_extractor(_fmaps_fn_complex, _fmaps_fn_simple, \
                                                                                subject=subject,\
                                                sample_batch_size=sample_batch_size, autocorr_output_pix=autocorr_output_pix, \
                                                n_prf_sd_out=n_prf_sd_out, aperture=aperture, \
                                                feature_types_exclude=feature_types_exclude, do_varpart=do_varpart, \
                                                group_all_hl_feats=group_all_hl_feats, compute_features = compute_features, \
                                                                                device=device)      
        feature_info = [_feature_extractor.feature_column_labels, _feature_extractor.feature_types_include]
        
    elif 'sketch_tokens' in fitting_type:
        
        map_resolution = 227
        _feature_extractor = sketch_token_features.sketch_token_feature_extractor(subject, device, map_resolution=map_resolution, \
                                                                                  aperture = aperture, \
                                                             n_prf_sd_out = n_prf_sd_out, \
                                             batch_size=sample_batch_size, mult_patch_by_prf=mult_patch_by_prf, do_avg_pool = do_avg_pool,\
                                             do_pca = do_pca_st, min_pct_var = min_pct_var, max_pc_to_retain = max_pc_to_retain)
        
    elif 'bdcn' in fitting_type:
        
         # Set up the contour feature extractor
        pretrained_model_file = os.path.join(default_paths.bdcn_path,'params','bdcn_pretrained_on_bsds500.pth')
        _feature_extractor = bdcn_features.bdcn_feature_extractor(pretrained_model_file, device, aperture=aperture, \
                                                                  n_prf_sd_out = n_prf_sd_out, \
                                                   batch_size=10, map_ind=map_ind, mult_patch_by_prf=mult_patch_by_prf,
                                                downsample_factor = downsample_factor, do_nms = do_nms, \
                                             do_pca = do_pca_bdcn, min_pct_var = min_pct_var, max_pc_to_retain = max_pc_to_retain)

        
    if 'plus_sketch_tokens' in fitting_type:
        map_resolution = 227
        _feature_extractor2 = sketch_token_features.sketch_token_feature_extractor(subject, device, map_resolution=map_resolution, \
                                                                                   aperture = aperture, \
                                                             n_prf_sd_out = n_prf_sd_out, \
                                       batch_size=sample_batch_size, mult_patch_by_prf=mult_patch_by_prf, do_avg_pool = do_avg_pool,\
                                                   do_pca = do_pca_st, min_pct_var = min_pct_var, max_pc_to_retain = max_pc_to_retain)
        _feature_extractor = merge_features.combined_feature_extractor([_feature_extractor, _feature_extractor2], \
                                                                           [name1,'sketch_tokens'], do_varpart = do_varpart)
    
        
    elif 'plus_bdcn' in fitting_type:
             # Set up the contour feature extractor
            pretrained_model_file = os.path.join(default_paths.bdcn_path,'params','bdcn_pretrained_on_bsds500.pth')
            _feature_extractor2 = bdcn_features.bdcn_feature_extractor(pretrained_model_file, device, aperture=aperture, \
                                                                      n_prf_sd_out = n_prf_sd_out, \
                                                       batch_size=10, map_ind=map_ind, mult_patch_by_prf=mult_patch_by_prf,
                                                    downsample_factor = downsample_factor, do_nms = do_nms, \
                                              do_pca = do_pca_bdcn, min_pct_var = min_pct_var, max_pc_to_retain = max_pc_to_retain)
            
            _feature_extractor = merge_features.combined_feature_extractor([_feature_extractor, _feature_extractor2], \
                                                                           [name1,'bdcn'], do_varpart = do_varpart)
    
    
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
                                                       dtype=fpX, debug=debug)
        if 'plus' in fitting_type:
            pc = []
            for m in _feature_extractor.modules:           
                if hasattr(m, 'pct_var_expl'):
                    pcm = [m.pct_var_expl, m.min_pct_var,  m.n_comp_needed]                  
                else:
                    pcm = None
                pc.append(pcm)
        else:
            m = _feature_extractor
            if hasattr(m, 'pct_var_expl'):
                pc = [m.pct_var_expl, m.min_pct_var,  m.n_comp_needed]
            else:
                pc = None
            
        partial_masks, partial_version_names = _feature_extractor.get_partial_versions()
            
        sys.stdout.flush()
        val_cc=None
        val_r2=None
       
        save_all(fn2save, fitting_type)   
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
                                     
        save_all(fn2save, fitting_type)
   

    ########## SUPPORT FUNCTIONS HERE ###############

if __name__ == '__main__':
    
    # get all the arguments (in separate file because there are many)
    args = arg_parser.get_args()

    # now actually call the function to execute fitting...
 
    fit_fwrf(fitting_type = args.fitting_type, subject=args.subject, volume_space = args.volume_space, up_to_sess = args.up_to_sess, \
             n_ori = args.n_ori, n_sf = args.n_sf, nonlin_fn = args.nonlin_fn==1,  padding_mode = args.padding_mode, \
             group_all_hl_feats = args.group_all_hl_feats, \
             sample_batch_size = args.sample_batch_size, voxel_batch_size = args.voxel_batch_size, \
             zscore_features = args.zscore_features==1, ridge = args.ridge==1, \
             shuffle_images = args.shuffle_images==1, random_images = args.random_images==1, random_voxel_data = args.random_voxel_data==1, \
             do_fitting = args.do_fitting==1, do_val = args.do_val==1, do_varpart = args.do_varpart==1, 
             date_str = args.date_str, \
             shuff_rnd_seed = args.shuff_rnd_seed, debug = args.debug, \
             do_pca_pyr_hl = args.do_pca_pyr_hl==1, do_pca_st = args.do_pca_st==1, do_pca_bdcn = args.do_pca_bdcn==1, \
             min_pct_var = args.min_pct_var, max_pc_to_retain = args.max_pc_to_retain, map_ind = args.map_ind, \
             n_prf_sd_out = args.n_prf_sd_out, mult_patch_by_prf = args.mult_patch_by_prf==1, do_avg_pool = args.do_avg_pool==1, \
             downsample_factor = args.downsample_factor, do_nms = args.do_nms==1)
