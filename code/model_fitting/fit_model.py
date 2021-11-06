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
from feature_extraction import texture_statistics_gabor, sketch_token_features, \
                texture_statistics_pyramid, alexnet_features
from utils import nsd_utils, roi_utils, default_paths, coco_utils

import initialize_fitting as initialize_fitting
import arg_parser as arg_parser
import merge_features, fwrf_fit, fwrf_predict, reconstruct

fpX = np.float32
device = initialize_fitting.init_cuda()

#################################################################################################
        
    
def fit_fwrf(fitting_type, fitting_type2=None, \
             subject=1, volume_space = True, up_to_sess = 1, \
             n_ori = 4, n_sf = 4, \
             group_all_hl_feats = False, \
             sample_batch_size = 50, voxel_batch_size = 100, \
             zscore_features = True, zscore_in_groups = False, ridge = True, \
             shuffle_images = False, random_images = False, random_voxel_data = False, \
             do_fitting = True, use_precomputed_prfs = False, do_val = True, \
             do_stack=False, do_tuning=True, do_sem_disc=True, \
             do_varpart = True, do_roi_recons=False, do_voxel_recons=False, date_str = 0, \
             shuff_rnd_seed = 0, debug = False, \
             use_pca_st_feats = False, use_lda_st_feats = False, lda_discrim_type = None, \
             use_pca_pyr_feats_ll = False, use_pca_pyr_feats_hl = False,\
             min_pct_var = 99, max_pc_to_retain = 400, \
             max_pc_to_retain_pyr_ll = 100, max_pc_to_retain_pyr_hl = 100, \
             alexnet_layer_name='Conv5_ReLU', alexnet_padding_mode=None, \
             which_prf_grid=1):
    
    def save_all(fn2save, fitting_type):
        """
        Define all the important parameters that have to be saved
        """
        dict2save = {
        'subject': subject,
        'volume_space': volume_space,
        'fitting_type': fitting_type,
        'fitting_type2': fitting_type2,
        'voxel_mask': voxel_mask,
        'brain_nii_shape': brain_nii_shape,
        'image_order': image_order,
        'voxel_index': voxel_index,
        'voxel_roi': voxel_roi,
        'voxel_ncsnr': voxel_ncsnr, 
        'aperture': aperture,
        'aperture_rf_range': aperture_rf_range,
        'which_prf_grid': which_prf_grid,
        'models': models,        
        'best_losses': best_losses,           
        'best_lambdas': best_lambdas,
        'best_params': best_params,       
        'lambdas': lambdas, 
        'val_cc': val_cc,
        'val_r2': val_r2,    
        'partial_masks': partial_masks, 
        'partial_version_names': partial_version_names,        
        'zscore_features': zscore_features, 
        'zscore_in_groups': zscore_in_groups,
        'ridge': ridge,
        'debug': debug,
        'up_to_sess': up_to_sess,
        'shuff_rnd_seed': shuff_rnd_seed,
        'use_precomputed_prfs': use_precomputed_prfs,
        }
        # Might be some more things to save, depending what kind of fitting this is
        if do_stack:
            dict2save.update({
            'stack_result': stack_result,
            'stack_result_lo': stack_result_lo,
            'partial_models_used_for_stack': partial_models_used_for_stack,
            'train_r2': train_r2, 
            'train_cc': train_cc,
            })
        if do_roi_recons:
            dict2save.update({
            'pop_recs': pop_recs
            })
        if do_voxel_recons:
            dict2save.update({
            'voxel_recs': voxel_recs
            })
        if do_tuning:
            dict2save.update({
            'corr_each_feature': corr_each_feature
            })
        if do_sem_disc:
            dict2save.update({
            'discrim_each_axis': discrim_each_axis
            })
        if 'sketch_tokens' in fitting_type:
            dict2save.update({
            'min_pct_var': min_pct_var,
            'max_pc_to_retain': max_pc_to_retain,           
            'use_pca_st_feats': use_pca_st_feats, 
            'use_lda_st_feats': use_lda_st_feats,
            'lda_discrim_type': lda_discrim_type, 
            })
            
        if 'pyramid' in fitting_type:
            dict2save.update({
            'min_pct_var': min_pct_var,
            'max_pc_to_retain_pyr_ll': max_pc_to_retain_pyr_ll,
            'max_pc_to_retain_pyr_hl': max_pc_to_retain_pyr_hl,
            'use_pca_pyr_feats_ll': use_pca_pyr_feats_ll,
            'use_pca_pyr_feats_hl': use_pca_pyr_feats_hl,
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
            'autocorr_output_pix': autocorr_output_pix,
            'group_all_hl_feats': group_all_hl_feats,
            })
        if 'alexnet' in fitting_type:
            dict2save.update({
            'alexnet_layer_name': alexnet_layer_name,
            'alexnet_padding_mode': alexnet_padding_mode,
            })
            
        print('\nSaving to %s\n'%fn2save)
        torch.save(dict2save, fn2save, pickle_protocol=4)

    if fitting_type2=='':
        fitting_type2 = None
    if date_str==0 or date_str=='0' or date_str=='':
        date_str = None
    if alexnet_padding_mode=='':
        alexnet_padding_mode=None
        
    if do_fitting==False and date_str is None:
        raise ValueError('if you want to start midway through the process (--do_fitting=False), then specify the date when training result was saved (--date_str).')

    if do_fitting==True and date_str is not None:
        raise ValueError('if you want to do fitting from scratch (--do_fitting=True), specify --date_str=None (rather than entering a date)')

    if 'pyramid' in fitting_type:
        model_name = initialize_fitting.get_pyramid_model_name(ridge, n_ori, n_sf, use_pca_pyr_feats_ll = use_pca_pyr_feats_ll, use_pca_pyr_feats_hl = use_pca_pyr_feats_hl)
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
    
    elif 'sketch_tokens' in fitting_type:
        if use_pca_st_feats:
            # not allowing both of these to be true
            use_lda_st_feats = False
            lda_discrim_type=None
        model_name = initialize_fitting.get_sketch_tokens_model_name(use_pca_st_feats, \
                                         use_lda_st_feats, lda_discrim_type, max_pc_to_retain)   
        name1 = 'sketch_tokens'
        
    elif 'alexnet' in fitting_type:
        model_name = initialize_fitting.get_alexnet_model_name(alexnet_layer_name)   
        name1 = 'alexnet'
        
    else:
        raise ValueError('fitting type "%s" not recognized'%fitting_type2)
        
    if fitting_type2 is not None:
        if 'sketch_tokens' in fitting_type2:
            model_name2 = initialize_fitting.get_sketch_tokens_model_name(use_pca_st_feats, use_lda_st_feats, \
                                                                          lda_discrim_type, max_pc_to_retain)   
            name2 = 'sketch_tokens'
        elif 'alexnet' in fitting_type2:
            model_name2 = initialize_fitting.get_alexnet_model_name(alexnet_layer_name)
            name2 = 'alexnet'
        else: 
            raise ValueError('fitting type 2 "%s" not recognized'%fitting_type2)
        model_name = model_name + '_plus_' + model_name2    

    
    if do_stack:
        stack_result = None
        stack_result_lo = None
        partial_models_used_for_stack = None
        train_r2 = None
        train_cc = None
        model_name += '_stacked'
        
    if do_voxel_recons:
        voxel_recs = None
    if do_roi_recons:
        pop_recs = None
    if do_tuning:
        corr_each_feature = None
    if do_sem_disc:
        discrim_each_axis = None
        
    output_dir, fn2save = initialize_fitting.get_save_path(subject, volume_space, model_name, shuffle_images, \
                                                           random_images, random_voxel_data, debug, date_str)
    
    # decide what voxels to use  
    voxel_mask, voxel_index, voxel_roi, voxel_ncsnr, brain_nii_shape = roi_utils.get_voxel_roi_info(subject, \
                                                            volume_space, include_all=True, include_body=True)

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
   
    # More params for fitting
    holdout_size, lambdas = initialize_fitting.get_fitting_pars(trn_voxel_data, zscore_features, ridge=ridge)
    # Params for the spatial aspect of the model (possible pRFs)
    aperture_rf_range = 1.1
    aperture, models = initialize_fitting.get_prf_models(aperture_rf_range=aperture_rf_range, which_grid=which_prf_grid) 

    if use_precomputed_prfs:
        best_model_each_voxel = initialize_fitting.load_precomputed_prfs(fitting_type,subject)
        print(trn_voxel_data.shape)
        print(len(best_model_each_voxel))
        assert(len(best_model_each_voxel)==trn_voxel_data.shape[1])
    else:
        best_model_each_voxel = None
        
    if 'pyramid' in fitting_type:
        
        # Set up the pyramid
        compute_features = False
        _fmaps_fn = texture_statistics_pyramid.steerable_pyramid_extractor(pyr_height = n_sf, n_ori = n_ori)
        # Initialize the "texture" model which builds on first level feature maps
        _feature_extractor = texture_statistics_pyramid.texture_feature_extractor(_fmaps_fn,\
                  subject=subject, feature_types_exclude=feature_types_exclude, \
                  which_prf_grid=which_prf_grid, \
                  do_varpart = do_varpart, zscore_in_groups = zscore_in_groups,\
                  group_all_hl_feats = group_all_hl_feats, compute_features = compute_features, \
                  use_pca_feats_ll = use_pca_pyr_feats_ll, use_pca_feats_hl = use_pca_pyr_feats_hl, \
                  min_pct_var = min_pct_var, max_pc_to_retain_ll = max_pc_to_retain_pyr_ll, \
                  max_pc_to_retain_hl = max_pc_to_retain_pyr_hl, device=device)
        feature_info = [_feature_extractor.feature_column_labels, _feature_extractor.feature_types_include]
        
    elif 'gabor' in fitting_type:
        
        # Set up the Gabor filtering modules
        _gabor_ext_complex, _gabor_ext_simple, _fmaps_fn_complex, _fmaps_fn_simple = \
                initialize_fitting.get_gabor_feature_map_fn(n_ori, n_sf,device=device);    
        # Initialize the "texture" model which builds on first level feature maps
        autocorr_output_pix=5
        compute_features = False
        _feature_extractor = texture_statistics_gabor.texture_feature_extractor(_fmaps_fn_complex, _fmaps_fn_simple, \
                                subject=subject, which_prf_grid=which_prf_grid, \
                                autocorr_output_pix=autocorr_output_pix, \
                                feature_types_exclude=feature_types_exclude, do_varpart=do_varpart, \
                                group_all_hl_feats=group_all_hl_feats, compute_features = compute_features, \
                                device=device)      
        feature_info = [_feature_extractor.feature_column_labels, _feature_extractor.feature_types_include]
        
    elif 'sketch_tokens' in fitting_type:

        _feature_extractor = sketch_token_features.sketch_token_feature_extractor(subject=subject, device=device,\
                 which_prf_grid=which_prf_grid, \
                 use_pca_feats = use_pca_st_feats, min_pct_var = min_pct_var, max_pc_to_retain = max_pc_to_retain, \
                 use_lda_feats = use_lda_st_feats, lda_discrim_type = lda_discrim_type, zscore_in_groups = zscore_in_groups)
    
    elif 'alexnet' in fitting_type:
        
        if alexnet_layer_name=='all_conv':
            assert(fitting_type2 is None)
            fe = []
            n_layers = 5
            names = ['Conv%d_ReLU'%(ll+1) for ll in range(n_layers)]
            for ll in range(n_layers):
                fe.append(alexnet_features.alexnet_feature_extractor(subject=subject, \
                             layer_name=names[ll], device=device, which_prf_grid=which_prf_grid, \
                                                                 padding_mode = alexnet_padding_mode))
            _feature_extractor = merge_features.combined_feature_extractor(fe, names, do_varpart=do_varpart)
               
        else:
            _feature_extractor = alexnet_features.alexnet_feature_extractor(subject=subject, \
                                     layer_name=alexnet_layer_name, device=device, \
                                    which_prf_grid=which_prf_grid, padding_mode = alexnet_padding_mode)
    
        
    if fitting_type2 is not None:

        if 'sketch_tokens' in fitting_type2:

            _feature_extractor2 = sketch_token_features.sketch_token_feature_extractor(subject=subject, \
                device=device,which_prf_grid=which_prf_grid, \
                 use_pca_feats = use_pca_st_feats, min_pct_var = min_pct_var, max_pc_to_retain = max_pc_to_retain, \
             use_lda_feats = use_lda_st_feats, lda_discrim_type = lda_discrim_type,\
                                       zscore_in_groups = zscore_in_groups)
            
        elif 'alexnet' in fitting_type2:
            assert(alexnet_layer_name is not 'all_conv')
            _feature_extractor2 = alexnet_features.alexnet_feature_extractor(subject=subject, \
                                 layer_name=alexnet_layer_name, device=device, \
                                 which_prf_grid=which_prf_grid, padding_mode = alexnet_padding_mode)
            
            
        _feature_extractor = merge_features.combined_feature_extractor([_feature_extractor, \
                                _feature_extractor2], [name1,name2], do_varpart = do_varpart)

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
        print(len(trn_stim_data))
    
        best_losses, best_lambdas, best_params, best_train_holdout_preds, holdout_trial_order = \
                            fwrf_fit.fit_fwrf_model(trn_stim_data, trn_voxel_data, \
                                   _feature_extractor, models, \
                                   lambdas, best_model_each_voxel = best_model_each_voxel, \
                                   zscore=zscore_features, add_bias=add_bias, \
                                   voxel_batch_size=voxel_batch_size, holdout_size=holdout_size, \
                                   shuffle=shuffle, shuff_rnd_seed=shuff_rnd_seed, device=device, \
                                   dtype=fpX, debug=debug)
        trn_holdout_voxel_data_pred = best_train_holdout_preds
        
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
        if 'feature_info' in list(out.keys()):
            feature_info = out['feature_info']
        val_cc = out['val_cc']
        val_r2 = out['val_r2']
        if do_stack:
            train_cc=out['train_cc']
            train_r2=out['train_r2']
            stack_result=out['stack_result']
            stack_result_lo =out['stack_result_lo']
            partial_models_used_for_stack=out['partial_models_used_for_stack']
        if do_roi_recons:
            pop_recs=None
        if do_voxel_recons:
            voxel_recs=None
     
        shuff_rnd_seed=out['shuff_rnd_seed']
        
        assert(out['up_to_sess']==up_to_sess)
       
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
        # Here is where any model-specific additional initialization steps are done
        # Includes initializing pca params arrays, if doing pca
        image_size = None
        _feature_extractor.init_for_fitting(image_size=image_size, models=models, dtype=fpX)
        val_cc, val_r2, val_voxel_data_pred, features_each_prf = \
            fwrf_predict.validate_fwrf_model(best_params, models, val_voxel_data, val_stim_data, \
                     _feature_extractor, zscore=zscore_features, sample_batch_size=sample_batch_size, \
                                         voxel_batch_size=voxel_batch_size, debug=debug, dtype=fpX)
        partial_masks, partial_version_names = _feature_extractor.get_partial_versions()
                                     
        save_all(fn2save, fitting_type)
        
    ### ESTIMATE VOXELS' FEATURE TUNING BASED ON CORRELATION WITH EACH FEATURE ######
    sys.stdout.flush()
    if do_tuning:
  
        gc.collect()
        torch.cuda.empty_cache()
        print('about to start feature tuning analysis')
        sys.stdout.flush()
        corr_each_feature = fwrf_predict.get_feature_tuning(best_params, features_each_prf, val_voxel_data_pred, debug=debug)
        
        save_all(fn2save, fitting_type)
        
    ### ESTIMATE SEMANTIC DISCRIMINABILITY OF EACH VOXEL'S PREDICTED RESPONSES ######
    sys.stdout.flush()
    if do_sem_disc:
  
        gc.collect()
        torch.cuda.empty_cache()
        print('about to start semantic discriminability analysis')
        sys.stdout.flush()
        labels_all = coco_utils.load_labels_each_prf(subject, which_prf_grid, \
                                                 image_inds=val_stim_data, models=models,verbose=False)
        discrim_each_axis = fwrf_predict.get_semantic_discrim(best_params, labels_all, \
                                                      val_voxel_data_pred, debug=debug)
        
        save_all(fn2save, fitting_type)
        
    ######### COMPUTE STACKING WEIGHTS AND PERFORMANCE OF STACKED MODELS ###########
    sys.stdout.flush()
    if do_stack: 
        
        if len(partial_version_names)>1:
            gc.collect()
            torch.cuda.empty_cache()
            print('about to start stacking analysis')
            sys.stdout.flush()

            trn_holdout_voxel_data = trn_voxel_data[holdout_trial_order,:]
            stack_result, stack_result_lo, partial_models_used_for_stack, train_r2, train_cc  = \
                fwrf_predict.run_stacking(_feature_extractor, trn_holdout_voxel_data, val_voxel_data, \
                                          trn_holdout_voxel_data_pred, val_voxel_data_pred, debug=debug)

            save_all(fn2save, fitting_type)
        else:
            print('Skipping stacking analysis because you only have one set of features.')

    ######### INVERTED ENCODING MODEL #############################
    sys.stdout.flush()
    if do_roi_recons: 

        gc.collect()
        torch.cuda.empty_cache()
        print('about to start ROI reconstruction analysis')
        sys.stdout.flush()

        roi_def = roi_utils.get_combined_rois(subject, volume_space=volume_space, \
                                      include_all=True, include_body=True, verbose=False)
        pop_recs = \
            reconstruct.get_population_recons(best_params, models, val_voxel_data, roi_def, \
                  val_stim_data, _feature_extractor, zscore=zscore_features, debug=debug, dtype=fpX)
        save_all(fn2save, fitting_type)
        
    sys.stdout.flush()
    if do_voxel_recons: 

        gc.collect()
        torch.cuda.empty_cache()
        print('about to start voxelwise reconstruction analysis')
        sys.stdout.flush()

        voxel_recs = \
            reconstruct.get_single_voxel_recons(best_params, models, val_voxel_data, val_stim_data, \
                                      _feature_extractor, zscore=zscore_features, debug=debug, dtype=fpX)
        save_all(fn2save, fitting_type)
        
           
            
            
    ########## SUPPORT FUNCTIONS HERE ###############

if __name__ == '__main__':
    
    # get all the arguments (in separate file because there are many)
    args = arg_parser.get_args()
   
    # now actually call the function to execute fitting...
 
    fit_fwrf(fitting_type = args.fitting_type, fitting_type2 = args.fitting_type2, \
             subject=args.subject, volume_space = args.volume_space, \
             up_to_sess = args.up_to_sess, \
             n_ori = args.n_ori, n_sf = args.n_sf,
             group_all_hl_feats = args.group_all_hl_feats, \
             sample_batch_size = args.sample_batch_size, voxel_batch_size = args.voxel_batch_size, \
             zscore_features = args.zscore_features==1, zscore_in_groups = args.zscore_in_groups==1, \
             ridge = args.ridge==1, \
             shuffle_images = args.shuffle_images==1, random_images = args.random_images==1, \
             random_voxel_data = args.random_voxel_data==1, \
             do_fitting = args.do_fitting==1, use_precomputed_prfs = args.use_precomputed_prfs==1, \
             do_val = args.do_val==1, do_stack = args.do_stack==1, \
             do_tuning = args.do_tuning==1, do_sem_disc = args.do_sem_disc==1, \
             do_varpart = args.do_varpart==1, do_roi_recons = args.do_roi_recons==1, \
             do_voxel_recons = args.do_voxel_recons==1, date_str = args.date_str, \
             shuff_rnd_seed = args.shuff_rnd_seed, debug = args.debug, \
             use_pca_pyr_feats_ll = args.use_pca_pyr_feats_ll==1, \
             use_pca_pyr_feats_hl = args.use_pca_pyr_feats_hl==1, \
             use_pca_st_feats = args.use_pca_st_feats==1, \
             use_lda_st_feats = args.use_lda_st_feats==1, \
             lda_discrim_type = args.lda_discrim_type, \
             min_pct_var = args.min_pct_var, max_pc_to_retain = args.max_pc_to_retain, \
             max_pc_to_retain_pyr_ll = args.max_pc_to_retain_pyr_ll, \
             max_pc_to_retain_pyr_hl = args.max_pc_to_retain_pyr_hl, \
             alexnet_layer_name = args.alexnet_layer_name, \
             alexnet_padding_mode = args.alexnet_padding_mode, \
             which_prf_grid = args.which_prf_grid)
             
