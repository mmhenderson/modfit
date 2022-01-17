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
                texture_statistics_pyramid, alexnet_features, semantic_features, clip_features
from utils import nsd_utils, roi_utils, default_paths, coco_utils

import initialize_fitting as initialize_fitting
import arg_parser as arg_parser
import merge_features, fwrf_fit, fwrf_predict, reconstruct

fpX = np.float32
device = initialize_fitting.init_cuda()

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

#################################################################################################
        
    
def fit_fwrf(fitting_types, model_name, \
             subject=1, volume_space = True, up_to_sess = 1, single_sess = None,\
             n_ori_pyr = 4, n_sf_pyr = 4, \
             n_ori_gabor = 4, n_sf_gabor = 4, gabor_nonlin_fn=False, \
             group_all_hl_feats = False, \
             sample_batch_size = 50, voxel_batch_size = 100, \
             zscore_features = True, ridge = True, \
             shuffle_images = False, random_images = False, random_voxel_data = False, \
             do_fitting = True, use_precomputed_prfs = False, do_val = True, \
             do_stack=False, do_tuning=True, do_sem_disc=True, \
             do_varpart = True, do_roi_recons=False, do_voxel_recons=False, date_str = 0, \
             shuff_rnd_seed = 0, debug = False, \
             use_pca_st_feats = False, use_lda_st_feats = False, lda_discrim_type = None, \
             use_pca_pyr_feats_hl = False,\
             alexnet_layer_name='Conv5_ReLU', alexnet_padding_mode=None, \
             use_pca_alexnet_feats = False, \
             clip_layer_name='Block15', clip_model_architecture='RN50', \
             use_pca_clip_feats = True, \
             semantic_discrim_type=None, \
             which_prf_grid=1, save_pred_data=False):
    
    def save_all(fn2save):
    
        """
        Define all the important parameters that have to be saved
        """
        dict2save = {
        'subject': subject,
        'volume_space': volume_space,
        'fitting_types': fitting_types, 
        'voxel_mask': voxel_mask,
        'brain_nii_shape': brain_nii_shape,
        'image_order': image_order,
        'voxel_index': voxel_index,
        'voxel_roi': voxel_roi,
        'voxel_ncsnr': voxel_ncsnr, 
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
        'ridge': ridge,
        'debug': debug,
        'up_to_sess': up_to_sess,
        'single_sess': single_sess,
        'shuff_rnd_seed': shuff_rnd_seed,
        'use_precomputed_prfs': use_precomputed_prfs,
        'saved_prfs_fn': saved_prfs_fn,
        'best_layer_each_voxel': best_layer_each_voxel,
        'saved_best_layer_fn': saved_best_layer_fn,
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
        if save_pred_data:
            dict2save.update({
            'val_voxel_data': val_voxel_data,
            'val_voxel_data_pred': val_voxel_data_pred,
            'val_stim_data': val_stim_data,
            })
        if np.any(['semantic' in ft for ft in fitting_types]):
            dict2save.update({
            'semantic_discrim_type': semantic_discrim_type,
            })
        if np.any(['sketch_tokens' in ft for ft in fitting_types]):
            dict2save.update({         
            'use_pca_st_feats': use_pca_st_feats, 
            'use_lda_st_feats': use_lda_st_feats,
            'lda_discrim_type': lda_discrim_type, 
            })          
        if np.any(['pyramid' in ft for ft in fitting_types]):
            dict2save.update({
            'use_pca_pyr_feats_hl': use_pca_pyr_feats_hl,
            'pyramid_feature_info':pyramid_feature_info,
            'group_all_hl_feats': group_all_hl_feats,
            })            
        if np.any(['gabor' in ft for ft in fitting_types]):
            dict2save.update({
            'feature_table_simple': _gabor_ext_simple.feature_table,
            'filter_pars_simple': _gabor_ext_simple.gabor_filter_pars,
            'orient_filters_simple': _gabor_ext_simple.filter_stack,  
            'feature_table_complex': _gabor_ext_complex.feature_table,
            'filter_pars_complex': _gabor_ext_complex.gabor_filter_pars,
            'orient_filters_complex': _gabor_ext_complex.filter_stack, 
            'feature_types_exclude': feature_types_exclude,
            'gabor_feature_info':gabor_feature_info,
            'autocorr_output_pix': autocorr_output_pix,
            'group_all_hl_feats': group_all_hl_feats,
            'gabor_nonlin_fn': gabor_nonlin_fn,
            })
        if np.any(['alexnet' in ft for ft in fitting_types]):
            dict2save.update({
            'alexnet_layer_name': alexnet_layer_name,
            'alexnet_padding_mode': alexnet_padding_mode,
            'use_pca_alexnet_feats': use_pca_alexnet_feats, 
            })
        if np.any(['clip' in ft for ft in fitting_types]):
            dict2save.update({
            'clip_layer_name': alexnet_layer_name,
            'clip_model_architecture': clip_model_architecture,
            'use_pca_clip_feats': use_pca_clip_feats,   
            })

        print('\nSaving to %s\n'%fn2save)
        torch.save(dict2save, fn2save, pickle_protocol=4)

    if date_str==0 or date_str=='0' or date_str=='':
        date_str = None
    if alexnet_padding_mode=='':
        alexnet_padding_mode=None
    if single_sess==0:
        single_sess=None        
    if use_pca_st_feats:
        # not allowing both of these to be true
        use_lda_st_feats = False
        lda_discrim_type=None    
    if do_fitting==False and date_str is None:
        raise ValueError('if you want to start midway through the process (--do_fitting=False), then specify the date when training result was saved (--date_str).')
    if do_fitting==True and date_str is not None:
        raise ValueError('if you want to do fitting from scratch (--do_fitting=True), specify --date_str=None (rather than entering a date)')       
    if not do_fitting and do_stack:
        raise ValueError('to do stacking analysis, need to start from scratch (--do_fitting=True)')  
    if (do_sem_disc or do_tuning) and not do_val:
        raise ValueError('to do tuning analysis or semantic discriminability, need to run validation again (--do_val=True)')    
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
    if np.any(['alexnet' in ft for ft in fitting_types]):
        dnn_model='alexnet'
        n_dnn_layers = 5;
        assert(not np.any(['clip' in ft for ft in fitting_types]))
    elif np.any(['clip' in ft for ft in fitting_types]):
        dnn_model='clip'
        n_dnn_layers = 16;
        assert(not np.any(['alexnet' in ft for ft in fitting_types]))
    else:
        dnn_model = None
          
    output_dir, fn2save = initialize_fitting.get_save_path(subject, volume_space, model_name, shuffle_images, \
                                                           random_images, random_voxel_data, debug, date_str)
    
    ########## LOADING THE DATA #############################################################################
    # decide what voxels to use  
    voxel_mask, voxel_index, voxel_roi, voxel_ncsnr, brain_nii_shape = roi_utils.get_voxel_roi_info(subject, \
                                                            volume_space, include_all=True, include_body=True)

    if single_sess is not None:
        sessions = np.array([single_sess])
    else:
        sessions = np.arange(0,up_to_sess)
    zscore_betas_within_sess = True
    image_inds_only = True
    # Get all data and corresponding images, in two splits. Always a fixed set that gets left out
    trn_stim_data, trn_voxel_data, val_stim_data, val_voxel_data, \
            image_order, image_order_trn, image_order_val = nsd_utils.get_data_splits(subject, \
                                      sessions=sessions, image_inds_only = image_inds_only, \
                                      voxel_mask=voxel_mask, volume_space=volume_space, \
                                      zscore_betas_within_sess=zscore_betas_within_sess, \
                                  shuffle_images=shuffle_images, random_images=random_images, \
                                    random_voxel_data=random_voxel_data)

    n_voxels = trn_voxel_data.shape[1]
    if dnn_model is not None and (alexnet_layer_name=='best_layer' or clip_layer_name=='best_layer'):
        # special case, going to fit groups of voxels separately according to which dnn layer was best
        # creating a list of voxel masks here that will define the subsets to loop over.
        assert(do_fitting==True)
        assert(do_stack==False and do_roi_recons==False and do_voxel_recons==False)       
        best_layer_each_voxel, saved_best_layer_fn = \
                  initialize_fitting.load_best_model_layers(subject, dnn_model)
        voxel_subset_masks = [best_layer_each_voxel==ll for ll in range(n_dnn_layers)]
        assert(len(best_layer_each_voxel)==n_voxels)
    else:
        # going to fit all voxels w same model
        voxel_subset_masks = [np.ones((n_voxels,), dtype=bool)]
        best_layer_each_voxel = None;
        saved_best_layer_fn = None;
    
    if image_inds_only==True:
        # The features are pre-computed, so we will just load them rather than passing in images.
        # Going to pass the image indices (into 10,000 dim array) instead of images to fitting and val functions, 
        # which will tell which rows of feature matrices to use. 
        trn_stim_data = image_order_trn
        val_stim_data = image_order_val
   
    ########## DEFINE PARAMETERS #############################################################################
    holdout_size, lambdas = initialize_fitting.get_fitting_pars(trn_voxel_data, zscore_features, ridge=ridge, gabor_nonlin_fn=gabor_nonlin_fn)
    
    # Params for the spatial aspect of the model (possible pRFs)
    models = initialize_fitting.get_prf_models(which_grid=which_prf_grid) 
    if shuff_rnd_seed==0:
        shuff_rnd_seed = int(time.strftime('%M%H%d', time.localtime()))       

    if use_precomputed_prfs:
        # If we already computed pRFs for this subject on some model, can load those now and use them during 
        # fitting. Faster than fitting pRFs each time.
        best_model_each_voxel, saved_prfs_fn = initialize_fitting.load_precomputed_prfs(subject)
        print(trn_voxel_data.shape)
        print(len(best_model_each_voxel))
        assert(len(best_model_each_voxel)==trn_voxel_data.shape[1])
    else:
        best_model_each_voxel = None
        saved_prfs_fn = None
        
   
    # looping over subsets of the voxels - used for clip/alexnet when layer_name is "best_layer"    
    # otherwise this loop only goes once and voxel_subset_mask is all ones.
    for vi, voxel_subset_mask in enumerate(voxel_subset_masks):

        trn_voxel_data_use = trn_voxel_data[:,voxel_subset_mask]
        val_voxel_data_use = val_voxel_data[:,voxel_subset_mask]
        if best_model_each_voxel is not None:
            best_model_each_voxel_use = best_model_each_voxel[voxel_subset_mask]
        else:
            best_model_each_voxel_use = None
        print('voxel mask %d of %d, number of voxels this loop=%d'%(vi, len(voxel_subset_masks), trn_voxel_data_use.shape[1]))
        if trn_voxel_data_use.shape[1]==0:
            print('no voxels, continuing loop')
            continue
            
        ########## CREATE FEATURE EXTRACTOR MODULES ###################################################################
        # all these modules do is load sets of pre-computed features in organized way.
        # first making a list of all the modules of interest (different feature spaces)
        fe = []
        fe_names = []
        for ft in fitting_types:   

            if 'pyramid' in ft:
                # Set up the pyramid
                compute_features = False
                include_ll=True
                include_hl=True
                _fmaps_fn = texture_statistics_pyramid.steerable_pyramid_extractor(pyr_height = n_sf_pyr, n_ori = n_ori_pyr)
                # Initialize the "texture" model which builds on first level feature maps
                _feature_extractor = texture_statistics_pyramid.texture_feature_extractor(_fmaps_fn,\
                          subject=subject, include_ll=include_ll, include_hl=include_hl, \
                          which_prf_grid=which_prf_grid, \
                          do_varpart = do_varpart,\
                          group_all_hl_feats = group_all_hl_feats, \
                          compute_features = compute_features, \
                          use_pca_feats_hl = use_pca_pyr_feats_hl, \
                          device=device)
                fe.append(_feature_extractor)
                fe_names.append(ft)
                pyramid_feature_info = [_feature_extractor.feature_column_labels, _feature_extractor.feature_types_include]

            elif 'gabor' in ft:
                if 'solo' in ft:
                    feature_types_exclude = ['pixel', 'simple_feature_means', 'autocorrs', 'crosscorrs']
                    group_all_hl_feats_gabor = False
                elif 'texture' in ft:
                    feature_types_exclude = []
                    group_all_hl_feats_gabor = group_all_hl_feats
                # Set up the Gabor filtering modules
                _gabor_ext_complex, _gabor_ext_simple, _fmaps_fn_complex, _fmaps_fn_simple = \
                            initialize_fitting.get_gabor_feature_map_fn(n_ori_gabor, n_sf_gabor,device=device,\
                            nonlin_fn=gabor_nonlin_fn);    
                # Initialize the "texture" model which builds on first level feature maps
                autocorr_output_pix=5
                compute_features = False
                _feature_extractor = texture_statistics_gabor.texture_feature_extractor(_fmaps_fn_complex, _fmaps_fn_simple, \
                            subject=subject, which_prf_grid=which_prf_grid, \
                            autocorr_output_pix=autocorr_output_pix, \
                            feature_types_exclude=feature_types_exclude, do_varpart=do_varpart, \
                            group_all_hl_feats=group_all_hl_feats_gabor, nonlin_fn=gabor_nonlin_fn, \
                            compute_features = compute_features, device=device)      
                fe.append(_feature_extractor)
                fe_names.append(ft)
                gabor_feature_info = [_feature_extractor.feature_column_labels, _feature_extractor.feature_types_include]

            elif 'sketch_tokens' in ft:
                _feature_extractor = sketch_token_features.sketch_token_feature_extractor(subject=subject, device=device,\
                         which_prf_grid=which_prf_grid, \
                         use_pca_feats = use_pca_st_feats,\
                         use_lda_feats = use_lda_st_feats, lda_discrim_type = lda_discrim_type)
                fe.append(_feature_extractor)
                fe_names.append(ft)
          
            elif 'alexnet' in ft:
                if alexnet_layer_name=='all_conv':
                    names = ['Conv%d_ReLU'%(ll+1) for ll in range(n_dnn_layers)]
                    for ll in range(n_dnn_layers):
                        _feature_extractor = alexnet_features.alexnet_feature_extractor(subject=subject, \
                                     layer_name=names[ll], device=device, which_prf_grid=which_prf_grid, \
                                     padding_mode = alexnet_padding_mode, use_pca_feats=use_pca_alexnet_feats)
                        fe.append(_feature_extractor)   
                        fe_names.append('alexnet_%s'%names[ll])
                elif alexnet_layer_name=='best_layer':
                    this_layer_name = 'Conv%d_ReLU'%(vi+1)
                    print(this_layer_name)
                    _feature_extractor = alexnet_features.alexnet_feature_extractor(subject=subject, \
                                     layer_name=this_layer_name, device=device, \
                                     which_prf_grid=which_prf_grid, padding_mode = alexnet_padding_mode, \
                                     use_pca_feats=use_pca_alexnet_feats)
                    fe.append(_feature_extractor)
                    fe_names.append(ft)
                else:
                    _feature_extractor = alexnet_features.alexnet_feature_extractor(subject=subject, \
                                     layer_name=alexnet_layer_name, device=device, \
                                     which_prf_grid=which_prf_grid, padding_mode = alexnet_padding_mode, \
                                     use_pca_feats=use_pca_alexnet_feats)
                    fe.append(_feature_extractor)
                    fe_names.append(ft)
          
            elif 'clip' in ft:
                if clip_layer_name=='all_resblocks':
                    names = ['block%d'%(ll) for ll in range(n_dnn_layers)]
                    for ll in range(n_dnn_layers):
                        _feature_extractor = clip_features.clip_feature_extractor(subject=subject, \
                                     layer_name=names[ll], device=device, which_prf_grid=which_prf_grid, \
                                     model_architecture=clip_model_architecture,\
                                     use_pca_feats=use_pca_clip_feats);
                        fe.append(_feature_extractor)   
                        fe_names.append('clip_%s'%names[ll])
                elif clip_layer_name=='best_layer':
                    this_layer_name = 'block%d'%(vi)
                    print(this_layer_name)
                    _feature_extractor = clip_features.clip_feature_extractor(subject=subject, \
                                     layer_name=this_layer_name, device=device, which_prf_grid=which_prf_grid, \
                                     model_architecture=clip_model_architecture,\
                                     use_pca_feats=use_pca_clip_feats);
                    fe.append(_feature_extractor)
                    fe_names.append(ft) 
                else:
                    _feature_extractor = clip_features.clip_feature_extractor(subject=subject, \
                                     layer_name=clip_layer_name, device=device, which_prf_grid=which_prf_grid, \
                                     model_architecture=clip_model_architecture,\
                                     use_pca_feats=use_pca_clip_feats);
                    fe.append(_feature_extractor)
                    fe_names.append(ft)   
          
            elif 'semantic' in ft:
                _feature_extractor = semantic_features.semantic_feature_extractor(subject=subject, \
                                        discrim_type=semantic_discrim_type, device=device, \
                                        which_prf_grid=which_prf_grid)
                fe.append(_feature_extractor)
                fe_names.append(ft)
          
        # Now combine subsets of features into a single module
        if len(fe)>1:
            _feature_extractor = merge_features.combined_feature_extractor(fe, fe_names, do_varpart = do_varpart)
        else:
            _feature_extractor = fe[0]

        #### FIT ENCODING MODEL ###################################################################################

        if do_fitting:
            gc.collect()
            torch.cuda.empty_cache()
            print('\nStarting training...\n')
            
            # add an intercept
            add_bias=True
            # determines whether to shuffle before separating the nested heldout data for lambda and param selection. 
            # always using true.
            shuffle=True 
            print(len(trn_stim_data))

            best_losses_tmp, best_lambdas_tmp, best_weights_tmp, best_biases_tmp, \
                best_prf_models_tmp, features_mean, features_std, \
                best_train_holdout_preds, holdout_trial_order = \
                                fwrf_fit.fit_fwrf_model(trn_stim_data, trn_voxel_data_use, \
                                       _feature_extractor, models, \
                                       lambdas, best_model_each_voxel = best_model_each_voxel_use, \
                                       zscore=zscore_features, add_bias=add_bias, \
                                       voxel_batch_size=voxel_batch_size, holdout_size=holdout_size, \
                                       shuffle=shuffle, shuff_rnd_seed=shuff_rnd_seed, device=device, \
                                       dtype=fpX, debug=debug)
            trn_holdout_voxel_data_pred = best_train_holdout_preds
          
            # getting info about how variance partition was set up
            partial_masks_tmp, partial_version_names = _feature_extractor.get_partial_versions()

            # taking the fit params for this set of voxels and putting them into the full array over all voxels
            if vi==0:               
                best_losses = np.zeros((n_voxels, best_losses_tmp.shape[1]), dtype=best_losses_tmp.dtype)
                best_lambdas = np.zeros((n_voxels, best_lambdas_tmp.shape[1]), dtype=best_lambdas_tmp.dtype)
                best_weights = np.zeros((n_voxels, best_weights_tmp.shape[1], \
                                         best_weights_tmp.shape[2]), dtype=best_weights_tmp.dtype)
                best_biases = np.zeros((n_voxels, best_biases_tmp.shape[1]), dtype=best_biases_tmp.dtype)
                best_prf_models = np.zeros((n_voxels, best_prf_models_tmp.shape[1]), \
                                           dtype=best_prf_models_tmp.dtype)
                partial_masks = [[] for ii in range(len(voxel_subset_masks))]        
            
            best_losses[voxel_subset_mask,:] = best_losses_tmp
            best_lambdas[voxel_subset_mask,:] = best_lambdas_tmp
            max_features = _feature_extractor.max_features
            if best_weights.shape[1]<max_features:
                n2pad = max_features - best_weights.shape[1]
                print('padding by %d elements'%n2pad)
                print(np.shape(best_weights))
                best_weights = np.pad(best_weights, [[0,0], [0, n2pad], [0,0]])
                print(np.shape(best_weights))
            best_weights[voxel_subset_mask,0:max_features,:] = best_weights_tmp
            best_biases[voxel_subset_mask,:] = best_biases_tmp
            best_prf_models[voxel_subset_mask,:] = best_prf_models_tmp
            partial_masks[vi] = partial_masks_tmp
            print(partial_masks[vi].shape)
            # "best_params_tmp" will be passed to validation functions (just these voxels)
            # "best_params" will be saved (all voxels)
            best_params_tmp = [models[best_prf_models_tmp,:], best_weights_tmp, best_biases_tmp, \
                               features_mean, features_std, best_prf_models_tmp]
            best_params = [models[best_prf_models,:], best_weights, best_biases, \
                               features_mean, features_std, best_prf_models]
            
            sys.stdout.flush()
            if vi==0:
                val_cc=None
                val_r2=None
                if save_pred_data:
                    val_voxel_data_pred=None

            save_all(fn2save)   
            print('\nSaved training results\n')        
            sys.stdout.flush()

        else:

            # stuff that needs to happen if we are resuming this code after the "fit" step but before validation
            print('\nLoading the results of training from %s\n'%fn2save)
            out = torch.load(fn2save)
            best_losses = out['best_losses']
            best_lambdas = out['best_lambdas']
            best_params = out['best_params']
            best_params_tmp = best_params
            
            val_cc = out['val_cc']
            val_r2 = out['val_r2']

            if 'val_voxel_data_pred' in list(out.keys()):
                assert(save_pred_data)
                val_voxel_data_pred = out['val_voxel_data_pred']
            if 'corr_each_feature' in list(out.keys()):
                assert(do_tuning)
                corr_each_feature = out['corr_each_feature']
            if 'discrim_each_axis' in list(out.keys()):
                assert(do_sem_disc)
                discrim_each_axis = out['discrim_each_axis']
            if 'voxel_recs' in list(out.keys()):
                assert(do_voxel_recons)
                voxel_recs = out['voxel_recs']
            if 'pop_recs' in list(out.keys()):
                assert(do_roi_recons)
                pop_recs = out['pop_recs']
            if 'stack_result' in list(out.keys()):
                assert(do_stack)
                stack_result = out['stack_result']
                stack_result_lo = out['stack_result_lo']
                partial_models_used_for_stack = out['partial_models_used_for_stack']
                train_r2 = out['train_r2']
                train_cc = out['train_cc']

            shuff_rnd_seed=out['shuff_rnd_seed']

            assert(out['up_to_sess']==up_to_sess)
            assert(out['which_prf_grid']==which_prf_grid)

            image_size = None
            _feature_extractor.init_for_fitting(image_size=image_size, models=models, dtype=fpX)
            partial_masks, partial_version_names = _feature_extractor.get_partial_versions()


        ######### VALIDATE MODEL ON HELD-OUT TEST SET ##############################################
        sys.stdout.flush()
        if do_val: 
            gc.collect()
            torch.cuda.empty_cache()
            print('about to start validation')
            sys.stdout.flush()
    
            val_cc_tmp, val_r2_tmp, val_voxel_data_pred, features_each_prf = \
                fwrf_predict.validate_fwrf_model(best_params_tmp, models, val_voxel_data_use, val_stim_data, \
                         _feature_extractor, zscore=zscore_features, sample_batch_size=sample_batch_size, \
                                             voxel_batch_size=voxel_batch_size, debug=debug, dtype=fpX)
            if vi==0:
                val_cc = np.zeros((n_voxels, val_cc_tmp.shape[1]), dtype=val_cc_tmp.dtype)
                val_r2 = np.zeros((n_voxels, val_r2_tmp.shape[1]), dtype=val_r2_tmp.dtype)               
            val_cc[voxel_subset_mask,:] = val_cc_tmp
            val_r2[voxel_subset_mask,:] = val_r2_tmp
                
            save_all(fn2save)

        ### ESTIMATE VOXELS' FEATURE TUNING BASED ON CORRELATION WITH EACH FEATURE ######
        sys.stdout.flush()
        if do_tuning:

            gc.collect()
            torch.cuda.empty_cache()
            print('about to start feature tuning analysis')
            sys.stdout.flush()
            corr_each_feature_tmp = fwrf_predict.get_feature_tuning(best_params_tmp, features_each_prf, \
                                                                    val_voxel_data_pred, debug=debug)
            if vi==0:
                corr_each_feature = np.zeros((n_voxels, corr_each_feature_tmp.shape[1]), dtype=corr_each_feature_tmp.dtype)  
            max_features = _feature_extractor.max_features
            if corr_each_feature.shape[1]<max_features:
                n2pad = max_features - corr_each_feature.shape[1]
                print('padding by %d elements'%n2pad)
                print(np.shape(corr_each_feature))
                corr_each_feature = np.pad(corr_each_feature, [[0,0], [0, n2pad]])
                print(np.shape(corr_each_feature))
            corr_each_feature[voxel_subset_mask,0:max_features] = corr_each_feature_tmp
            
            save_all(fn2save)

        ### ESTIMATE SEMANTIC DISCRIMINABILITY OF EACH VOXEL'S PREDICTED RESPONSES ######
        sys.stdout.flush()
        if do_sem_disc:

            gc.collect()
            torch.cuda.empty_cache()
            print('about to start semantic discriminability analysis')
            sys.stdout.flush()
            labels_all = coco_utils.load_labels_each_prf(subject, which_prf_grid, \
                                                     image_inds=val_stim_data, models=models,verbose=False)
            discrim_each_axis_tmp = fwrf_predict.get_semantic_discrim(best_params_tmp, labels_all, \
                                                          val_voxel_data_pred, debug=debug)
            if vi==0:
                discrim_each_axis = np.zeros((n_voxels, discrim_each_axis_tmp.shape[1]), dtype=discrim_each_axis_tmp.dtype)            
            discrim_each_axis[voxel_subset_mask,:] = discrim_each_axis_tmp
            
            save_all(fn2save)

        ######### COMPUTE STACKING WEIGHTS AND PERFORMANCE OF STACKED MODELS ###########
        sys.stdout.flush()
        if do_stack: 

            if len(partial_version_names)>1:
                gc.collect()
                torch.cuda.empty_cache()
                print('about to start stacking analysis')
                sys.stdout.flush()

                trn_holdout_voxel_data = trn_voxel_data_use[holdout_trial_order,:]
                stack_result, stack_result_lo, partial_models_used_for_stack, train_r2, train_cc  = \
                    fwrf_predict.run_stacking(_feature_extractor, trn_holdout_voxel_data, val_voxel_data_use, \
                                              trn_holdout_voxel_data_pred, val_voxel_data_pred, debug=debug)

                save_all(fn2save)
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
                reconstruct.get_population_recons(best_params_tmp, models, val_voxel_data_use, roi_def, \
                      val_stim_data, _feature_extractor, zscore=zscore_features, debug=debug, dtype=fpX)
            save_all(fn2save)

        sys.stdout.flush()
        if do_voxel_recons: 

            gc.collect()
            torch.cuda.empty_cache()
            print('about to start voxelwise reconstruction analysis')
            sys.stdout.flush()

            voxel_recs = \
                reconstruct.get_single_voxel_recons(best_params_tmp, models, val_voxel_data_use, val_stim_data, \
                                          _feature_extractor, zscore=zscore_features, debug=debug, dtype=fpX)
            save_all(fn2save)

           
            
            
    ########## SUPPORT FUNCTIONS HERE ###############

if __name__ == '__main__':
    
    # get all the arguments (in separate file because there are many)
    args = arg_parser.get_args()
   
    
    model_name, fitting_types = initialize_fitting.get_full_save_name(args)
    
    # now actually call the function to execute fitting...
    fit_fwrf(fitting_types = fitting_types, model_name = model_name, \
             semantic_discrim_type = args.semantic_discrim_type, \
             subject=args.subject, volume_space = args.volume_space, \
             up_to_sess = args.up_to_sess, single_sess = args.single_sess, \
             n_ori_pyr = args.n_ori_pyr, n_sf_pyr = args.n_sf_pyr, \
             n_ori_gabor = args.n_ori_gabor, n_sf_gabor = args.n_sf_gabor, \
             gabor_nonlin_fn = args.gabor_nonlin_fn==1, \
             group_all_hl_feats = args.group_all_hl_feats, \
             sample_batch_size = args.sample_batch_size, voxel_batch_size = args.voxel_batch_size, \
             zscore_features = args.zscore_features==1, \
             ridge = args.ridge==1, \
             shuffle_images = args.shuffle_images==1, random_images = args.random_images==1, \
             random_voxel_data = args.random_voxel_data==1, \
             do_fitting = args.do_fitting==1, use_precomputed_prfs = args.use_precomputed_prfs==1, \
             do_val = args.do_val==1, do_stack = args.do_stack==1, \
             do_tuning = args.do_tuning==1, do_sem_disc = args.do_sem_disc==1, \
             do_varpart = args.do_varpart==1, do_roi_recons = args.do_roi_recons==1, \
             do_voxel_recons = args.do_voxel_recons==1, date_str = args.date_str, \
             shuff_rnd_seed = args.shuff_rnd_seed, debug = args.debug, \
             use_pca_pyr_feats_hl = args.use_pca_pyr_feats_hl==1, \
             use_pca_st_feats = args.use_pca_st_feats==1, \
             use_lda_st_feats = args.use_lda_st_feats==1, \
             lda_discrim_type = args.lda_discrim_type, \
             alexnet_layer_name = args.alexnet_layer_name, \
             alexnet_padding_mode = args.alexnet_padding_mode, \
             use_pca_alexnet_feats = args.use_pca_alexnet_feats==1, \
             clip_layer_name = args.clip_layer_name, \
             clip_model_architecture = args.clip_model_architecture, \
             use_pca_clip_feats = args.use_pca_clip_feats==1,\
             which_prf_grid = args.which_prf_grid, \
             save_pred_data = args.save_pred_data==1)
             
