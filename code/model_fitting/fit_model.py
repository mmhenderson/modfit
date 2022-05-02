"""
Run the model fitting for FWRF model. 
There are a few different versions of fitting in this script, the input arguments tell which kind of fitting to do.
"""

# import basic modules
import sys
import os
import time
import numpy as np
import argparse

# import custom modules
from feature_extraction import fwrf_features, semantic_features, merge_features
from utils import nsd_utils, roi_utils, default_paths

import initialize_fitting, arg_parser, fwrf_fit, fwrf_predict

try:
    device = initialize_fitting.init_cuda()
except:
    device = 'cpu:0'
    
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

print('numpy version: %s'%np.__version__)
#################################################################################################
        
    
def fit_fwrf(args):

    model_name, fitting_types = initialize_fitting.get_full_save_name(args)
    output_dir, fn2save = initialize_fitting.get_save_path(model_name, args)
    sys.stdout.flush()
    
    def save_all(fn2save):
    
        """
        Define all the important parameters that have to be saved
        """
        dict2save = {
        'subject': args.subject,
        'volume_space': args.volume_space,
        'fitting_types': fitting_types, 
        'voxel_mask': voxel_mask,
        'average_image_reps': args.average_image_reps, 
        'brain_nii_shape': brain_nii_shape,
        'image_order': image_order,
        'voxel_index': voxel_index,
        'voxel_roi': voxel_roi,
        'voxel_ncsnr': voxel_ncsnr, 
        'which_prf_grid': args.which_prf_grid,
        'models': prf_models,        
        'best_losses': best_losses,           
        'best_lambdas': best_lambdas,
        'best_params': best_params,       
        'lambdas': lambdas, 
        'val_cc': val_cc,
        'val_r2': val_r2,    
        'partial_masks': partial_masks, 
        'partial_version_names': partial_version_names,        
        'zscore_features': args.zscore_features, 
        'ridge': args.ridge,
        'debug': args.debug,
        'up_to_sess': args.up_to_sess,
        'single_sess': args.single_sess,
        'shuff_rnd_seed': args.shuff_rnd_seed,
        'use_precomputed_prfs': args.use_precomputed_prfs,
        'saved_prfs_fn': saved_prfs_fn,
        'best_layer_each_voxel': best_layer_each_voxel,
        'saved_best_layer_fn': saved_best_layer_fn,
        'voxel_subset_is_done_trn': voxel_subset_is_done_trn,
        'voxel_subset_is_done_val': voxel_subset_is_done_val,
        'trial_subset': args.trial_subset, 
        }
        # Might be some more things to save, depending what kind of fitting this is
        if args.compute_sessionwise_r2:
            dict2save.update({
            'val_r2_sess': val_r2_sess,
            'val_cc_sess': val_cc_sess,
            })
        if args.use_model_residuals:
            dict2save.update({
            'residuals_model_name': args.residuals_model_name, 
            'residuals_model_filename': residuals_model_filename,
            })
        if args.do_tuning:
            dict2save.update({
            'corr_each_feature': corr_each_feature
            })
        if args.do_sem_disc:
            dict2save.update({
            'sem_discrim_each_axis': sem_discrim_each_axis,
            'sem_corr_each_axis': sem_corr_each_axis,
            'discrim_type_list': discrim_type_list,
            'n_sem_samp_each_axis': n_sem_samp_each_axis,
            'mean_each_sem_level': mean_each_sem_level,
            'axes_to_do': axes_to_do, 
            'sem_partial_corrs': sem_partial_corrs, 
            'sem_partial_n_samp': sem_partial_n_samp, 
            'axes_to_balance': axes_to_balance,
            'sem_discrim_each_axis_balanced': sem_discrim_each_axis_balanced,
            'sem_corr_each_axis_balanced': sem_corr_each_axis_balanced,           
            'n_sem_samp_each_axis_balanced': n_sem_samp_each_axis_balanced,
            'mean_each_sem_level_balanced': mean_each_sem_level_balanced,
            })
        if np.any(['semantic' in ft for ft in fitting_types]):
            dict2save.update({
            'semantic_feature_set': args.semantic_feature_set,
            })
        if np.any(['sketch_tokens' in ft for ft in fitting_types]):
            dict2save.update({         
            'use_pca_st_feats': args.use_pca_st_feats,
            'use_residual_st_feats': args.use_residual_st_feats,
            })          
        if np.any(['pyramid' in ft for ft in fitting_types]):
            dict2save.update({
            'use_pca_pyr_feats_hl': args.use_pca_pyr_feats_hl,
            'pyramid_feature_info':pyramid_feature_info,
            'group_all_hl_feats': args.group_all_hl_feats,
            'do_pyr_varpart': args.do_pyr_varpart,
            })            
        if np.any(['gabor' in ft for ft in fitting_types]):
            dict2save.update({
            'n_ori_gabor': args.n_ori_gabor,
            'n_sf_gabor': args.n_sf_gabor,
            'gabor_nonlin_fn': args.gabor_nonlin_fn,
            'use_pca_gabor_feats': args.use_pca_gabor_feats,
            })
        if np.any(['alexnet' in ft for ft in fitting_types]):
            dict2save.update({
            'alexnet_layer_name': args.alexnet_layer_name,
            'alexnet_padding_mode': args.alexnet_padding_mode,
            'use_pca_alexnet_feats': args.use_pca_alexnet_feats, 
            })
        if np.any(['clip' in ft for ft in fitting_types]):
            dict2save.update({
            'clip_layer_name': args.clip_layer_name,
            'clip_model_architecture': args.clip_model_architecture,
            'use_pca_clip_feats': args.use_pca_clip_feats,   
            })

        print('\nSaving to %s\n'%fn2save)
        print(dict2save.keys())
        np.save(fn2save, dict2save, allow_pickle=True)

    if (args.from_scratch) and not (args.date_str==0 or args.date_str=='0' or args.date_str==''):
        raise ValueError('if --from_scratch=True, should specify --date_str=0 (rather than entering a date)')    
    if (args.do_sem_disc or args.do_tuning) and not args.do_val:
        raise ValueError('to do tuning analysis or semantic discriminability, need to run validation (--do_val=True)')       
      
    if args.use_model_residuals and len(args.residuals_model_name)==0:
        raise ValueError('must specify the name of model the residuals are from')
        
    val_r2 = None; 
    val_cc = None;
    
    val_r2_sess = None;
    val_cc_sess = None;
    
    corr_each_feature = None
    
    sem_discrim_each_axis = None
    sem_corr_each_axis = None
    discrim_type_list = None
    n_sem_samp_each_axis = None
    mean_each_sem_level = None
    
    axes_to_do = None
    sem_partial_corrs = None
    sem_partial_n_samp = None
    
    axes_to_balance = None
    sem_discrim_each_axis_balanced = None
    sem_corr_each_axis_balanced = None
    n_sem_samp_each_axis_balanced = None
    mean_each_sem_level_balanced = None
        
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
    if not args.use_model_residuals:
        # normal case, using voxel data to fit model
        voxel_data, image_order, val_inds, holdout_inds, session_inds = \
                                    nsd_utils.get_data_splits(args.subject, \
                                    sessions=sessions, \
                                    voxel_mask=voxel_mask, volume_space=args.volume_space, \
                                    zscore_betas_within_sess=True, \
                                    shuffle_images=args.shuffle_images,\
                                    average_image_reps=args.average_image_reps, \
                                    random_voxel_data=args.random_voxel_data)
        
    else:
        # special case, using the residuals of another encoding model as input voxel data.
        voxel_data, image_order, val_inds, holdout_inds, session_inds, residuals_model_filename = \
                                    initialize_fitting.load_model_residuals(args, sessions)
    voxel_data_val = voxel_data[val_inds,:]
    voxel_data_trn = voxel_data[~val_inds & ~holdout_inds,:]
    voxel_data_holdout = voxel_data[holdout_inds,:]
    image_inds_val = image_order[val_inds]
    image_inds_trn = image_order[~val_inds & ~holdout_inds]
    image_inds_holdout = image_order[holdout_inds]
    
    if session_inds is not None:
        session_inds_val = session_inds[val_inds]
        session_inds_trn = session_inds[~val_inds & ~holdout_inds]
        session_inds_holdout = session_inds[holdout_inds]
    n_voxels = voxel_data_trn.shape[1]   

    ########## DEFINE PARAMETERS #############################################################################
    
    lambdas = initialize_fitting.get_lambdas(fitting_types=fitting_types, \
                                             zscore_features=args.zscore_features, ridge=args.ridge)
    prf_models = initialize_fitting.get_prf_models(which_grid=args.which_prf_grid, verbose=True) 
    n_prfs = prf_models.shape[0]
    
    sys.stdout.flush()
    
    if (args.trial_subset!='all'):
        
        if not args.debug:
            assert(args.up_to_sess==40)
        assert(args.average_image_reps==True)
        
        trn_trials_use, holdout_trials_use, val_trials_use = \
                initialize_fitting.get_subsampled_trial_order(image_inds_trn, \
                                                              image_inds_holdout, \
                                                              image_inds_val, \
                                                              args=args, \
                                                              index=0)
        
        print('min trn trials: %d'%np.min(np.sum(trn_trials_use, axis=0)))
        print('min holdout trials: %d'%np.min(np.sum(holdout_trials_use, axis=0)))
        print('min val trials: %d'%np.min(np.sum(val_trials_use, axis=0)))
        
    else:
        trn_trials_use = None
        holdout_trials_use = None
        val_trials_use = None
   
    ########## LOAD PRECOMPUTED PRFS ##########################################################################
        
    if args.use_precomputed_prfs:
        # If we already computed pRFs for this subject on some model, can load those now and use them during 
        # fitting. Faster than fitting pRFs each time.
        best_model_each_voxel, saved_prfs_fn = initialize_fitting.load_precomputed_prfs(args.subject, args)
        assert(len(best_model_each_voxel)==n_voxels)
    else:
        # otherwise fitting all params from scratch.
        best_model_each_voxel = None
        saved_prfs_fn = None

    ########### LOOPING OVER VOXEL SUBSETS ############################################################
    # used for clip/alexnet when layer_name is "best_layer", diff voxels get fit w different features
    # otherwise this loop only goes once and voxel_subset_mask is all ones.
    
    # define voxel subsets (if using)      
    if dnn_model is not None and (args.alexnet_layer_name=='best_layer' or args.clip_layer_name=='best_layer'):
        # special case, going to fit groups of voxels separately according to which dnn layer was best
        # creating a list of voxel masks here that will define the subsets to loop over.
        best_layer_each_voxel, saved_best_layer_fn = \
                  initialize_fitting.load_best_model_layers(args.subject, dnn_model)
        voxel_subset_masks = [best_layer_each_voxel==ll for ll in range(n_dnn_layers)]
        assert(len(best_layer_each_voxel)==n_voxels)
        assert(not args.save_model_residuals)
    else:
        # going to fit all voxels w same model
        voxel_subset_masks = [np.ones((n_voxels,), dtype=bool)]
        best_layer_each_voxel = None;
        saved_best_layer_fn = None;
        
    if not args.from_scratch:
        
        # stuff that needs to happen if we are resuming from some intermediate point
        print('\nLoading the results of training from %s\n'%fn2save)
        last_saved = np.load(fn2save, allow_pickle=True).item()
        # make sure that training was actually done, otherwise should start over 
        assert(np.any(last_saved['voxel_subset_is_done_trn']))
        assert(last_saved['up_to_sess']==args.up_to_sess)
        assert(last_saved['debug']==args.debug)
        assert(last_saved['which_prf_grid']==args.which_prf_grid)
        assert(np.all(last_saved['lambdas']==lambdas))
        assert(last_saved['saved_prfs_fn']==saved_prfs_fn)
        assert(last_saved['saved_best_layer_fn']==saved_best_layer_fn)

        voxel_subset_is_done_trn = last_saved['voxel_subset_is_done_trn']
        voxel_subset_is_done_val = last_saved['voxel_subset_is_done_val']
        
        if args.do_tuning:
            if 'corr_each_feature' in last_saved.keys() and last_saved['corr_each_feature'] is not None:
                corr_each_feature = last_saved['corr_each_feature']
            else:
                voxel_subset_is_done_val = np.zeros(np.shape(voxel_subset_is_done_val), dtype=bool)
        else:
            assert('corr_each_feature' not in last_saved.keys() or last_saved['corr_each_feature'] is None)
            
        if args.do_sem_disc:
            if (not args.overwrite_sem_disc) \
               and 'sem_discrim_each_axis' in last_saved.keys() \
               and last_saved['sem_discrim_each_axis'] is not None:
                sem_discrim_each_axis = last_saved['sem_discrim_each_axis']
                sem_corr_each_axis = last_saved['sem_corr_each_axis']
                discrim_type_list = last_saved['discrim_type_list']
                n_sem_samp_each_axis = last_saved['n_sem_samp_each_axis']
                mean_each_sem_level = last_saved['mean_each_sem_level']
                
                axes_to_do = last_saved['axes_to_do']
                sem_partial_corrs = last_saved['sem_partial_corrs']
                sem_partial_n_samp = last_saved['sem_partial_n_samp']
                
                axes_to_balance = last_saved['axes_to_balance']
                sem_discrim_each_axis_balanced = last_saved['sem_discrim_each_axis_balanced']
                sem_corr_each_axis_balanced = last_saved['sem_corr_each_axis_balanced']               
                n_sem_samp_each_axis_balanced = last_saved['n_sem_samp_each_axis_balanced']
                mean_each_sem_level_balanced = last_saved['mean_each_sem_level_balanced']
            else:
                voxel_subset_is_done_val = np.zeros(np.shape(voxel_subset_is_done_val), dtype=bool)
        else:
            assert('sem_discrim_each_axis' not in last_saved.keys() or last_saved['sem_discrim_each_axis'] is None)
                

        print('training done for subsets:')
        print(voxel_subset_is_done_trn)
        print('validation done for subsets:')
        print(voxel_subset_is_done_val)
        best_losses = last_saved['best_losses']
        best_lambdas = last_saved['best_lambdas']
        best_params = last_saved['best_params'] 
        best_prf_model_pars, best_weights, best_biases, \
                           features_mean, features_std, best_prf_models = best_params
        val_cc = last_saved['val_cc']
        val_r2 = last_saved['val_r2']
      
    else:
        voxel_subset_is_done_trn = np.zeros((len(voxel_subset_masks),),dtype=bool)
        voxel_subset_is_done_val = np.zeros((len(voxel_subset_masks),),dtype=bool)
        
    # Start the loop
    for vi, voxel_subset_mask in enumerate(voxel_subset_masks):
        
        voxel_data_trn_use = voxel_data_trn[:,voxel_subset_mask]
        voxel_data_holdout_use = voxel_data_holdout[:,voxel_subset_mask]
        voxel_data_val_use = voxel_data_val[:,voxel_subset_mask]
        if best_model_each_voxel is not None:
            best_model_each_voxel_use = best_model_each_voxel[voxel_subset_mask]
        else:
            best_model_each_voxel_use = None
        print('\nStarting fitting for voxel mask %d of %d, number of voxels this loop=%d'%(vi, \
                                           len(voxel_subset_masks), voxel_data_trn_use.shape[1]))
        if voxel_data_trn_use.shape[1]==0:
            print('no voxels, continuing loop')
            voxel_subset_is_done_trn = True
            continue
        
        ########## CREATE FEATURE LOADERS ###################################################################
        # these help to load sets of pre-computed features in an organized way.
        # first making a list of all the modules of interest (different feature spaces)
        fe = []
        fe_names = []
        for ft in fitting_types:   

            if 'gabor_solo' in ft:
                feat_loader = fwrf_features.fwrf_feature_loader(subject=args.subject,\
                                                                which_prf_grid=args.which_prf_grid,\
                                                                feature_type='gabor_solo',\
                                                                n_ori=args.n_ori_gabor, n_sf=args.n_sf_gabor,\
                                                                nonlin_fn=args.gabor_nonlin_fn, \
                                                                use_pca_feats=args.use_pca_gabor_feats)
        
                fe.append(feat_loader)
                fe_names.append(ft)
            elif 'pyramid' in ft:
                feat_loader = fwrf_features.fwrf_feature_loader(subject=args.subject,\
                                                                which_prf_grid=args.which_prf_grid, \
                                                                feature_type='pyramid_texture',\
                                                                n_ori=args.n_ori_pyr, n_sf=args.n_sf_pyr,\
                                                                include_ll=True, include_hl=True,\
                                                                use_pca_feats_hl = args.use_pca_pyr_feats_hl,\
                                                                do_varpart=args.do_pyr_varpart,\
                                                                group_all_hl_feats=args.group_all_hl_feats)       
                fe.append(feat_loader)
                fe_names.append(ft)
                pyramid_feature_info = [feat_loader.feature_column_labels, feat_loader.feature_types_include]

            elif 'sketch_tokens' in ft:
                feat_loader = fwrf_features.fwrf_feature_loader(subject=args.subject,\
                                                                which_prf_grid=args.which_prf_grid, \
                                                                feature_type='sketch_tokens',\
                                                                use_pca_feats = args.use_pca_st_feats, \
                                                                use_residual_st_feats = args.use_residual_st_feats)
                fe.append(feat_loader)
                fe_names.append(ft)
          
            elif 'alexnet' in ft:
                if args.alexnet_layer_name=='all_conv':
                    names = ['Conv%d_ReLU'%(ll+1) for ll in range(n_dnn_layers)]
                    for ll in range(n_dnn_layers):
                        feat_loader = fwrf_features.fwrf_feature_loader(subject=args.subject,\
                                                                which_prf_grid=args.which_prf_grid, \
                                                                feature_type='alexnet',layer_name=names[ll],\
                                                                use_pca_feats = args.use_pca_alexnet_feats,\
                                                                padding_mode = args.alexnet_padding_mode)
                        fe.append(feat_loader)   
                        fe_names.append('alexnet_%s'%names[ll])
                elif args.alexnet_layer_name=='best_layer':
                    this_layer_name = 'Conv%d_ReLU'%(vi+1)
                    print(this_layer_name)
                    feat_loader = fwrf_features.fwrf_feature_loader(subject=args.subject,\
                                                                which_prf_grid=args.which_prf_grid, \
                                                                feature_type='alexnet',layer_name=this_layer_name,\
                                                                use_pca_feats = args.use_pca_alexnet_feats,\
                                                                padding_mode = args.alexnet_padding_mode)
                    fe.append(feat_loader)   
                    fe_names.append(ft)
                else:
                    feat_loader = fwrf_features.fwrf_feature_loader(subject=args.subject,\
                                                                which_prf_grid=args.which_prf_grid, \
                                                                feature_type='alexnet',layer_name=args.alexnet_layer_name,\
                                                                use_pca_feats = args.use_pca_alexnet_feats,\
                                                                padding_mode = args.alexnet_padding_mode)
                    fe.append(feat_loader)
                    fe_names.append(ft)
          
            elif 'clip' in ft:
                if args.clip_layer_name=='all_resblocks':
                    names = ['block%d'%(ll) for ll in range(n_dnn_layers)]
                    for ll in range(n_dnn_layers):
                        feat_loader = fwrf_features.fwrf_feature_loader(subject=args.subject,\
                                                                which_prf_grid=args.which_prf_grid, \
                                                                feature_type='clip',layer_name=names[ll],\
                                                                model_architecture=args.clip_model_architecture,\
                                                                use_pca_feats=args.use_pca_clip_feats)
                        fe.append(feat_loader)   
                        fe_names.append('clip_%s'%names[ll])
                elif args.clip_layer_name=='best_layer':
                    this_layer_name = 'block%d'%(vi)
                    print(this_layer_name)
                    feat_loader = fwrf_features.fwrf_feature_loader(subject=args.subject,\
                                                                which_prf_grid=args.which_prf_grid, \
                                                                feature_type='clip',layer_name=this_layer_name,\
                                                                model_architecture=args.clip_model_architecture,\
                                                                use_pca_feats=args.use_pca_clip_feats)
                    fe.append(feat_loader)
                    fe_names.append(ft) 
                else:
                    feat_loader = fwrf_features.fwrf_feature_loader(subject=args.subject,\
                                                                which_prf_grid=args.which_prf_grid, \
                                                                feature_type='clip',layer_name=args.clip_layer_name,\
                                                                model_architecture=args.clip_model_architecture,\
                                                                use_pca_feats=args.use_pca_clip_feats)
                    fe.append(feat_loader)
                    fe_names.append(ft)   
          
            elif 'semantic' in ft:
                this_feature_set = ft.split('semantic_')[1]
                if '_pca' in this_feature_set:
                    this_feature_set = this_feature_set.split('_pca')[0]
                    use_pca_feats=True
                else:
                    use_pca_feats=False
                print('semantic feature set: %s'%this_feature_set)
                print('use pca: %s'%use_pca_feats)
                feat_loader = semantic_features.semantic_feature_loader(subject=args.subject,\
                                                                which_prf_grid=args.which_prf_grid, \
                                                                feature_set=this_feature_set, \
                                                                use_pca_feats=use_pca_feats, \
                                                                remove_missing=False)
                fe.append(feat_loader)
                fe_names.append(ft)
          
        # Now combine subsets of features into a single module
        if len(fe)>1:
            feat_loader_full = merge_features.combined_feature_loader(fe, fe_names, do_varpart = args.do_varpart, \
                                                                     include_solo_models=args.include_solo_models)
        else:
            feat_loader_full = fe[0]
        max_features = feat_loader_full.max_features 
        
        # getting info about how variance partition will be set up
        partial_masks_tmp, partial_version_names = feat_loader_full.get_partial_versions()      
        print(partial_version_names)
        print(np.sum(partial_masks_tmp, axis=1))
        if vi==0:
            n_partial_versions = len(partial_version_names)
            partial_masks = [[] for ii in range(len(voxel_subset_masks))] 
        partial_masks[vi] = partial_masks_tmp
        assert(len(partial_version_names)==n_partial_versions)
        
        if (vi==0) and args.from_scratch:
            # preallocate some arrays for params over all voxels
            best_losses = np.zeros((n_voxels, n_partial_versions), dtype=np.float32)
            best_lambdas = np.zeros((n_voxels, n_partial_versions), dtype=int)
            best_weights = np.zeros((n_voxels, max_features, n_partial_versions), dtype=np.float32)
            best_biases = np.zeros((n_voxels, n_partial_versions), dtype=np.float32)
            best_prf_models = np.zeros((n_voxels, n_partial_versions), dtype=int)
            features_mean = np.zeros((n_prfs, max_features,len(voxel_subset_masks)), dtype=np.float32)
            features_std = np.zeros((n_prfs, max_features,len(voxel_subset_masks)), dtype=np.float32)
            val_cc = np.zeros((n_voxels, n_partial_versions), dtype=np.float32)
            val_r2 = np.zeros((n_voxels, n_partial_versions), dtype=np.float32) 
            val_cc_sess = np.zeros((n_voxels, n_partial_versions, 40), dtype=np.float32)
            val_r2_sess = np.zeros((n_voxels, n_partial_versions, 40), dtype=np.float32) 
            
        # if this current feature set has more features than the first one, 
        # might need to pad the weights array to make it fit.                 
        if best_weights.shape[1]<max_features:
            n2pad = max_features - best_weights.shape[1]
            print('padding by %d elements'%n2pad)
            print(np.shape(best_weights))
            print(np.shape(features_mean))
            print(np.shape(features_std))
            best_weights = np.pad(best_weights, [[0,0], [0, n2pad], [0,0]])            
            features_mean = np.pad(features_mean, [[0,0], [0, n2pad], [0,0]])
            features_std = np.pad(features_std, [[0,0], [0, n2pad], [0,0]])
            print(np.shape(best_weights))
            print(np.shape(features_mean))
            print(np.shape(features_std))
            
        #### FIT ENCODING MODEL ###################################################################################

        if not voxel_subset_is_done_trn[vi]:
   
            print('\nStarting training (voxel subset %d of %d)...\n'%(vi, len(voxel_subset_masks)))
            print(len(image_inds_trn))

            best_losses_tmp, best_lambdas_tmp, best_weights_tmp, best_biases_tmp, \
                best_prf_models_tmp, features_mean_tmp, features_std_tmp, \
                                = fwrf_fit.fit_fwrf_model(image_inds_trn, voxel_data_trn_use, \
                                                          image_inds_holdout, voxel_data_holdout_use, \
                                                        feat_loader_full, prf_models, lambdas, \
                                                        best_model_each_voxel = best_model_each_voxel_use, \
                                                        zscore=args.zscore_features, \
                                                        add_bias=True, \
                                                        voxel_batch_size=args.voxel_batch_size, \
                                                        device=device, \
                                                        trials_use_each_prf_trn = trn_trials_use, \
                                                        trials_use_each_prf_holdout = holdout_trials_use, \
                                                        dtype=np.float32, debug=args.debug)
            
            # taking the fit params for this set of voxels and putting them into the full array over all voxels
            best_losses[voxel_subset_mask,:] = best_losses_tmp
            best_lambdas[voxel_subset_mask,:] = best_lambdas_tmp            
            best_weights[voxel_subset_mask,0:max_features,:] = best_weights_tmp
            features_mean[:,0:max_features,vi] = features_mean_tmp
            features_std[:,0:max_features,vi] = features_std_tmp
            best_biases[voxel_subset_mask,:] = best_biases_tmp
            best_prf_models[voxel_subset_mask,:] = best_prf_models_tmp
            
            voxel_subset_is_done_trn[vi] = True
   
        else:
            best_losses_tmp = best_losses[voxel_subset_mask,:]
            best_lambdas_tmp = best_lambdas[voxel_subset_mask,:]
            best_weights_tmp = best_weights[voxel_subset_mask,0:max_features,:]
            best_biases_tmp = best_biases[voxel_subset_mask,:]
            best_prf_models_tmp = best_prf_models[voxel_subset_mask,:]
            features_mean_tmp = features_mean[:,0:max_features,vi]
            features_std_tmp = features_std[:,0:max_features,vi]
            
        # "best_params_tmp" will be passed to validation functions (just these voxels)
        # "best_params" will be saved (all voxels)
        best_params_tmp = [prf_models[best_prf_models_tmp,:], best_weights_tmp, best_biases_tmp, \
                           features_mean_tmp, features_std_tmp, best_prf_models_tmp]
        best_params = [prf_models[best_prf_models,:], best_weights, best_biases, \
                           features_mean, features_std, best_prf_models]
        sys.stdout.flush()
        save_all(fn2save)   
            
        ######### VALIDATE MODEL ON HELD-OUT TEST SET ##############################################
        sys.stdout.flush()
        if args.do_val and not voxel_subset_is_done_val[vi]: 
            
            print('Starting validation (voxel subset %d of %d)...\n'%(vi, len(voxel_subset_masks)))
            sys.stdout.flush()
    
            val_cc_tmp, val_r2_tmp, voxel_data_val_pred, features_each_prf = \
                fwrf_predict.validate_fwrf_model(best_params_tmp, prf_models, \
                                                 voxel_data_val_use, image_inds_val, \
                                                 feat_loader_full, zscore=args.zscore_features, \
                                                 sample_batch_size=args.sample_batch_size, \
                                                 voxel_batch_size=args.voxel_batch_size, \
                                                 debug=args.debug, \
                                                 trials_use_each_prf = val_trials_use, \
                                                 dtype=np.float32, device=device)
                     
            val_cc[voxel_subset_mask,:] = val_cc_tmp
            val_r2[voxel_subset_mask,:] = val_r2_tmp
            if (not args.do_tuning) and (not args.do_sem_disc):
                voxel_subset_is_done_val[vi] = True
            save_all(fn2save) 
            
            if args.compute_sessionwise_r2:
                for se in np.arange(40):
                    trials_this_sess = (session_inds_val==se)                   
                    if np.sum(trials_this_sess)==0:
                        val_cc_sess[voxel_subset_mask,:,se] = np.nan
                        val_r2_sess[voxel_subset_mask,:,se] = np.nan
                    else:
                        print('computing validation accuracy for session %d, %d trials'%(se, np.sum(trials_this_sess)))
                        val_trials_use_this_sess = np.tile(trials_this_sess[:,np.newaxis], [1,n_prfs])
                        val_cc_sess_tmp, val_r2_sess_tmp, _, _ = \
                            fwrf_predict.validate_fwrf_model(best_params_tmp, prf_models, \
                                                             voxel_data_val_use, image_inds_val, \
                                                             feat_loader_full, zscore=args.zscore_features, \
                                                             sample_batch_size=args.sample_batch_size, \
                                                             voxel_batch_size=args.voxel_batch_size, \
                                                             debug=args.debug, \
                                                             trials_use_each_prf = val_trials_use_this_sess, \
                                                             dtype=np.float32, device=device)

                        val_cc_sess[voxel_subset_mask,:,se] = val_cc_sess_tmp
                        val_r2_sess[voxel_subset_mask,:,se] = val_r2_sess_tmp
                if (not args.do_tuning) and (not args.do_sem_disc):
                    voxel_subset_is_done_val[vi] = True
                save_all(fn2save) 

            
            ### ESTIMATE VOXELS' FEATURE TUNING #####################################################################
            sys.stdout.flush()
            if args.do_tuning:

                print('\nStarting feature tuning analysis (voxel subset %d of %d)...\n'%(vi, len(voxel_subset_masks)))
                sys.stdout.flush()
                corr_each_feature_tmp = fwrf_predict.get_feature_tuning(best_params_tmp, features_each_prf, \
                                                                        voxel_data_val_pred, \
                                                                        trials_use_each_prf = val_trials_use, \
                                                                        debug=args.debug)
                if vi==0:
                    corr_each_feature = np.zeros((n_voxels, corr_each_feature_tmp.shape[1]), dtype=corr_each_feature_tmp.dtype)  
                max_features = feat_loader_full.max_features
                if corr_each_feature.shape[1]<max_features:
                    n2pad = max_features - corr_each_feature.shape[1]
                    print('padding by %d elements'%n2pad)
                    print(np.shape(corr_each_feature))
                    corr_each_feature = np.pad(corr_each_feature, [[0,0], [0, n2pad]])
                    print(np.shape(corr_each_feature))
                corr_each_feature[voxel_subset_mask,0:max_features] = corr_each_feature_tmp                
                if not args.do_sem_disc:
                    voxel_subset_is_done_val[vi] = True
                save_all(fn2save)

            ### ESTIMATE SEMANTIC DISCRIMINABILITY #########################################################################
            sys.stdout.flush()
            if args.do_sem_disc:
                
                print('\nStarting semantic discriminability analysis (voxel subset %d of %d)...\n'%(vi, len(voxel_subset_masks)))
                sys.stdout.flush()
                labels_all, discrim_type_list, unique_labs_each = \
                        initialize_fitting.load_labels_each_prf(args.subject, args.which_prf_grid,\
                                                        image_inds=image_inds_val, \
                                                        models=prf_models,verbose=False, \
                                                        debug=args.debug)
                discrim_tmp, corr_tmp, n_samp_tmp, mean_tmp = \
                        fwrf_predict.get_semantic_discrim(best_params_tmp, \
                                                          labels_all, unique_labs_each, \
                                                          voxel_data_val_pred,\
                                                          trials_use_each_prf = val_trials_use, \
                                                          debug=args.debug)
                if vi==0:
                    sem_discrim_each_axis = np.zeros((n_voxels, discrim_tmp.shape[1]), \
                                                     dtype=discrim_tmp.dtype) 
                    sem_corr_each_axis = np.zeros((n_voxels, corr_tmp.shape[1]), \
                                                     dtype=corr_tmp.dtype)
                    n_sem_samp_each_axis = np.zeros((n_voxels, n_samp_tmp.shape[1], n_samp_tmp.shape[2]), \
                                                     dtype=n_samp_tmp.dtype)
                    mean_each_sem_level = np.zeros((n_voxels, mean_tmp.shape[1], mean_tmp.shape[2]), \
                                                     dtype=mean_tmp.dtype)
                sem_discrim_each_axis[voxel_subset_mask,:] = discrim_tmp
                sem_corr_each_axis[voxel_subset_mask,:] = corr_tmp
                n_sem_samp_each_axis[voxel_subset_mask,:,:] = n_samp_tmp
                mean_each_sem_level[voxel_subset_mask,:,:] = mean_tmp
#                 voxel_subset_is_done_val[vi] = True
                save_all(fn2save)
    
                # compute partial correlations for some axes 
                axes_to_do = [0,2,3]
                print('\nGoing to compute partial correlations, for these pairs of axes:')
                print([discrim_type_list[aa] for aa in axes_to_do])
                partial_corr_tmp, n_samp_tmp = \
                        fwrf_predict.get_semantic_partial_corrs(best_params_tmp, \
                                                          labels_all, axes_to_do=axes_to_do, \
                                                          unique_labels_each=unique_labs_each, \
                                                          val_voxel_data_pred=voxel_data_val_pred,\
                                                          trials_use_each_prf = val_trials_use, \
                                                          debug=args.debug)
                if vi==0:                 
                    sem_partial_corrs = np.zeros((n_voxels, partial_corr_tmp.shape[1]), \
                                                     dtype=partial_corr_tmp.dtype)
                    sem_partial_n_samp = np.zeros((n_voxels, n_samp_tmp.shape[1], n_samp_tmp.shape[2]), \
                                                     dtype=n_samp_tmp.dtype)

                sem_partial_corrs[voxel_subset_mask,:] = partial_corr_tmp
                sem_partial_n_samp[voxel_subset_mask,:,:] = n_samp_tmp
        
                voxel_subset_is_done_val[vi] = True
                save_all(fn2save)
                
                
                
        if args.save_model_residuals:
            # going to compute model predictions on entire data set, and save them as a separate
            # file for use later on.
            _, all_dat_r2, all_dat_pred, _ = \
            fwrf_predict.validate_fwrf_model(best_params_tmp, prf_models, \
                                             voxel_data, image_order, \
                                             feat_loader_full, zscore=args.zscore_features, \
                                             sample_batch_size=args.sample_batch_size, \
                                             voxel_batch_size=args.voxel_batch_size, \
                                             debug=args.debug, \
                                             trials_use_each_prf = None, \
                                             dtype=np.float32, device=device)

            initialize_fitting.save_model_residuals(voxel_data, all_dat_pred[:,:,0], \
                                                    output_dir, model_name, \
                                                    image_order, val_inds, \
                                                    session_inds, all_dat_r2[:,0], \
                                                    args)
            
                 
        # Done!

if __name__ == '__main__':
    
    args = arg_parser.get_args()
    fit_fwrf(args)
