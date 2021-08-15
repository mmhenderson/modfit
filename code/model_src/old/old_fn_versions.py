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




# OLD version

def get_bbox_from_prf(prf_params, image_size, n_prf_sd_out=2, verbose=False):
    """
    For a given pRF center and size, calculate the square bounding box that captures a specified number of SDs from the center (default=2 SD)
    Returns [xmin, xmax, ymin, ymax]
    """
    x,y,sigma = prf_params
    n_pix = image_size[0]
    assert(image_size[1]==n_pix)
    assert(sigma>0 and n_prf_sd_out>0)
    
    # decide on the window to use for correlations, based on prf parameters. Patch goes # SD from the center (2 by default).
    pix_from_center = int(sigma*n_prf_sd_out*n_pix)
    # center goes [row ind, col ind]
    center = np.array((int(np.floor(n_pix/2  - y*n_pix)), int(np.floor(x*n_pix + n_pix/2)))) # note that the x/y dims get swapped here because of how pRF parameters are defined.
    # ensure that the patch never tries to go outside image bounds...at the corners it initially will be outside. 
    # TODO: figure out what to do with these edge cases, because padding will probably introduce artifacts. 
    # For now just reducing the size of the region so that it's a square with all corners inside image bounds. 
    center = np.minimum(np.maximum(center,0), n_pix-1) 
   
    mindist2edge=np.min([n_pix-center[0]-1, n_pix-center[1]-1, center[0], center[1]])  
    pix_from_center = np.minimum(pix_from_center, mindist2edge)    
    if pix_from_center==0 and verbose:
        print('Warning: your patch only has one pixel (for n_pix: %d and prf params: [%.2f, %.2f, %.2f])\n'%(n_pix,x,y,sigma))
        
    assert(not np.any(np.array(center)<pix_from_center) and not np.any(image_size-np.array(center)<=pix_from_center))
    xmin = center[0]-pix_from_center
    xmax = center[0]+pix_from_center+1
    ymin = center[1]-pix_from_center
    ymax = center[1]+pix_from_center+1
        
    return [xmin, xmax, ymin, ymax]
 
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
    holdout_size, lambdas = initialize_fitting.get_fitting_pars(trn_voxel_data, zscore_features=True, ridge=ridge)    
#     holdout_size, lambdas = initialize_fitting.get_fitting_pars(trn_voxel_data, zscore_features, ridge=ridge)    

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
    print('\nvalidating model')
    gc.collect()
    torch.cuda.empty_cache()
    val_cc, val_r2 = fwrf_predict.validate_model(best_params, val_voxel_single_trial_data, val_stim_single_trial_data, _fmaps_fn, sample_batch_size, voxel_batch_size, aperture, dtype=fpX, combs_zstats = combs_zstats)
    print('\ndone validating')
    # As a less model-sensitive way of assessing tuning, directly measure each voxel's correlation with each feature channel.
    # Using validation set data. 
    print('\ngetting voxel/feat corrs')
    gc.collect()
    torch.cuda.empty_cache()
    features_each_model_val, voxel_feature_correlations_val, ignore1, ignore2 =  fwrf_predict.get_voxel_feature_corrs(best_params, models, _fmaps_fn, val_voxel_single_trial_data, 
                                                                                                    val_stim_single_trial_data, sample_batch_size, aperture, device, debug=False, dtype=fpX, combs_zstats = combs_zstats)
    print('\ndone getting voxel/feat corrs')
    ### SAVE THE RESULTS TO DISK #########
    print('\nabout to save')
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
    }, fn2save, pickle_protocol=4)
    print('\ndone saving')
 
    
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
    }, fn2save, pickle_protocol=4)

           


     
def learn_params_combinations_ridge_regression(images, voxel_data, _fmaps_fn, models, lambdas, aperture=1.0, zscore=False, sample_batch_size=100, voxel_batch_size=100, holdout_size=100, shuffle=True, add_bias=False, debug=False):
    
    """ 
    Fit model that includes second order "combinations" consisting of multiplying features from gabor model.
    """
    
    dtype = images.dtype.type
    device = next(_fmaps_fn.parameters()).device
    trn_size = len(voxel_data) - holdout_size
    assert trn_size>0, 'Training size needs to be greater than zero'
    
    print ('trn_size = %d (%.1f%%)' % (trn_size, float(trn_size)*100/len(voxel_data)))
    print ('dtype = %s' % dtype)
    print ('device = %s' % device)
    print ('---------------------------------------')
    
    # First do shuffling of data and define set to hold out
    n_trials = len(images)
    n_prfs = len(models)
    n_voxels = voxel_data.shape[1]
    order = np.arange(len(voxel_data), dtype=int)
    if shuffle:
        np.random.shuffle(order)
    images = images[order]
    voxel_data = voxel_data[order]  
    trn_data = voxel_data[:trn_size]
    out_data = voxel_data[trn_size:]
    
    # Looping over the feature maps once with a batch of images, to get their sizes
    n_features, fmaps_rez = get_fmaps_sizes(_fmaps_fn, images[0:sample_batch_size], device)
    n_features_full = n_features*n_features+n_features # this is total dim of feature matrix once we add in second order combinations.
    
    # Create full model value buffers    
    best_models = np.full(shape=(n_voxels,), fill_value=-1, dtype=int)   
    best_lambdas = np.full(shape=(n_voxels,), fill_value=-1, dtype=int)
    best_losses = np.full(fill_value=np.inf, shape=(n_voxels), dtype=dtype)
    best_w_params = np.zeros(shape=(n_voxels, n_features_full ), dtype=dtype)

    if add_bias:
        best_w_params = np.concatenate([best_w_params, np.ones(shape=(len(best_w_params),1), dtype=dtype)], axis=1)
    features_mean = None
    features_std = None
    if zscore:
        features_mean = np.zeros(shape=(n_voxels, n_features_full), dtype=dtype)
        features_std  = np.zeros(shape=(n_voxels, n_features_full), dtype=dtype)
    
    combs_zstats = np.zeros(shape=(n_prfs,4), dtype=dtype)
        
    # going to save the covariance matrices too, see how correlated features are in training data.
    covar_each_model = np.zeros(shape=(n_features_full, n_features_full, n_prfs), dtype=dtype)
    
    start_time = time.time()
    vox_loop_time = 0
    print ('')
    
    with torch.no_grad():
        
        # Looping over models (here models are different spatial RF definitions)
        for m,(x,y,sigma) in enumerate(models):
            if debug and m>1:
                break

            t = time.time()            
            # Get features for the desired pRF, across all trn set image            
            features, zs = get_features_in_prf_combinations((x,y,sigma), _fmaps_fn, images, sample_batch_size, aperture, device)     
            combs_zstats[m,:] = zs     
            elapsed = time.time() - t
        
            # Calculate covariance of raw feature activations, just to look at later
            covar_each_model[:,:,m] = np.cov(np.transpose(features))                        

            if zscore:  
                features_m = np.mean(features, axis=0, keepdims=True) #[:trn_size]
                features_s = np.std(features, axis=0, keepdims=True) + 1e-6          
                features -= features_m
                features /= features_s    
                
            if add_bias:
                features = np.concatenate([features, np.ones(shape=(len(features), 1), dtype=dtype)], axis=1)
            
            # separate design matrix into training/held out data (for lambda selection)
            trn_features = features[:trn_size]
            out_features = features[trn_size:]   

            
            # Send matrices to gpu
            _xtrn = torch_utils._to_torch(trn_features, device=device)
            _xout = torch_utils._to_torch(out_features, device=device)   
            
            # Do part of the matrix math involved in ridge regression optimization out of the loop, 
            # because this part will be same for all the voxels.
#             _cof = _cofactor_fn(_xtrn, lambdas, device=device)
            _cof = _cofactor_fn_cpu(_xtrn, lambdas)
            
            # Now looping over batches of voxels (only reason is because can't store all in memory at same time)
            vox_start = time.time()
            for rv,lv in numpy_utility.iterate_range(0, n_voxels, voxel_batch_size):
                sys.stdout.write('\rfitting model %4d of %-4d, voxels [%6d:%-6d] of %d' % (m, n_prfs, rv[0], rv[-1], n_voxels))

                # Send matrices to gpu
                _vtrn = torch_utils._to_torch(trn_data[:,rv], device=device)
                _vout = torch_utils._to_torch(out_data[:,rv], device=device)

                # Here is where optimization happens - relatively simple matrix math inside loss fn.
                _betas, _loss = _loss_fn(_cof, _vtrn, _xout, _vout) #   [#lambda, #feature, #voxel, ], [#lambda, #voxel]
                # Now have a set of weights (in betas) and a loss value for every voxel and every lambda. 
                # goal is then to choose for each voxel, what is the best lambda and what weights went with that lambda.

                # first choose best lambda value and the loss that went with it.
                _values, _select = torch.min(_loss, dim=0)
                betas = torch_utils.get_value(_betas)
                values, select = torch_utils.get_value(_values), torch_utils.get_value(_select)

                # comparing this loss to the other models for each voxel (e.g. the other RF position/sizes)
                imp = values<best_losses[rv]

                if np.sum(imp)>0:
                    # for whichever voxels had improvement relative to previous models, save parameters now
                    # this means we won't have to save all params for all models, just best.
                    arv = np.array(rv)[imp]
                    li = select[imp]
                    best_lambdas[arv] = li
                    best_losses[arv] = values[imp]
                    best_models[arv] = m
                    if zscore:
                        features_mean[arv] = features_m # broadcast over updated voxels
                        features_std[arv]  = features_s
                    # taking the weights associated with the best lambda value
                    best_w_params[arv,:] = numpy_utility.select_along_axis(betas[:,:,imp], li, run_axis=2, choice_axis=0).T
                    
            vox_loop_time += (time.time() - vox_start)
            elapsed = (time.time() - vox_start)

    # Print information about how fitting went...
    total_time = time.time() - start_time
    inv_time = total_time - vox_loop_time
    return_params = [best_w_params[:,:n_features_full],]
    if add_bias:
        return_params += [best_w_params[:,-1],]
    else: 
        return_params += [None,]
    print ('\n---------------------------------------')
    print ('total time = %fs' % total_time)
    print ('total throughput = %fs/voxel' % (total_time / n_voxels))
    print ('voxel throughput = %fs/voxel' % (vox_loop_time / n_voxels))
    print ('setup throughput = %fs/model' % (inv_time / n_prfs))
    sys.stdout.flush()
    return best_losses, best_lambdas, [models[best_models],]+return_params+[features_mean, features_std]+[best_models], covar_each_model, combs_zstats


def learn_params_combinations_pca(images, voxel_data, _fmaps_fn, models, min_pct_var = 99, aperture=1.0, zscore=False, sample_batch_size=100, voxel_batch_size=100, holdout_size=100, shuffle=True, add_bias=False, debug=False):
    """
    
    Learn the parameters of the model, including "combinations" of first order features.
    Using PCA before fitting to decorrelate features.
    
    """
    
    pca = decomposition.PCA()

    dtype = images.dtype.type
    device = next(_fmaps_fn.parameters()).device
    trn_size = len(voxel_data) - holdout_size
    assert trn_size>0, 'Training size needs to be greater than zero'
    
    print ('trn_size = %d (%.1f%%)' % (trn_size, float(trn_size)*100/len(voxel_data)))
    print ('dtype = %s' % dtype)
    print ('device = %s' % device)
    print ('---------------------------------------')
       
    n_trials = len(images)
    n_prfs = len(models)
    n_voxels = voxel_data.shape[1]   
    order = np.arange(len(voxel_data), dtype=int)
    if shuffle:
        np.random.shuffle(order)
    images = images[order]
    voxel_data = voxel_data[order]  
    trn_data = voxel_data[:trn_size]
    out_data = voxel_data[trn_size:]
    
    # Looping over the feature maps once with a batch of images, to get their sizes
    n_features, fmaps_rez = get_fmaps_sizes(_fmaps_fn, images[0:sample_batch_size], device)
    n_features_full = n_features*n_features+n_features # this is total dim of feature matrix once we add in second order combinations.
    
    # Create full model value buffers    
    best_models = np.full(shape=(n_voxels,), fill_value=-1, dtype=int)   
    best_losses = np.full(fill_value=np.inf, shape=(n_voxels), dtype=dtype)
    best_w_params = np.zeros(shape=(n_voxels, n_features_full ), dtype=dtype)

    if add_bias:
        best_w_params = np.concatenate([best_w_params, np.ones(shape=(len(best_w_params),1), dtype=dtype)], axis=1)
    features_mean = None
    features_std = None
    if zscore:
        features_mean = np.zeros(shape=(n_voxels, n_features_full), dtype=dtype)
        features_std  = np.zeros(shape=(n_voxels, n_features_full), dtype=dtype)
       
    combs_zstats = np.zeros(shape=(n_prfs,4), dtype=dtype)
    
    # will save pca stuff as well
    pca_wts = np.zeros(shape=(n_features_full, n_features_full, n_prfs), dtype=dtype) # will be [ncomponents x nfeatures x nmodels]
    pca_pre_mean = np.zeros(shape=(n_features_full, n_prfs), dtype=dtype)
    pct_var_expl = np.zeros(shape=(n_features_full, n_prfs), dtype=dtype)
    n_comp_needed = np.zeros(shape=(n_prfs), dtype=np.int)
    
    start_time = time.time()
    vox_loop_time = 0
    print ('')
    
    with torch.no_grad():
        
        # Looping over models (here models are different spatial RF definitions)
        for m,(x,y,sigma) in enumerate(models):
            if debug and m>1:
                break

            t = time.time()            
            # Get features for the desired pRF, across all trn set images            
            features, zs = get_features_in_prf_combinations((x,y,sigma), _fmaps_fn, images, sample_batch_size, aperture, device)     
            combs_zstats[m,:] = zs
            elapsed = time.time() - t
#             print(features.shape)
            # separate design matrix into training/held out data (for lambda selection)
            trn_features = features[:trn_size]
            out_features = features[trn_size:]   

            # Perform PCA to decorrelate feats and reduce dimensionality
            pca.fit(trn_features)
            trn_scores = pca.transform(trn_features)
            out_scores = pca.transform(out_features)
            wts = pca.components_
            ev = pca.explained_variance_
            ev = ev/np.sum(ev)*100
            pca_wts[0:len(ev),:,m] = wts # save a record of the transformation to interpret encoding model weights later [ncomponents x nfeatures]
            pca_pre_mean[:,m] = pca.mean_ # mean of each feature, nfeatures long - needed to reproduce transformation
            pct_var_expl[0:len(ev),m] = ev   # max len of ev is the number of components (note for a small # samples, this could be smaller than total feature #)
            ncompneeded = int(np.where(np.cumsum(ev)>min_pct_var)[0][0] if np.any(np.cumsum(ev)>min_pct_var) else len(ev))
            n_comp_needed[m] = ncompneeded
            print('\nx=%.1f, y=%.1f, sigma=%.1f: retaining %d components to expl %d pct var\n'%(x,y,sigma, ncompneeded, min_pct_var))
            trn_features = trn_scores[:,0:ncompneeded]
            out_features = out_scores[:,0:ncompneeded]
 
            if zscore:  
                features_m = np.mean(trn_features, axis=0, keepdims=True) #[:trn_size]
                features_s = np.std(trn_features, axis=0, keepdims=True) + 1e-6          
                trn_features -= features_m
                trn_features /= features_s    

            if add_bias:
                trn_features = np.concatenate([trn_features, np.ones(shape=(len(trn_features), 1), dtype=dtype)], axis=1)
                out_features = np.concatenate([out_features, np.ones(shape=(len(out_features), 1), dtype=dtype)], axis=1)

            # Send matrices to gpu
            _xtrn = torch_utils._to_torch(trn_features, device=device)
            _xout = torch_utils._to_torch(out_features, device=device)   

            # Do part of the matrix math involved in ridge regression optimization out of the loop, 
            # because this part will be same for all the voxels.
            _cof = _cofactor_fn_cpu(_xtrn, lambdas = [0.0]) # no ridge param here because already regularizing by doing pca first

            # Now looping over batches of voxels (only reason is because can't store all in memory at same time)
            vox_start = time.time()
            for rv,lv in numpy_utility.iterate_range(0, n_voxels, voxel_batch_size):
                sys.stdout.write('\rfitting model %4d of %-4d, voxels [%6d:%-6d] of %d' % (m, n_prfs, rv[0], rv[-1], n_voxels))

                # Send matrices to gpu
                _vtrn = torch_utils._to_torch(trn_data[:,rv], device=device)
                _vout = torch_utils._to_torch(out_data[:,rv], device=device)

                # Here is where optimization happens - relatively simple matrix math inside loss fn.
                _betas, _loss = _loss_fn(_cof, _vtrn, _xout, _vout) #   [#lambda, #feature, #voxel, ], [#lambda, #voxel]
                # Now have a set of weights (in betas) and a loss value for every voxel and every lambda. 
                # goal is then to choose for each voxel, what is the best lambda and what weights went with that lambda.

                # choose best lambda value and the loss that went with it.
                _values, _select = torch.min(_loss, dim=0)
                betas = torch_utils.get_value(_betas)
                values, select = torch_utils.get_value(_values), torch_utils.get_value(_select)

                # comparing this loss to the other models for each voxel (e.g. the other RF position/sizes)
                imp = values<best_losses[rv]

                if np.sum(imp)>0:
                    # for whichever voxels had improvement relative to previous models, save parameters now
                    # this means we won't have to save all params for all models, just best.
                    arv = np.array(rv)[imp]
                    li = select[imp]

                    best_losses[arv] = values[imp]
                    best_models[arv] = m
                    if zscore:
                        features_mean[arv,0:ncompneeded] = features_m # broadcast over updated voxels
                        features_std[arv,0:ncompneeded]  = features_s
                        features_mean[arv,ncompneeded:] = 0.0 # make sure to fill zeros here
                        features_std[arv,ncompneeded:] = 0.0 # make sure to fill zeros here
                    # taking the weights associated with the best lambda value
                    # remember that they won't fill entire matrix, rest of values stay at zero
                    best_w_params[arv,0:ncompneeded] = numpy_utility.select_along_axis(betas[:,0:ncompneeded,imp], li, run_axis=2, choice_axis=0).T
                    best_w_params[arv,ncompneeded:] = 0.0 # make sure to fill zeros here
                    # bias is always last value, even if zeros for the later features
                    if add_bias:
                        best_w_params[arv,-1] = numpy_utility.select_along_axis(betas[:,-1,imp], li, run_axis=1, choice_axis=0).T

            vox_loop_time += (time.time() - vox_start)
            elapsed = (time.time() - vox_start)

    # Print information about how fitting went...
    total_time = time.time() - start_time
    inv_time = total_time - vox_loop_time
    return_params = [best_w_params[:,:n_features_full],]
    if add_bias:
        return_params += [best_w_params[:,-1],]
    else: 
        return_params += [None,]
    print ('\n---------------------------------------')
    print ('total time = %fs' % total_time)
    print ('total throughput = %fs/voxel' % (total_time / n_voxels))
    print ('voxel throughput = %fs/voxel' % (vox_loop_time / n_voxels))
    print ('setup throughput = %fs/model' % (inv_time / n_prfs))
    sys.stdout.flush()
    return best_losses, [pca_wts, pct_var_expl, min_pct_var, n_comp_needed, pca_pre_mean], [models[best_models],]+return_params+[features_mean, features_std]+[best_models], combs_zstats



 

def _cofactor_fn(_x, lambdas, device):
    '''
    Generating a matrix needed to solve ridge regression model for each lambda value.
    Ridge regression (Tikhonov) solution is :
    w = (X^T*X + I*lambda)^-1 * X^T * Y
    This func will return (X^T*X + I*lambda)^-1 * X^T. 
    So once we have that, can just multiply by training data (Y) to get weights.
    returned size is [nLambdas x nFeatures x nTrials]
    '''
    _f = torch.stack([(torch.mm(torch.t(_x), _x) + torch.eye(_x.size()[1], device=device) * l).inverse() for l in lambdas], axis=0) 
    
    # [#lambdas, #feature, #feature]       
    return torch.tensordot(_f, _x, dims=[[2],[1]]) # [#lambdas, #feature, #sample]




class Torch_fwRF_voxel_block(nn.Module):
    '''
    This is the module that maps from feature maps to voxel predictions according to weights.
    This works for a batch of voxels at a time. 
    Initialize with one set of voxels, but can use load_voxel_block to run w different batches
    '''

    def __init__(self, _fmaps_fn, params, input_shape=(1,3,227,227), aperture=1.0, pc=None, combs_zstats=None):
        super(Torch_fwRF_voxel_block, self).__init__()
        print('Making fwrf module...')
        self.aperture = aperture
        models, weights, bias, features_mt, features_st, best_model_inds = params
        device = next(_fmaps_fn.parameters()).device
        _x =torch.empty((1,)+input_shape[1:], device=device).uniform_(0, 1)
        _fmaps = _fmaps_fn(_x)
        self.fmaps_rez = []
        for k,_fm in enumerate(_fmaps):
            assert _fm.size()[2]==_fm.size()[3], 'All feature maps need to be square'
            self.fmaps_rez += [_fm.size()[2],]
        
        self.prfs = []
        for k,n_pix in enumerate(self.fmaps_rez):
            prf = numpy_utility.make_gaussian_mass_stack(models[:,0], models[:,1], models[:,2], n_pix, size=aperture, dtype=np.float32)[2]
            self.prfs += [nn.Parameter(torch.from_numpy(prf).to(device), requires_grad=False),]
            self.register_parameter('prfs%d'%k, self.prfs[-1])
            
        self.weights = nn.Parameter(torch.from_numpy(weights).to(device), requires_grad=False)
        self.bias = None
        if bias is not None:
            self.bias = nn.Parameter(torch.from_numpy(bias).to(device), requires_grad=False)
            
        self.features_m = None
        self.features_s = None
        if features_mt is not None:
            self.features_m = nn.Parameter(torch.from_numpy(features_mt.T).to(device), requires_grad=False)
        if features_st is not None:
            self.features_s = nn.Parameter(torch.from_numpy(features_st.T).to(device), requires_grad=False)
       
        # add in params related to pca on training features, if this was done. otherwise ignore.
        self.pca_wts = None
        self.n_comp_needed = None
        self.pca_pre_mean = None
        if pc is not None:
            self.pca_wts = pc[0]
            self.n_comp_needed = pc[3]
            self.pca_pre_mean = pc[4]
            self.n_comp_this_batch = nn.Parameter(torch.from_numpy(self.n_comp_needed[best_model_inds]).to(device), requires_grad=False)
            self.pca_wts_this_batch = nn.Parameter(torch.from_numpy(self.pca_wts[:,:,best_model_inds]).to(device), requires_grad=False)
            self.pca_premean_this_batch = nn.Parameter(torch.from_numpy(self.pca_pre_mean[:,best_model_inds]).to(device), requires_grad=False)
        
        self.combs_zstats = None
        if combs_zstats is not None:
            self.combs_zstats = combs_zstats
            self.combs_zstats_this_batch = nn.Parameter(torch.from_numpy(self.combs_zstats[best_model_inds,:]).to(device), requires_grad=False)
            
    def load_voxel_block(self, *params):
        # This takes a given set of parameters for the voxel batch of interest, and puts them 
        # into the right fields of the module so we can use them in a forward pass.
        models = params[0]
                
        for _prfs,n_pix in zip(self.prfs, self.fmaps_rez):
            prfs = numpy_utility.make_gaussian_mass_stack(models[:,0], models[:,1], models[:,2], n_pix, size=self.aperture, dtype=np.float32)[2]
            if len(prfs)<_prfs.size()[0]:
                pp = np.zeros(shape=_prfs.size(), dtype=prfs.dtype)
                pp[:len(prfs)] = prfs
                torch_utils.set_value(_prfs, pp)
            else:
                torch_utils.set_value(_prfs, prfs)
                
        if self.combs_zstats is not None:
            best_model_inds = params[5]
            torch_utils.set_value(self.combs_zstats_this_batch, self.combs_zstats[best_model_inds,:])
            
        if self.pca_wts is not None:
            
            # figure out which pca parameters go with which voxels in this voxel batch
            best_model_inds = params[5]
#             print([self.pca_wts_this_batch.shape[0],len(best_model_inds)])
            if len(best_model_inds)<self.pca_wts_this_batch.shape[2]:
                
                # if this is a small batch of trials, pad it with zeros                
                pp1 = np.zeros(shape=self.pca_wts_this_batch.shape, dtype=self.pca_wts.dtype)
                pp1[:,:,0:len(best_model_inds)] = self.pca_wts[:,:,best_model_inds]
                
                pp2 = np.zeros(shape=self.n_comp_this_batch.shape, dtype=self.n_comp_needed.dtype)
                pp2[0:len(best_model_inds)] = self.n_comp_needed[best_model_inds]   
                
                pp3 = np.zeros(shape=self.pca_premean_this_batch.shape, dtype=self.pca_pre_mean.dtype)
                pp3[:,0:len(best_model_inds)] = self.pca_pre_mean[:,best_model_inds]
                
                
                torch_utils.set_value(self.pca_wts_this_batch,   pp1)
                torch_utils.set_value(self.n_comp_this_batch,   pp2)
                torch_utils.set_value(self.pca_premean_this_batch,   pp3)
            else:
                torch_utils.set_value(self.pca_wts_this_batch,   self.pca_wts[:,:,best_model_inds])
                torch_utils.set_value(self.n_comp_this_batch,   self.n_comp_needed[best_model_inds])
                torch_utils.set_value(self.pca_premean_this_batch, self.pca_pre_mean[:,best_model_inds])
                
        for _p,p in zip([self.weights, self.bias], params[1:3]):
            if _p is not None:
                if len(p)<_p.size()[0]:
                    pp = np.zeros(shape=_p.size(), dtype=p.dtype)
                    pp[:len(p)] = p
                    torch_utils.set_value(_p, pp)
                else:
                    torch_utils.set_value(_p, p)
                    
        for _p,p in zip([self.features_m, self.features_s], params[3:]):
            if _p is not None:
                if len(p)<_p.size()[1]:
                    pp = np.zeros(shape=(_p.size()[1], _p.size()[0]), dtype=p.dtype)
                    pp[:len(p)] = p
                    torch_utils.set_value(_p, pp.T)
                else:
                    torch_utils.set_value(_p, p.T)
 
    def forward(self, _fmaps):

        _features = torch.cat([torch.tensordot(_fm, _prf, dims=[[2,3], [1,2]]) for _fm,_prf in zip(_fmaps, self.prfs)], dim=1) # [#samples, #features, #voxels]
        
        if self.combs_zstats is not None:    
            # convert these first order features into a larger matrix that includes second-order combinations of features            
            f = torch_utils._to_torch(np.zeros(shape=(_features.shape[0], _features.shape[1]*_features.shape[1]+_features.shape[1], _features.shape[2]),dtype=self.combs_zstats.dtype), device=_features.device)
            for vv in range(_features.shape[2]):
                f_first = (_features[:,:,vv] - self.combs_zstats[vv,0])/self.combs_zstats[vv,1]
                f_second = torch.tile(_features[:,:,vv], [1,_features.shape[1]]) * torch.repeat_interleave(_features[:,:,vv], _features.shape[1], axis=1)
                f_second = (f_second - self.combs_zstats[vv,2])/self.combs_zstats[vv,3]
                f[:,:,vv] = torch.cat([f_first, f_second], axis=1)
                
            _features = f
#         print(_features.shape)
        
        if self.pca_wts is not None:            
        
            # apply the pca matrix to each voxel - to keep all features same length, put zeros for components past the desired number.
            features_pca = torch_utils._to_torch(np.zeros(shape=_features.shape, dtype=self.pca_wts.dtype), device=_features.device)
            
            # features is [#samples, #features, #voxels]
            for vv in range(_features.shape[2]):
#                 print([vv, self.n_comp_this_batch.shape, self.pca_wts_this_batch.shape, self.pca_premean_this_batch.shape])
                features_submean = _features[:,:,vv] - torch.tile(torch.unsqueeze(self.pca_premean_this_batch[:,vv], dim=0), [_features.shape[0],1])
                
                features_pca[:, 0:self.n_comp_this_batch[vv], vv] = torch.tensordot(features_submean, self.pca_wts_this_batch[0:self.n_comp_this_batch[vv],:,vv], dims=[[1],[1]]) 

            _features = features_pca

        if self.features_m is not None:    
            # features_m is [nfeatures x nvoxels]
            _features = _features - torch.tile(torch.unsqueeze(self.features_m, dim=0), [_features.shape[0], 1, 1])

        if self.features_s is not None:
            _features = _features/torch.tile(torch.unsqueeze(self.features_s, dim=0), [_features.shape[0], 1, 1])
            _features[torch.isnan(_features)] = 0.0 # this applies in the pca case when last few columns of features are missing

        # features is [#samples, #features, #voxels] - swap dims to [#voxels, #samples, features]
        _features = torch.transpose(torch.transpose(_features, 0, 2), 1, 2)
        # weights is [#voxels, #features]
        # _r will be [#voxels, #samples, 1] - then [#samples, #voxels]
        _r = torch.squeeze(torch.bmm(_features, torch.unsqueeze(self.weights, 2)), dim=2).t() 
  
        if self.bias is not None:
            _r = _r + torch.tile(torch.unsqueeze(self.bias, 0), [_r.shape[0],1])
            
        return _r

    
def get_predictions(images, _fmaps_fn, _fwrf_fn, params, sample_batch_size=100):
    """
    The predictive fwRF model for arbitrary input image.

    Parameters
    ----------
    images : ndarray, shape (#samples, #channels, x, y)
        Input image block.
    _fmaps_fn: Torch module
        Torch module that returns a list of torch tensors.
        This is defined previously, maps from images to feature maps.
    _fwrf_fn: Torch module
        Torch module that compute the fwrf model for one batch of voxels
        Defined in Torch_fwrf_voxel_block
    params: list including all of the following:
    [
        models : ndarray, shape (#voxels, 3)
            The RF model (x, y, sigma) associated with each voxel.
        weights : ndarray, shape (#voxels, #features)
            Tuning weights
        bias: Can contain a bias parameter of shape (#voxels) if add_bias is True.
           Tuning biases: None if there are no bias
        features_mean (optional): ndarray, shape (#voxels, #feature)
            None if zscore is False. Otherwise returns zscoring average per feature.
        features_std (optional): ndarray, shape (#voxels, #feature)
            None if zscore is False. Otherwise returns zscoring std.dev. per feature.
    ]
    sample_batch_size (default: 100)
        The sample batch size (used where appropriate)

    Returns
    -------
    pred : ndarray, shape (#samples, #voxels)
        The prediction of voxel activities for each voxels associated with the input images.
    """
    
    dtype = images.dtype.type
    device = next(_fmaps_fn.parameters()).device
    _params = [_p for _p in _fwrf_fn.parameters()]
    voxel_batch_size = _params[0].size()[0]    
    n_trials, n_voxels = len(images), len(params[0])

    pred = np.full(fill_value=0, shape=(n_trials, n_voxels), dtype=dtype)
    start_time = time.time()
    
    with torch.no_grad():
        
        ## Looping over voxels here in batches, will eventually go through all.
        for rv, lv in numpy_utility.iterate_range(0, n_voxels, voxel_batch_size):
            
            # for this voxel batch, put the right parameters into the _fwrf_fn module
            # so that we can do forward pass...
            _fwrf_fn.load_voxel_block(*[p[rv] if p is not None else None for p in params])
            pred_block = np.full(fill_value=0, shape=(n_trials, voxel_batch_size), dtype=dtype)
            
            # Now looping over validation set trials in batches
            for rt, lt in numpy_utility.iterate_range(0, n_trials, sample_batch_size):
                sys.stdout.write('\rsamples [%5d:%-5d] of %d, voxels [%6d:%-6d] of %d' % (rt[0], rt[-1], n_trials, rv[0], rv[-1], n_voxels))
                # Get predictions for this set of trials.
                pred_block[rt] = torch_utils.get_value(_fwrf_fn(_fmaps_fn(torch_utils._to_torch(images[rt], device)))) 
                
            pred[:,rv] = pred_block[:,:lv]
            
    total_time = time.time() - start_time
    print ('\n---------------------------------------')
    print ('total time = %fs' % total_time)
    print ('sample throughput = %fs/sample' % (total_time / n_trials))
    print ('voxel throughput = %fs/voxel' % (total_time / n_voxels))
    sys.stdout.flush()
    return pred
