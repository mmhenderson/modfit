import sys
import os
import struct
import time
import numpy as np
import h5py
from tqdm import tqdm
import pickle
import math
import sklearn
from sklearn import decomposition

import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.functional as F
import torch.optim as optim

from utils import numpy_utility, torch_utils
from model_src import texture_statistics

 

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



def _cofactor_fn_cpu(_x, lambdas):
    '''
    Generating a matrix needed to solve ridge regression model for each lambda value.
    Ridge regression (Tikhonov) solution is :
    w = (X^T*X + I*lambda)^-1 * X^T * Y
    This func will return (X^T*X + I*lambda)^-1 * X^T. 
    So once we have that, can just multiply by training data (Y) to get weights.
    returned size is [nLambdas x nFeatures x nTrials]
    This version makes sure that the torch inverse operation is done on the cpu, and in floating point-64 precision.
    Otherwise get bad results for small lambda values. This seems to be a torch-specific bug.
    
    '''
    device_orig = _x.device
    type_orig = _x.dtype
    # switch to this specific format which works with inverse
    _x = _x.to('cpu').to(torch.float64)
    _f = torch.stack([(torch.mm(torch.t(_x), _x) + torch.eye(_x.size()[1], device='cpu', dtype=torch.float64) * l).inverse() for l in lambdas], axis=0) 
    
    # [#lambdas, #feature, #feature] 
    cof = torch.tensordot(_f, _x, dims=[[2],[1]]) # [#lambdas, #feature, #sample]
    
    # put back to whatever way it was before, so that we can continue with other operations as usual
    return cof.to(device_orig).to(type_orig)



def _loss_fn(_cofactor, _vtrn, _xout, _vout):
    '''
    Calculate loss given "cofactor" from cofactor_fn, training data, held-out design matrix, held out data.
    returns weights (betas) based on equation
    w = (X^T*X + I*lambda)^-1 * X^T * Y
    also returns loss for these weights w the held out data. SSE is loss func here.
    '''

    _beta = torch.tensordot(_cofactor, _vtrn, dims=[[2], [0]]) # [#lambdas, #feature, #voxel]
    _pred = torch.tensordot(_xout, _beta, dims=[[1],[1]]) # [#samples, #lambdas, #voxels]
    _loss = torch.sum(torch.pow(_vout[:,None,:] - _pred, 2), dim=0) # [#lambdas, #voxels]
    return _beta, _loss


def get_fmaps_sizes(_fmaps_fn, image_batch, device):
    """ 
    Passing a batch of images through feature maps, in order to compute sizes.
    Returns number of total features across all groups of maps, and the resolution of each map group.
    """
    n_features = 0
    _x = torch.tensor(image_batch).to(device) # the input variable.
    _fmaps = _fmaps_fn(_x)
    resolutions_each_sf = []
    for k,_fm in enumerate(_fmaps):
        n_features = n_features + _fm.size()[1]
        resolutions_each_sf.append(_fm.size()[2])
    
    return n_features, resolutions_each_sf



def get_features_in_prf(prf_params, _fmaps_fn, images, sample_batch_size, aperture, device, to_numpy=True):
    """
    For a given set of images and a specified pRF position and size, compute the
    activation in each feature map channel. Returns [nImages x nFeatures]
    """
    
    dtype = images.dtype.type
    with torch.no_grad():
        
        x,y,sigma = prf_params
        n_trials = images.shape[0]

        # pass first batch of images through feature map, just to get sizes.
        n_features, fmaps_rez = get_fmaps_sizes(_fmaps_fn, images[0:sample_batch_size], device)

        features = np.zeros(shape=(n_trials, n_features), dtype=dtype)
        if to_numpy==False:
             features = torch_utils._to_torch(features, device=device)
                
        # Define the RF for this "model" version - at several resolutions.
        _prfs = [torch_utils._to_torch(numpy_utility.make_gaussian_mass(x, y, sigma, n_pix, size=aperture, \
                                  dtype=dtype)[2], device=device) for n_pix in fmaps_rez]

        # To make full design matrix for all trials, first looping over trials in batches to get the features
        # Only reason to loop is memory constraints, because all trials is big matrices.
        t = time.time()
        n_batches = np.ceil(n_trials/sample_batch_size)
        bb=-1
        for rt,rl in numpy_utility.iterate_range(0, n_trials, sample_batch_size):

            bb=bb+1
#             sys.stdout.write('\rbatch %d of %d'%(bb,n_batches))
            # multiplying feature maps by RFs here. 
            # we have one specified RF position for this version of the model. 
            # Feature maps in _fm go [nTrials x nFeatures(orientations) x nPixels x nPixels]
            # spatial RFs in _prfs go [nPixels x nPixels]
            # purpose of the for looping within this statement is to loop over map resolutions 
            # (e.g. spatial frequencies in model)
            # output _features is [nTrials x nFeatures*nResolutions], so a 2D matrix. 
            # Combining features/resolutions here finally, so we can solve for weights 
            # in that full orient x SF feature space.

            # then combine this with the other "batches" of trials to make a full "model space tensor"
            # features is [nTrialsTotal x nFeatures*nResolutions]

            # note this is concatenating SFs together from low to high - 
            # cycles through all orient channels in order for first SF, then again for next SF.
            _features = torch.cat([torch.tensordot(_fm, _prf, dims=[[2,3], [0,1]]) \
                                   for _fm,_prf in zip(_fmaps_fn(torch_utils._to_torch(images[rt], \
                                           device=device)), _prfs)], dim=1) # [#samples, #features]

            # Add features for this batch to full design matrix over all trials
            if to_numpy:
                features[rt] = torch_utils.get_value(_features)
            else:
                features[rt] = _features
                
        elapsed = time.time() - t
#         print('\nComputing features took %d sec'%elapsed)
        
    return features



def get_features_in_prf_combinations(prf_params, _fmaps_fn, images, sample_batch_size, aperture, device, combs_zstats=None):
    """
    For a given set of images and a specified pRF position and size, compute the
    activation in each feature map channel. Returns [nImages x nFeatures]
    # Use combinations of features in model here, so nFeatures output is nFeatures*nFeatures + nFeatures
    """
    
    dtype = images.dtype.type
    with torch.no_grad():
        
        x,y,sigma = prf_params
        n_trials = images.shape[0]

        # pass first batch of images through feature map, just to get sizes.
        n_features, fmaps_rez = get_fmaps_sizes(_fmaps_fn, images[0:sample_batch_size], device)

        features = np.zeros(shape=(n_trials, n_features*n_features+n_features), dtype=dtype)

        # Define the RF for this "model" version - at several resolutions.
        _prfs = [torch_utils._to_torch(numpy_utility.make_gaussian_mass(x, y, sigma, n_pix, size=aperture, \
                                  dtype=dtype)[2], device=device) for n_pix in fmaps_rez]

        # To make full design matrix for all trials, first looping over trials in batches to get the features
        # Only reason to loop is memory constraints, because all trials is big matrices.
        t = time.time()
        n_batches = np.ceil(n_trials/sample_batch_size)
        bb=-1
        for rt,rl in numpy_utility.iterate_range(0, n_trials, sample_batch_size):

            bb=bb+1
            # multiplying feature maps by RFs here. 
            # features is [nTrialsTotal x nFeatures*nResolutions]
            # note this is concatenating SFs together from low to high - 
            # cycles through all orient channels in order for first SF, then again for next SF.
            _features = torch.cat([torch.tensordot(_fm, _prf, dims=[[2,3], [0,1]]) \
                                   for _fm,_prf in zip(_fmaps_fn(torch_utils._to_torch(images[rt], \
                                           device=device)), _prfs)], dim=1) # [#samples, #features]
            
            # Add features for this batch to full design matrix over all trials
            f = torch_utils.get_value(_features)
            # create all the possible combinations
            f_combs = np.tile(f, [1,n_features]) * np.repeat(f, n_features, axis=1)
            # final array will have all these concatenated
            features[rt,0:n_features] = f
            features[rt,n_features:] = f_combs
            
        elapsed = time.time() - t
#         print('\nComputing features took %d sec'%elapsed)
        
        # to make first and second order features comparable, scale each separately by overall mean and variance 
        # doing this across all images so that it doesn't disrupt covariance structure within each set of features.
        if combs_zstats is None:
#             print('computing mean and std')
            mean_first = np.mean(features[:,0:n_features], axis=None)
            std_first = np.std(features[:,0:n_features], axis=None)        
            mean_combs = np.mean(features[:,n_features:], axis=None)
            std_combs= np.std(features[:,n_features:], axis=None)
        else:
#             print('using input mean and std:')
#             print(combs_zstats)
            # if this is validation pass, these stats have already been computed on trn set so use them here.
            mean_first = combs_zstats[0]
            std_first = combs_zstats[1]
            mean_combs = combs_zstats[2]
            std_combs = combs_zstats[3]
                        
        features[:,0:n_features] = (features[:,0:n_features] - mean_first)/std_first        
        features[:,n_features:] = (features[:,n_features:] - mean_combs)/std_combs
        
        
    return features, np.array([mean_first, std_first, mean_combs, std_combs])

def learn_params_ridge_regression(images, voxel_data, _fmaps_fn, models, lambdas, aperture=1.0, zscore=False, sample_batch_size=100, voxel_batch_size=100, holdout_size=100, shuffle=True, add_bias=False, debug=False):
    """
    Learn the parameters of the fwRF model

    Parameters
    ----------
    images : ndarray, shape (#samples, #channels, x, y)
        Input image block.
    voxels: ndarray, shape (#samples, #voxels)
        Input voxel activities.
    _fmaps_fn: Torch module
        Torch module that returns a list of torch tensors.
        This is defined previously, maps from images to list feature maps.
    models: ndarray, shape (#candidateRF, 3)
        The (x, y, sigma) of all candidate RFs for gridsearch.
    lambdas: ndarray, shape (#candidateRegression)
        The rigde parameter candidates.
    aperture (default: 1.0): scalar
        The span of the stimulus in the unit used for the RF models.
    zscore (default: False)
        Whether to zscore the feature maps or not.
    sample_batch_size (default: 100)
        The sample batch size (used where appropriate)
    voxel_batch_size (default: 100) 
        The voxel batch size (used where appropriate)
    holdout_size (default: 100) 
        The holdout size for model and hyperparameter selection
    shuffle (default: True)
        Whether to shuffle the training set or not.
    add_bias (default: False)
        Whether to add a bias term to the rigde regression or not.

    Returns
    -------
    losses : ndarray, shape (#voxels)
        The final loss for each voxel.
    lambdas : ndarray, shape (#voxels)
        The regression regularization index for each voxel.
    models : ndarray, shape (#voxels, 3)
        The RF model (x, y, sigma) associated with each voxel.
    params : list of ndarray, shape (#voxels, #features)
        Can contain a bias parameter of shape (#voxels) if add_bias is True.
    features_mean : ndarray, shape (#voxels, #feature)
        None if zscore is False. Otherwise returns zscoring average per feature.
    features_std : ndarray, shape (#voxels, #feature)
        None if zscore is False. Otherwise returns zscoring std.dev. per feature.
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
    
    # Create full model value buffers    
    best_models = np.full(shape=(n_voxels,), fill_value=-1, dtype=int)   
    best_lambdas = np.full(shape=(n_voxels,), fill_value=-1, dtype=int)
    best_losses = np.full(fill_value=np.inf, shape=(n_voxels), dtype=dtype)
    best_w_params = np.zeros(shape=(n_voxels, n_features ), dtype=dtype)

    if add_bias:
        best_w_params = np.concatenate([best_w_params, np.ones(shape=(len(best_w_params),1), dtype=dtype)], axis=1)
    features_mean = None
    features_std = None
    if zscore:
        features_mean = np.zeros(shape=(n_voxels, n_features), dtype=dtype)
        features_std  = np.zeros(shape=(n_voxels, n_features), dtype=dtype)
    
    # going to save the covariance matrices too, see how correlated features are in training data.
    covar_each_model = np.zeros(shape=(n_features, n_features, n_prfs), dtype=dtype)
    
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
            features = get_features_in_prf((x,y,sigma), _fmaps_fn, images, sample_batch_size, aperture, device)     
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
    return_params = [best_w_params[:,:n_features],]
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
    return best_losses, best_lambdas, [models[best_models],]+return_params+[features_mean, features_std]+[best_models], covar_each_model


def fit_texture_model_ridge(images, voxel_data, _fmaps_fn_complex, _fmaps_fn_simple, models, lambdas, 
                            aperture=1.0, include_autocorrs=True, include_crosscorrs=True, autocorr_output_pix=5, n_prf_sd_out=2, zscore=False, sample_batch_size=100, voxel_batch_size=100, 
                            holdout_size=100, shuffle=True, add_bias=False, debug=False):
   
   
    dtype = images.dtype.type
    device = next(_fmaps_fn_complex.parameters()).device
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
    n_features_complex, fmaps_rez = get_fmaps_sizes(_fmaps_fn_complex, images[0:sample_batch_size], device)
    n_sf = len(fmaps_rez)
    n_ori = int(n_features_complex/n_sf)
    ori_pairs = np.vstack([[[oo1, oo2] for oo2 in np.arange(oo1+1, n_ori)] for oo1 in range(n_ori) if oo1<n_ori-1])
    n_ori_pairs = np.shape(ori_pairs)[0]
    n_phases=2
    
    # how many features will be needed, in total?
    if include_autocorrs and include_crosscorrs:
        n_features_total = 4 + n_ori*n_sf + n_ori*n_sf*n_phases \
                + n_ori*n_sf*autocorr_output_pix**2 + n_ori*n_sf*n_phases*autocorr_output_pix**2 \
                + n_sf*n_ori_pairs + n_sf*n_ori_pairs*n_phases + (n_sf-1)*n_ori**2 + (n_sf-1)*n_ori**2*n_phases
    elif include_crosscorrs:
        n_features_total = 4 + n_ori*n_sf + n_ori*n_sf*n_phases \
                + n_sf*n_ori_pairs + n_sf*n_ori_pairs*n_phases + (n_sf-1)*n_ori**2 + (n_sf-1)*n_ori**2*n_phases
    elif include_autocorrs:
        n_features_total = 4 + n_ori*n_sf + n_ori*n_sf*n_phases \
                + n_ori*n_sf*autocorr_output_pix**2 + n_ori*n_sf*n_phases*autocorr_output_pix**2
    else:
        n_features_total = 4 + n_ori*n_sf + n_ori*n_sf*n_phases
        
    # Create full model value buffers    
    best_models = np.full(shape=(n_voxels,), fill_value=-1, dtype=int)   
    best_lambdas = np.full(shape=(n_voxels,), fill_value=-1, dtype=int)
    best_losses = np.full(fill_value=np.inf, shape=(n_voxels), dtype=dtype)
    best_w_params = np.zeros(shape=(n_voxels, n_features_total), dtype=dtype)

    if add_bias:
        best_w_params = np.concatenate([best_w_params, np.ones(shape=(len(best_w_params),1), dtype=dtype)], axis=1)
    features_mean = None
    features_std = None
    if zscore:
        features_mean = np.zeros(shape=(n_voxels, n_features_total), dtype=dtype)
        features_std  = np.zeros(shape=(n_voxels, n_features_total), dtype=dtype)
    
    
    start_time = time.time()
    vox_loop_time = 0
    print ('')
    
    with torch.no_grad():
        
        # Looping over models (here models are different spatial RF definitions)
        for m,(x,y,sigma) in enumerate(models):
            if debug and m>1:
                break
            print('\nmodel %d\n'%m)
            t = time.time()            
            # Get features for the desired pRF, across all trn set image     
            all_feat_concat, feature_info = texture_statistics.compute_all_texture_features(_fmaps_fn_complex, _fmaps_fn_simple, images, 
                                                                         [x,y,sigma], sample_batch_size, include_autocorrs, include_crosscorrs, autocorr_output_pix, 
                                                                         n_prf_sd_out, aperture, device=device)
            features = all_feat_concat
#             print(np.any(np.isnan(features)))
#             features = get_features_in_prf((x,y,sigma), _fmaps_fn, images, sample_batch_size, aperture, device)     
            elapsed = time.time() - t
        
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

#             print(features)
            
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
    return_params = [best_w_params[:,:n_features_total],]
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
    return best_losses, best_lambdas, [models[best_models],]+return_params+[features_mean, features_std]+[best_models], feature_info


def learn_params_pca(images, voxel_data, _fmaps_fn, models, min_pct_var = 99, aperture=1.0, zscore=False, sample_batch_size=100, voxel_batch_size=100, holdout_size=100, shuffle=True, add_bias=False, debug=False):
    """
    
    Learn the parameters of the model, using PCA first to decorrelate features.
    
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
    
    # Create full model value buffers    
    best_models = np.full(shape=(n_voxels,), fill_value=-1, dtype=int)   
    best_losses = np.full(fill_value=np.inf, shape=(n_voxels), dtype=dtype)
    best_w_params = np.zeros(shape=(n_voxels, n_features ), dtype=dtype)

    if add_bias:
        best_w_params = np.concatenate([best_w_params, np.ones(shape=(len(best_w_params),1), dtype=dtype)], axis=1)
    features_mean = None
    features_std = None
    if zscore:
        features_mean = np.zeros(shape=(n_voxels, n_features), dtype=dtype)
        features_std  = np.zeros(shape=(n_voxels, n_features), dtype=dtype)
       
    # will save pca stuff as well
    pca_wts = np.zeros(shape=(n_features, n_features, n_prfs), dtype=dtype) # will be [ncomponents x nfeatures x nmodels]
    pca_pre_mean = np.zeros(shape=(n_features, n_prfs), dtype=dtype)
    pct_var_expl = np.zeros(shape=(n_features, n_prfs), dtype=dtype)
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
            # Get features for the desired pRF, across all trn set image            
            features = get_features_in_prf((x,y,sigma), _fmaps_fn, images, sample_batch_size, aperture, device)     
            elapsed = time.time() - t

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
    return_params = [best_w_params[:,:n_features],]
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
    return best_losses, [pca_wts, pct_var_expl, min_pct_var, n_comp_needed, pca_pre_mean], [models[best_models],]+return_params+[features_mean, features_std]+[best_models]


     
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



