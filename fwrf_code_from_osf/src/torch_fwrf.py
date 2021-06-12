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
import src.numpy_utility as pnu

import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.functional as F
import torch.optim as optim
from src.numpy_utility import iterate_range

def get_value(_x):
    return np.copy(_x.data.cpu().numpy())

def set_value(_x, x):
    if list(x.shape)!=list(_x.size()):
        _x.resize_(x.shape)
    _x.data.copy_(torch.from_numpy(x))
    
def _to_torch(x, device=None):
    return torch.from_numpy(x).float().to(device)    

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

    
def get_r2(actual,predicted):
  
    # calculate r2 for this fit.
    ssres = np.sum(np.power((predicted - actual),2));
#     print(ssres)
    sstot = np.sum(np.power((actual - np.mean(actual)),2));
#     print(sstot)
    r2 = 1-(ssres/sstot)
    
    return r2

    
def get_features_in_prf(prf_params, _fmaps_fn, images, sample_batch_size, aperture, device):
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

        # Define the RF for this "model" version - at several resolutions.
        _prfs = [_to_torch(pnu.make_gaussian_mass(x, y, sigma, n_pix, size=aperture, \
                                  dtype=dtype)[2], device=device) for n_pix in fmaps_rez]

        # To make full design matrix for all trials, first looping over trials in batches to get the features
        # Only reason to loop is memory constraints, because all trials is big matrices.
        t = time.time()
        n_batches = np.ceil(n_trials/sample_batch_size)
        bb=-1
        for rt,rl in pnu.iterate_range(0, n_trials, sample_batch_size):

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
                                   for _fm,_prf in zip(_fmaps_fn(_to_torch(images[rt], \
                                           device=device)), _prfs)], dim=1) # [#samples, #features]

            # Add features for this batch to full design matrix over all trials
            features[rt] = get_value(_features)
        elapsed = time.time() - t
#         print('\nComputing features took %d sec'%elapsed)
        
    return features

def get_features_in_prf_second_order(prf_params, _fmaps_fn, images, sample_batch_size, aperture, device, zstats=None):
    """
    For a given set of images and a specified pRF position and size, compute the
    activation in each feature map channel. Returns [nImages x nFeatures]
    # Use second order model here, so nFeatures output is nFeatures*nFeatures + nFeatures
    """
    
    dtype = images.dtype.type
    with torch.no_grad():
        
        x,y,sigma = prf_params
        n_trials = images.shape[0]

        # pass first batch of images through feature map, just to get sizes.
        n_features, fmaps_rez = get_fmaps_sizes(_fmaps_fn, images[0:sample_batch_size], device)

        features = np.zeros(shape=(n_trials, n_features*n_features+n_features), dtype=dtype)

        # Define the RF for this "model" version - at several resolutions.
        _prfs = [_to_torch(pnu.make_gaussian_mass(x, y, sigma, n_pix, size=aperture, \
                                  dtype=dtype)[2], device=device) for n_pix in fmaps_rez]

        # To make full design matrix for all trials, first looping over trials in batches to get the features
        # Only reason to loop is memory constraints, because all trials is big matrices.
        t = time.time()
        n_batches = np.ceil(n_trials/sample_batch_size)
        bb=-1
        for rt,rl in pnu.iterate_range(0, n_trials, sample_batch_size):

            bb=bb+1
            # multiplying feature maps by RFs here. 
            # features is [nTrialsTotal x nFeatures*nResolutions]
            # note this is concatenating SFs together from low to high - 
            # cycles through all orient channels in order for first SF, then again for next SF.
            _features = torch.cat([torch.tensordot(_fm, _prf, dims=[[2,3], [0,1]]) \
                                   for _fm,_prf in zip(_fmaps_fn(_to_torch(images[rt], \
                                           device=device)), _prfs)], dim=1) # [#samples, #features]

            # Add features for this batch to full design matrix over all trials
            f = get_value(_features)
            # create all the possible combinations
            f_combs = np.tile(f, [1,n_features]) * np.repeat(f, n_features, axis=1)
            # final array will have all these concatenated
            features[rt,0:n_features] = f
            features[rt,n_features:] = f_combs
            
        elapsed = time.time() - t
#         print('\nComputing features took %d sec'%elapsed)
        
        # to make first and second order features comparable, scale each separately by overall mean and variance 
        # doing this across all images so that it doesn't disrupt covariance structure within each set of features.
        if zstats is None:
#             print('computing mean and std')
            mean_first = np.mean(features[:,0:n_features], axis=None)
            std_first = np.std(features[:,0:n_features], axis=None)        
            mean_second = np.mean(features[:,n_features:], axis=None)
            std_second = np.std(features[:,n_features:], axis=None)
        else:
#             print('using input mean and std:')
#             print(zstats)
            # if this is validation pass, these stats have already been computed on trn set so use them here.
            mean_first = zstats[0]
            std_first = zstats[1]
            mean_second = zstats[2]
            std_second = zstats[3]
                        
        features[:,0:n_features] = (features[:,0:n_features] - mean_first)/std_first        
        features[:,n_features:] = (features[:,n_features:] - mean_second)/std_second
        
        
    return features, np.array([mean_first, std_first, mean_second, std_second])


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
            _xtrn = _to_torch(trn_features, device=device)
            _xout = _to_torch(out_features, device=device)   
            
            # Do part of the matrix math involved in ridge regression optimization out of the loop, 
            # because this part will be same for all the voxels.
#             _cof = _cofactor_fn(_xtrn, lambdas, device=device)
            _cof = _cofactor_fn_cpu(_xtrn, lambdas)
            
            # Now looping over batches of voxels (only reason is because can't store all in memory at same time)
            vox_start = time.time()
            for rv,lv in iterate_range(0, n_voxels, voxel_batch_size):
                sys.stdout.write('\rfitting model %4d of %-4d, voxels [%6d:%-6d] of %d' % (m, n_prfs, rv[0], rv[-1], n_voxels))

                # Send matrices to gpu
                _vtrn = _to_torch(trn_data[:,rv], device=device)
                _vout = _to_torch(out_data[:,rv], device=device)

                # Here is where optimization happens - relatively simple matrix math inside loss fn.
                _betas, _loss = _loss_fn(_cof, _vtrn, _xout, _vout) #   [#lambda, #feature, #voxel, ], [#lambda, #voxel]
                # Now have a set of weights (in betas) and a loss value for every voxel and every lambda. 
                # goal is then to choose for each voxel, what is the best lambda and what weights went with that lambda.

                # first choose best lambda value and the loss that went with it.
                _values, _select = torch.min(_loss, dim=0)
                betas = get_value(_betas)
                values, select = get_value(_values), get_value(_select)

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
                    best_w_params[arv,:] = pnu.select_along_axis(betas[:,:,imp], li, run_axis=2, choice_axis=0).T
                    
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
            _xtrn = _to_torch(trn_features, device=device)
            _xout = _to_torch(out_features, device=device)   

            # Do part of the matrix math involved in ridge regression optimization out of the loop, 
            # because this part will be same for all the voxels.
            _cof = _cofactor_fn_cpu(_xtrn, lambdas = [0.0]) # no ridge param here because already regularizing by doing pca first

            # Now looping over batches of voxels (only reason is because can't store all in memory at same time)
            vox_start = time.time()
            for rv,lv in iterate_range(0, n_voxels, voxel_batch_size):
                sys.stdout.write('\rfitting model %4d of %-4d, voxels [%6d:%-6d] of %d' % (m, n_prfs, rv[0], rv[-1], n_voxels))

                # Send matrices to gpu
                _vtrn = _to_torch(trn_data[:,rv], device=device)
                _vout = _to_torch(out_data[:,rv], device=device)

                # Here is where optimization happens - relatively simple matrix math inside loss fn.
                _betas, _loss = _loss_fn(_cof, _vtrn, _xout, _vout) #   [#lambda, #feature, #voxel, ], [#lambda, #voxel]
                # Now have a set of weights (in betas) and a loss value for every voxel and every lambda. 
                # goal is then to choose for each voxel, what is the best lambda and what weights went with that lambda.

                # choose best lambda value and the loss that went with it.
                _values, _select = torch.min(_loss, dim=0)
                betas = get_value(_betas)
                values, select = get_value(_values), get_value(_select)

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
                    best_w_params[arv,0:ncompneeded] = pnu.select_along_axis(betas[:,0:ncompneeded,imp], li, run_axis=2, choice_axis=0).T
                    best_w_params[arv,ncompneeded:] = 0.0 # make sure to fill zeros here
                    # bias is always last value, even if zeros for the later features
                    if add_bias:
                        best_w_params[arv,-1] = pnu.select_along_axis(betas[:,-1,imp], li, run_axis=1, choice_axis=0).T

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

def learn_params_second_order_pca(images, voxel_data, _fmaps_fn, models, min_pct_var = 99, aperture=1.0, zscore=False, sample_batch_size=100, voxel_batch_size=100, holdout_size=100, shuffle=True, add_bias=False, debug=False):
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
       
    second_order_zstats = np.zeros(shape=(n_prfs,4), dtype=dtype)
    
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
            features, zs = get_features_in_prf_second_order((x,y,sigma), _fmaps_fn, images, sample_batch_size, aperture, device)     
            second_order_zstats[m,:] = zs
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
            _xtrn = _to_torch(trn_features, device=device)
            _xout = _to_torch(out_features, device=device)   

            # Do part of the matrix math involved in ridge regression optimization out of the loop, 
            # because this part will be same for all the voxels.
            _cof = _cofactor_fn_cpu(_xtrn, lambdas = [0.0]) # no ridge param here because already regularizing by doing pca first

            # Now looping over batches of voxels (only reason is because can't store all in memory at same time)
            vox_start = time.time()
            for rv,lv in iterate_range(0, n_voxels, voxel_batch_size):
                sys.stdout.write('\rfitting model %4d of %-4d, voxels [%6d:%-6d] of %d' % (m, n_prfs, rv[0], rv[-1], n_voxels))

                # Send matrices to gpu
                _vtrn = _to_torch(trn_data[:,rv], device=device)
                _vout = _to_torch(out_data[:,rv], device=device)

                # Here is where optimization happens - relatively simple matrix math inside loss fn.
                _betas, _loss = _loss_fn(_cof, _vtrn, _xout, _vout) #   [#lambda, #feature, #voxel, ], [#lambda, #voxel]
                # Now have a set of weights (in betas) and a loss value for every voxel and every lambda. 
                # goal is then to choose for each voxel, what is the best lambda and what weights went with that lambda.

                # choose best lambda value and the loss that went with it.
                _values, _select = torch.min(_loss, dim=0)
                betas = get_value(_betas)
                values, select = get_value(_values), get_value(_select)

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
                    best_w_params[arv,0:ncompneeded] = pnu.select_along_axis(betas[:,0:ncompneeded,imp], li, run_axis=2, choice_axis=0).T
                    best_w_params[arv,ncompneeded:] = 0.0 # make sure to fill zeros here
                    # bias is always last value, even if zeros for the later features
                    if add_bias:
                        best_w_params[arv,-1] = pnu.select_along_axis(betas[:,-1,imp], li, run_axis=1, choice_axis=0).T

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
    return best_losses, [pca_wts, pct_var_expl, min_pct_var, n_comp_needed, pca_pre_mean], [models[best_models],]+return_params+[features_mean, features_std]+[best_models], second_order_zstats


class Torch_fwRF_voxel_block(nn.Module):
    '''
    This is the module that maps from feature maps to voxel predictions according to weights.
    This works for a batch of voxels at a time. 
    Initialize with one set of voxels, but can use load_voxel_block to run w different batches
    '''

    def __init__(self, _fmaps_fn, params, input_shape=(1,3,227,227), aperture=1.0, pc=None, second_order_zstats=None):
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
            prf = pnu.make_gaussian_mass_stack(models[:,0], models[:,1], models[:,2], n_pix, size=aperture, dtype=np.float32)[2]
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
        
        self.second_order_zstats = None
        if second_order_zstats is not None:
            self.second_order_zstats = second_order_zstats
            self.second_order_zstats_this_batch = nn.Parameter(torch.from_numpy(self.second_order_zstats[best_model_inds,:]).to(device), requires_grad=False)
            
    def load_voxel_block(self, *params):
        # This takes a given set of parameters for the voxel batch of interest, and puts them 
        # into the right fields of the module so we can use them in a forward pass.
        models = params[0]
                
        for _prfs,n_pix in zip(self.prfs, self.fmaps_rez):
            prfs = pnu.make_gaussian_mass_stack(models[:,0], models[:,1], models[:,2], n_pix, size=self.aperture, dtype=np.float32)[2]
            if len(prfs)<_prfs.size()[0]:
                pp = np.zeros(shape=_prfs.size(), dtype=prfs.dtype)
                pp[:len(prfs)] = prfs
                set_value(_prfs, pp)
            else:
                set_value(_prfs, prfs)
                
        if self.second_order_zstats is not None:
            best_model_inds = params[5]
            set_value(self.second_order_zstats_this_batch, self.second_order_zstats[best_model_inds,:])
            
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
                
                
                set_value(self.pca_wts_this_batch,   pp1)
                set_value(self.n_comp_this_batch,   pp2)
                set_value(self.pca_premean_this_batch,   pp3)
            else:
                set_value(self.pca_wts_this_batch,   self.pca_wts[:,:,best_model_inds])
                set_value(self.n_comp_this_batch,   self.n_comp_needed[best_model_inds])
                set_value(self.pca_premean_this_batch, self.pca_pre_mean[:,best_model_inds])
                
        for _p,p in zip([self.weights, self.bias], params[1:3]):
            if _p is not None:
                if len(p)<_p.size()[0]:
                    pp = np.zeros(shape=_p.size(), dtype=p.dtype)
                    pp[:len(p)] = p
                    set_value(_p, pp)
                else:
                    set_value(_p, p)
                    
        for _p,p in zip([self.features_m, self.features_s], params[3:]):
            if _p is not None:
                if len(p)<_p.size()[1]:
                    pp = np.zeros(shape=(_p.size()[1], _p.size()[0]), dtype=p.dtype)
                    pp[:len(p)] = p
                    set_value(_p, pp.T)
                else:
                    set_value(_p, p.T)
 
    def forward(self, _fmaps):

        _features = torch.cat([torch.tensordot(_fm, _prf, dims=[[2,3], [1,2]]) for _fm,_prf in zip(_fmaps, self.prfs)], dim=1) # [#samples, #features, #voxels]
        
        if self.second_order_zstats is not None:    
            # convert these first order features into a larger matrix that includes second-order combinations of features            
            f = _to_torch(np.zeros(shape=(_features.shape[0], _features.shape[1]*_features.shape[1]+_features.shape[1], _features.shape[2]),dtype=self.second_order_zstats.dtype), device=_features.device)
            for vv in range(_features.shape[2]):
                f_first = (_features[:,:,vv] - self.second_order_zstats[vv,0])/self.second_order_zstats[vv,1]
                f_second = torch.tile(_features[:,:,vv], [1,_features.shape[1]]) * torch.repeat_interleave(_features[:,:,vv], _features.shape[1], axis=1)
                f_second = (f_second - self.second_order_zstats[vv,2])/self.second_order_zstats[vv,3]
                f[:,:,vv] = torch.cat([f_first, f_second], axis=1)
                
            _features = f
#         print(_features.shape)
        
        if self.pca_wts is not None:            
        
            # apply the pca matrix to each voxel - to keep all features same length, put zeros for components past the desired number.
            features_pca = _to_torch(np.zeros(shape=_features.shape, dtype=self.pca_wts.dtype), device=_features.device)
            
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
        for rv, lv in iterate_range(0, n_voxels, voxel_batch_size):
            
            # for this voxel batch, put the right parameters into the _fwrf_fn module
            # so that we can do forward pass...
            _fwrf_fn.load_voxel_block(*[p[rv] if p is not None else None for p in params])
            pred_block = np.full(fill_value=0, shape=(n_trials, voxel_batch_size), dtype=dtype)
            
            # Now looping over validation set trials in batches
            for rt, lt in iterate_range(0, n_trials, sample_batch_size):
                sys.stdout.write('\rsamples [%5d:%-5d] of %d, voxels [%6d:%-6d] of %d' % (rt[0], rt[-1], n_trials, rv[0], rv[-1], n_voxels))
                # Get predictions for this set of trials.
                pred_block[rt] = get_value(_fwrf_fn(_fmaps_fn(_to_torch(images[rt], device)))) 
                
            pred[:,rv] = pred_block[:,:lv]
            
    total_time = time.time() - start_time
    print ('\n---------------------------------------')
    print ('total time = %fs' % total_time)
    print ('sample throughput = %fs/sample' % (total_time / n_trials))
    print ('voxel throughput = %fs/voxel' % (total_time / n_voxels))
    sys.stdout.flush()
    return pred



class add_nonlinearity(nn.Module):
    def __init__(self, _fmaps_fn, _nonlinearity):
        super(add_nonlinearity, self).__init__()
        self.fmaps_fn = _fmaps_fn
        self.nl_fn = _nonlinearity
    def forward(self, _x):
        return [self.nl_fn(_fm) for _fm in self.fmaps_fn(_x)]
