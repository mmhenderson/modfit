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

def fit_texture_model_ridge(images, voxel_data, _texture_fn, models, lambdas, zscore=False, voxel_batch_size=100, 
                            holdout_size=100, shuffle=True, add_bias=False, debug=False, shuff_rnd_seed=0, 
                            device=None, do_varpart=True):
   
    """
    Solve for encoding model weights using ridge regression.
    Inputs:
        images: the training images, [n_trials x 1 x height x width]
        voxel_data: the training voxel data, [n_trials x n_voxels]
        _texture_fn: module that maps from images to texture model features
        models: the list of possible pRFs to test, columns are [x, y, sigma]
        lambdas: ridge lambda parameters to test
        zscore: want to zscore each column of feature matrix before fitting?
        voxel_batch_size: how many voxels to use at a time for model fitting
        holdout_size: how many training trials to hold out for computing loss/lambda selection?
        shuffle: do we shuffle training data order before holding trials out?
        add_bias: add a column of ones to feature matrix, for an additive bias?
        debug: want to run a shortened version of this, to test it?
        shuff_rnd_seed: if we do shuffle training data (shuffle=True), what random seed to use? if zero, choose a new random seed in this code.
    Outputs:
        best_losses: loss value for each voxel (with best pRF and best lambda), eval on held out set
        best_lambdas: best lambda for each voxel (chosen based on loss w held out set)
        best_params: 
            [0] best pRF for each voxel [x,y,sigma]
            [1] best weights for each voxel/feature
            [2] if add_bias=True, best bias value for each voxel
            [3] if zscore=True, the mean of each feature before z-score
            [4] if zscore=True, the std of each feature before z-score
            [5] index of the best pRF for each voxel (i.e. index of row in "models")
        feature_info: describes types of features in texture model, see texture_feature_extractor in texture_statistics.py
        
    """
   
    dtype = images.dtype.type
    if device is None:
        device=torch.device('cpu:0')
#     device = next(_texture_fn.parameters()).device
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
        if shuff_rnd_seed==0:
            print('Computing a new random seed')
            shuff_rnd_seed = int(time.strftime('%M%H%d', time.localtime()))
        print('Seeding random number generator: seed is %d'%shuff_rnd_seed)
        np.random.seed(shuff_rnd_seed)
        np.random.shuffle(order)
    images = images[order]
    voxel_data = voxel_data[order]  
    trn_data = voxel_data[:trn_size]
    out_data = voxel_data[trn_size:]

    n_features_total = _texture_fn.n_features_total
    n_feature_types = len(_texture_fn.feature_types_include)
    if do_varpart:
        n_partial_versions = n_feature_types+1
        partial_version_names = ['full_model']+['leave_out_%s'%ff for ff in _texture_fn.feature_types_include]
        masks = np.concatenate([np.expand_dims(np.array(_texture_fn.feature_column_labels!=ff).astype('int'), axis=0) for ff in np.arange(-1,n_feature_types)], axis=0)
    else:
        n_partial_versions = 1;  
        partial_version_names = ['full_model']
        masks = np.ones([1,n_features_total])
    # "partial versions" will be listed as: [full model, leave out first set of features, leave out second set of features...]

    if add_bias:
        masks = np.concatenate([masks, np.ones([masks.shape[0],1])], axis=1) # always include intercept 
    masks = np.transpose(masks)
    # masks is [n_features_total (including intercept) x n_partial_versions]

    # Create full model value buffers    
    best_models = np.full(shape=(n_voxels,n_partial_versions), fill_value=-1, dtype=int)   
    best_lambdas = np.full(shape=(n_voxels,n_partial_versions), fill_value=-1, dtype=int)
    best_losses = np.full(fill_value=np.inf, shape=(n_voxels,n_partial_versions), dtype=dtype)
    # creating a third dim here, listing the "partial" versions of the model (setting to zero a subset of features at a time)
    best_w_params = np.zeros(shape=(n_voxels, n_features_total,n_partial_versions), dtype=dtype)

    if add_bias:
        best_w_params = np.concatenate([best_w_params, np.ones(shape=(n_voxels,1,n_partial_versions), dtype=dtype)], axis=1)

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
            print('\nGetting features for prf %d: [x,y,sigma] is [%.2f %.2f %.4f]'%(m, models[m,0],  models[m,1],  models[m,2] ))
            
            t = time.time()   
            
            # Get features for the desired pRF, across all trn set image   
        
            all_feat_concat, feature_info = _texture_fn(images, [x,y,sigma])
            
            features = torch_utils.get_value(all_feat_concat)
            
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

            zero_columns = np.sum(trn_features[:,0:-1], axis=0)==0
            if np.sum(zero_columns)>0:
                print('n zero columns: %d'%np.sum(zero_columns))
                for ff in range(len(feature_info[1])):
                    if np.sum(zero_columns[feature_info[0]==ff])>0:
                        print('   %d columns are %s'%(np.sum(zero_columns[feature_info[0]==ff]), feature_info[1][ff]))

            # Looping over versions of model w different features set to zero (variance partition)
            for pp in range(n_partial_versions):
                
                print('\nFitting version %d of %d: %s, '%(pp, n_partial_versions, partial_version_names[pp]))

                nonzero_inds = masks[:,pp]==1
                best_w_tmp = best_w_params[:,nonzero_inds,pp] # chunk of the full weights matrix to work with for this partial model

                # Send matrices to gpu
                _xtrn = torch_utils._to_torch(trn_features[:,nonzero_inds], device=device)
                _xout = torch_utils._to_torch(out_features[:,nonzero_inds], device=device)   

                # Do part of the matrix math involved in ridge regression optimization out of the loop, 
                # because this part will be same for all the voxels.
                _cof = _cofactor_fn_cpu(_xtrn, lambdas)

                # Now looping over batches of voxels (only reason is because can't store all in memory at same time)
                vox_start = time.time()
                for rv,lv in numpy_utility.iterate_range(0, n_voxels, voxel_batch_size):
                    sys.stdout.write('\rVoxels [%6d:%-6d] of %d' % (rv[0], rv[-1], n_voxels))

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
                    imp = values<best_losses[rv,pp]

                    if np.sum(imp)>0:                    
                        # for whichever voxels had improvement relative to previous models, save parameters now
                        # this means we won't have to save all params for all models, just best.
                        arv = np.array(rv)[imp]
                        li = select[imp]
                        best_lambdas[arv,pp] = li
                        best_losses[arv,pp] = values[imp]
                        best_models[arv,pp] = m
                        if zscore:
                            features_mean[arv] = features_m # broadcast over updated voxels
                            features_std[arv]  = features_s
                        # taking the weights associated with the best lambda value
                        best_w_tmp[arv,:] = numpy_utility.select_along_axis(betas[:,:,imp], li, run_axis=2, choice_axis=0).T

                best_w_params[:,nonzero_inds,pp] = best_w_tmp

                vox_loop_time += (time.time() - vox_start)
                elapsed = (time.time() - vox_start)
                sys.stdout.flush()

    # Print information about how fitting went...
    total_time = time.time() - start_time
    inv_time = total_time - vox_loop_time
    return_params = [best_w_params[:,:n_features_total,:]]
    if add_bias:
        return_params += [best_w_params[:,-1,:]]
    else: 
        return_params += [None,]
    print ('\n---------------------------------------')
    print ('total time = %fs' % total_time)
    print ('total throughput = %fs/voxel' % (total_time / n_voxels))
    print ('voxel throughput = %fs/voxel' % (vox_loop_time / n_voxels))
    print ('setup throughput = %fs/model' % (inv_time / n_prfs))
    sys.stdout.flush()
    
    best_params = [models[best_models],]+return_params+[features_mean, features_std]+[best_models]+[partial_version_names]
    
    return best_losses, best_lambdas, best_params, feature_info


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

