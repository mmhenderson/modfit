import sys
import time
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.functional as F
import torch.optim as optim

from utils import numpy_utils, torch_utils, texture_utils

"""
General code for fitting a 'feature weighted receptive field' model to fmri data - looping over many candidate pRF 
models for each voxel, find a set of weights that best predict its responses based on feature space of interest.
Can work for many different types of feature spaces, feature extraction implemented with nn.Module.

Original source of some of this code is the github repository:
https://github.com/styvesg/nsd
It was modified by MH to work for this project.
"""


def _cofactor_fn_cpu(_x, lambdas):
    '''
    Generating a matrix needed to solve ridge regression model for each lambda value.
    Ridge regression (Tikhonov) solution is :
    w = (X^T*X + I*lambda)^-1 * X^T * Y
    This func will return (X^T*X + I*lambda)^-1 * X^T. 
    So once we have that, can just multiply by training data (Y) to get weights.
    returned size is [nLambdas x nFeatures x nTrials]
    This version makes sure that the torch inverse operation is done on the cpu, and in floating point-64 precision.
    Otherwise get bad results for small lambda values. This seems to be a torch-specific bug, noted around May 2021.
    
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
    return _beta, _loss, _pred



def fit_fwrf_model(images, voxel_data, _feature_extractor, prf_models, lambdas, best_model_each_voxel=None,\
                   zscore=False, add_bias=False, voxel_batch_size=100, holdout_size=100, \
                       shuffle=True, shuff_rnd_seed=0, device=None, dtype=np.float32, debug=False):
    
    """
    Solve for encoding model weights using ridge regression.
    Inputs:
        images: the training images, [n_trials x 1 x height x width]
            OR for models where features were pre-computed, this is a list of indices [n_trials,] into the 10,000 long feature array.
        voxel_data: the training voxel data, [n_trials x n_voxels]
        _feature_extractor_fn: module that maps from images to model features
        prf_models: the list of possible pRFs to test, columns are [x, y, sigma]
        lambdas: ridge lambda parameters to test
        zscore: want to zscore each column of feature matrix before fitting?
        add_bias: add a column of ones to feature matrix, for an additive bias?
        voxel_batch_size: how many voxels to use at a time for model fitting
        holdout_size: how many training trials to hold out for computing loss/lambda selection?
        shuffle: do we shuffle training data order before holding trials out?      
        shuff_rnd_seed: if we do shuffle training data (shuffle=True), what random seed to use? if zero, choose a new random seed in this code.
        device: what device to use? cpu/cuda
        debug: want to run a shortened version of this, to test it?
    Outputs:
        best_losses: loss value for each voxel (with best pRF and best lambda), eval on held out set
        best_lambdas: best lambda for each voxel (chosen based on loss w held out set)
        best_params: 
            [0] best pRF for each voxel [x,y,sigma]
            [1] best weights for each voxel/feature
            [2] if add_bias=True, best bias value for each voxel
            [3] if zscore=True, the mean of each feature before z-score
            [4] if zscore=True, the std of each feature before z-score
            [5] index of the best pRF for each voxel (i.e. index of row in "prf_models")
        
    """

    if device is None:
        device=torch.device('cpu:0')

    print ('dtype = %s' % dtype)
    print ('device = %s' % device)

    n_trials = len(images)
    n_prfs = len(prf_models)
    n_voxels = voxel_data.shape[1]   

    # Get train/holdout splits.
    # Held-out data here is used for lamdba selection.
    # This is the inner part of nested cross-validation; there is another portion of data ('val')
    # which never enters this function.
    trn_size = n_trials - holdout_size
    assert trn_size>0, 'Training size needs to be greater than zero'
    print ('trn_size = %d (%.1f%%)' % (trn_size, float(trn_size)*100/len(voxel_data)))
    order = np.arange(len(voxel_data), dtype=int)
    if shuffle:
        if shuff_rnd_seed==0:
            print('Computing a new random seed')
            shuff_rnd_seed = int(time.strftime('%M%H%d', time.localtime()))
        print('Seeding random number generator: seed is %d'%shuff_rnd_seed)
        np.random.seed(shuff_rnd_seed)
        np.random.shuffle(order)
        
    images = images[order]
    
    train_trial_order = order[:trn_size]
    holdout_trial_order = order[trn_size:]

    trn_data = copy.deepcopy(voxel_data[train_trial_order,:])
    out_data = copy.deepcopy(voxel_data[holdout_trial_order,:])
    
    
    # Here is where any model-specific additional initialization steps are done
    # Includes initializing pca params arrays, if doing pca
    if len(images.shape)>1:
        image_size = images.shape[2:4]
    else:
        image_size = None
    _feature_extractor.init_for_fitting(image_size, prf_models, dtype)
    max_features = _feature_extractor.max_features

    # Decide whether to do any "partial" versions of the models (leaving out subsets of features)
    # Purpose is for variance partition
    masks, partial_version_names = _feature_extractor.get_partial_versions()
    n_partial_versions = len(partial_version_names) # will be one if skipping varpart
    if add_bias:
        masks = np.concatenate([masks, np.ones([masks.shape[0],1])], axis=1) # always include intercept 
    masks = np.transpose(masks)
    # masks is [n_features_total (including intercept) x n_partial_versions]

    # Initialize arrays to store model fitting params
    best_w_params = np.zeros(shape=(n_voxels, max_features ,n_partial_versions), dtype=dtype)
    best_prf_models = np.full(shape=(n_voxels,n_partial_versions), fill_value=-1, dtype=int)   
    best_lambdas = np.full(shape=(n_voxels,n_partial_versions), fill_value=-1, dtype=int)
    best_losses = np.full(fill_value=np.inf, shape=(n_voxels,n_partial_versions), dtype=dtype)

    # Initialize arrays to store the trial-wise predictions (need these for stacking)
    # Using JUST the held out trials here so the errors are always cross-validated
    best_train_holdout_preds = np.zeros(shape=(n_voxels, holdout_size, n_partial_versions), dtype=dtype)

    # Additional params that are optional
    if add_bias:
        best_w_params = np.concatenate([best_w_params, np.zeros(shape=(n_voxels,1,n_partial_versions), dtype=dtype)], axis=1)

    if zscore:
        features_mean = np.zeros(shape=(n_voxels, max_features), dtype=dtype)
        features_std  = np.zeros(shape=(n_voxels, max_features), dtype=dtype)
    else:
        features_mean = None
        features_std = None

    start_time = time.time()
    vox_loop_time = 0

    print ('---------------------------------------\n')
    
    with torch.no_grad(): # make sure local gradients are off to save memory
        
        # Looping over prf_models (here prf_models are different spatial RF definitions)
        for m,(x,y,sigma) in enumerate(prf_models):
            if debug and m>1:
                break
                
            print('\nGetting features for prf %d: [x,y,sigma] is [%.2f %.2f %.4f]'%(m, \
                                                        prf_models[m,0],  prf_models[m,1],  prf_models[m,2]))

            t = time.time()            

            # Get features for the desired pRF, across all trn set image  
            # Features is size [ntrials x nfeatures]
            # nfeatures may be less than max_features, because max_features is the largest number possible for any pRF.
            # feature_inds_defined is length max_features, and tells which of the features in max_features 
            # are included in features.
            features, feature_inds_defined = _feature_extractor(images, (x,y,sigma), m, fitting_mode=True)
            features = features.detach().cpu().numpy() 
            
            elapsed = time.time() - t

            n_features_actual = features.shape[1]
            
            if zscore:  
                features_m = np.mean(features, axis=0, keepdims=True) #[:trn_size]
                features_s = np.std(features, axis=0, keepdims=True) + 1e-6          
                features -= features_m
                features /= features_s    

            if add_bias:
                features = np.concatenate([features, np.ones(shape=(len(features), 1), dtype=dtype)], axis=1)
                feature_inds_defined = np.concatenate((feature_inds_defined, [True]), axis=0)
                
            trn_features = features[:trn_size,:]
            out_features = features[trn_size:,:]
            
            
            # Going to keep track of whether current prf is better than running best, for each voxel.
            # This is for the full model only.
            # Will use this to make sure for each partial model, we end up saving the params for the 
            # prf that was best w full model.
            if best_model_each_voxel is None:
                full_model_improved = np.zeros((n_voxels,),dtype=bool)
                voxels_to_fit = np.arange(0, n_voxels)
            else:
                voxels_to_fit = np.where(best_model_each_voxel==m)[0]
            
            if len(voxels_to_fit)==0:
                print('No voxels have this pRF saved as their best model, skipping it.')
                continue
            
            
            n_voxels_to_fit = len(voxels_to_fit)            
            n_voxel_batches = int(np.ceil(n_voxels_to_fit/voxel_batch_size))

            
            # Looping over versions of model w different features set to zero (variance partition)
            for pp in range(n_partial_versions):

                print('\nFitting version %d of %d: %s, '%(pp, n_partial_versions, partial_version_names[pp]))

                # nonzero_inds_full is length max_features (or max_features+1 if bias=True)
                # same size as the final params matrices will be.
                nonzero_inds_full = np.logical_and(masks[:,pp], feature_inds_defined)             
                # nonzero_inds_full is restricted to just indices that are defined for this prf - ie same size as features.
                nonzero_inds_short = masks[feature_inds_defined,pp]==1
        
                # Send matrices to gpu    
                _xtrn = torch_utils._to_torch(trn_features[:, nonzero_inds_short], device=device)
                _xout = torch_utils._to_torch(out_features[:, nonzero_inds_short], device=device)   

                # Do part of the matrix math involved in ridge regression optimization out of the loop, 
                # because this part will be same for all the voxels.
                _cof = _cofactor_fn_cpu(_xtrn, lambdas = lambdas) 

                # Now looping over batches of voxels (only reason is because can't store all in memory at same time)
                vox_start = time.time()

                for vi in range(n_voxel_batches):
                    
                    vinds = np.arange(voxel_batch_size*vi, np.min([voxel_batch_size*(vi+1), n_voxels_to_fit]))
                    rv = voxels_to_fit[vinds]
                    lv = len(vinds)

                    sys.stdout.write('\rfitting model %4d of %-4d, voxel batch %d of %d'%(m, n_prfs, vi, n_voxel_batches))
                    if best_model_each_voxel is not None:
                        print(vinds)
                        print(rv)
                    
                    # Send matrices to gpu
                    _vtrn = torch_utils._to_torch(trn_data[:,rv], device=device)
                    _vout = torch_utils._to_torch(out_data[:,rv], device=device)

                    # Here is where optimization happens - relatively simple matrix math inside loss fn.
                    _betas, _loss, _pred_out = _loss_fn(_cof, _vtrn, _xout, _vout) 
                    #   [#lambda, #feature, #voxel, ], [#lambda, #voxel], [trials x lambdas x voxels]
                    # Keep trial-by-trial predictions for each held-out set trial (need for stacking)
                    pred_out = torch_utils.get_value(_pred_out) 
                
                    # choose best lambda value and the loss that went with it.
                    _loss_values, _lambda_index = torch.min(_loss, dim=0)
                    loss_values, lambda_index = torch_utils.get_value(_loss_values), torch_utils.get_value(_lambda_index)
                    betas = torch_utils.get_value(_betas)
                    
                    if best_model_each_voxel is None:
                        
                        if pp==0:

                            # comparing this loss to the other prf_models for each voxel (e.g. the other RF position/sizes)
                            assert(partial_version_names[pp]=='full_model' or \
                                   partial_version_names[pp]=='full_combined_model')               
                            imp = loss_values<best_losses[rv,pp]
                            full_model_improved[rv] = imp

                        else:

                            # for the partial models we don't actually care which pRF was best for the partial model itself,
                            # just using which pRF is best for the full model. This way pRF is always the same even when 
                            # leaving a set of features out.
                            imp = full_model_improved[rv]

                    else:
                        
                        imp = np.ones((lv,))==1

                    if np.sum(imp)>0:

                        # for whichever voxels had improvement relative to previous prf_models, save parameters now
                        # this means we won't have to save all params for all prf_models, just best.
                        arv = np.array(rv)[imp]

                        lambda_inds = lambda_index[imp]
                        best_lambdas[arv,pp] = lambda_inds
                        best_losses[arv,pp] = loss_values[imp]                        
                        best_prf_models[arv,pp] = m
                        if zscore and pp==0:
                            
                            # only need to update the mean/std if we're working with the full model, 
                            # because those will be same for all partial versions.
                            fmean_tmp = copy.deepcopy(features_mean[arv,:])
                            fstd_tmp = copy.deepcopy(features_std[arv,:])
                            fmean_tmp[:,nonzero_inds_full[0:-1]] = features_m[0,nonzero_inds_short[0:-1]] 
                            # broadcast over updated voxels
                            fmean_tmp[:,~nonzero_inds_full[0:-1]] = 0.0
                            fstd_tmp[:,nonzero_inds_full[0:-1]] = features_s[0,nonzero_inds_short[0:-1]] 
                            # broadcast over updated voxels
                            fstd_tmp[:,~nonzero_inds_full[0:-1]] = 0.0
                            features_mean[arv,:] = fmean_tmp
                            features_std[arv,:] = fstd_tmp
                            
                        # taking the weights associated with the best lambda value
                        # remember that they won't fill entire matrix, rest of values stay at zero
                        best_w_tmp = copy.deepcopy(best_w_params[arv,:,pp])
                        best_w_tmp[:,nonzero_inds_full] = numpy_utils.select_along_axis(betas[:,:,imp], lambda_inds, \
                                                                                        run_axis=2, choice_axis=0).T
                        best_w_tmp[:,~nonzero_inds_full] = 0.0 # make sure to fill zeros here
                        best_w_params[arv,:,pp] = best_w_tmp
                        
                        # Save the trialwise predictions for all trials in their original order.
                        # Choosing predictions from whichever lambda was best.
                        best_train_holdout_preds[arv,:,pp] = numpy_utils.select_along_axis(pred_out[:,:,imp], \
                                                                               lambda_inds, run_axis=2, choice_axis=1).T;

                vox_loop_time += (time.time() - vox_start)
                elapsed = (time.time() - vox_start)
                sys.stdout.flush()

    # Print information about how fitting went...
    total_time = time.time() - start_time
    inv_time = total_time - vox_loop_time
    return_params = [best_w_params[:,0:max_features,:],]
    if add_bias:
        return_params += [best_w_params[:,-1,:],]
    else: 
        return_params += [None,]
    print ('\n---------------------------------------')
    print ('total time = %fs' % total_time)
    print ('total throughput = %fs/voxel' % (total_time / n_voxels))
    print ('voxel throughput = %fs/voxel' % (vox_loop_time / n_voxels))
    print ('setup throughput = %fs/model' % (inv_time / n_prfs))
    
    # This step clears the big feature maps for training data from feature extractor (no longer needed)
    _feature_extractor.clear_big_features()
    
    best_params = [prf_models[best_prf_models],]+return_params+[features_mean, features_std]+[best_prf_models]
    sys.stdout.flush()

    return best_losses, best_lambdas, best_params, best_train_holdout_preds, holdout_trial_order