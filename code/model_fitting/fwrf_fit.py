import sys
import time
import copy
import numpy as np

import torch

from utils import numpy_utils, torch_utils

"""
General code for fitting a 'feature weighted receptive field' model to fmri data - looping over many candidate pRF 
models for each voxel, find a set of weights that best predict its responses based on feature space of interest.
Can work for many different types of feature spaces (visual, semantic)
"""

def fit_fwrf_model(image_inds_trn, voxel_data_trn, \
                   image_inds_holdout, voxel_data_holdout, \
                   feature_loader, prf_models, lambdas, \
                   best_model_each_voxel=None,\
                   zscore=True, add_bias=True, \
                   voxel_batch_size=100, 
                   trials_use_each_prf_trn = None, \
                   trials_use_each_prf_holdout = None, \
                   device=None, dtype=np.float32, debug=False):
    
    """
    Solve for FWRF encoding model weights using ridge regression.
    Will loop over candidate pRFs and fit the model in each pRF, eventually choosing the pRF that 
    gives the lowest loss on held-out data partition. 
       
    code is based loosely on code from https://github.com/styvesg/nsd
    
    This code can also perform variance partition analysis by looping over various partial 
    versions of the model (different feature sets masked out). The definitions of these partial 
    versions is specified in the feature_loader.
    n_partial_versions is the number of partial versions that will be done. 
    If we're only fitting the full model, then n_partial_versions is 1
    
    Inputs:
        image_inds_trn (1D array): [n_trials,] numerical indices of the images included in training set - 
                               index into the 10,000 long list of images shown to each subject. 
        voxel_data_trn (2D array): training set data, [n_trials x n_voxels]
        image_inds_holdout (1D array): [n_trials,] numerical indices of the images included in holdout set - 
                               index into the 10,000 long list of images shown to each subject. 
        voxel_data_holdout (2D array): holdout set data, [n_trials x n_voxels]
        feature_loader (obj):  an object that loads pre-computed features for the feature space of 
                               interest, see "fwrf_features.py" or "semantic_features.py"
                               will take "image_inds" as input to load correct features
        prf_models (2D array): array of the candidate pRF parameters, [n_prfs x 3]; columns are x,y,sigma
        best_model_each_voxel (1D array, optional): if we already computed pRF params with 
                               another model and would like to re-use them, then this should hold the 
                               index of the best pRF for each voxel. If not specified, then we will fit 
                               pRFs from scratch now. default=None
        zscore (bool, optional): want to z-score each feature across trials? default=True
        add_bias (bool, optional): want to add intercept to the linear model? default=True
        voxel_batch_size (int, optional): how many voxels to fit at a time in a batch? default=100
                               Decreasing this can help with out of memory issues
        trials_use_each_prf_trn (2D array of booleans, optional): if you want to choose a sub-set of trials 
                                to fit, and choose them on a by-pRF basis. [n_trials x n_prfs]
                                for instance if we want to fit just the trials with animate in the prf, etc.
        trials_use_each_prf_holdout (2D array of booleans, optional): if you want to choose a sub-set of trials 
                                to fit, and choose them on a by-pRF basis. [n_trials x n_prfs]
                                for instance if we want to fit just the trials with animate in the prf, etc.
        device (optional): gpu or cpu device to use, default will be to use cpu
        dtype (optional): data type, default=np.float32
        debug (boolean, optional): want to run a fast (stop early) version of this code, to test it? 
                                default=False
        
    Outputs: 
        best_losses: [n_voxels, n_partial_versions] best loss for each voxel
        best_lambdas: [n_voxels, n_partial_versions] best lambda for each voxel
        best_weights: [n_voxels, max_features, n_partial_versions] best weights for each voxel
        best_biases: [n_voxels, n_partial_versions] best bias (intercept) for each voxel
        best_prf_models: [n_voxels, n_partial_versions] index of best pRF for each voxel
        features_mean: [n_prfs, max_features] mean of feature columns, needed for validation
        features_std: [n_prfs, max_features] std of feature columns, needed for validation
        
    """

    if device is None:
        device=torch.device('cpu:0')

    print ('dtype = %s' % dtype)
    print ('device = %s' % device)
    
    # concatenate trn/val inds here, makes it easier to extract features
    image_inds_concat = np.concatenate([image_inds_trn, image_inds_holdout], axis=0)
    n_trials = len(image_inds_concat)
    # make indices for how to get train/val trials back out of concat trials
    trninds = np.arange(n_trials)<len(image_inds_trn)
    valinds = np.arange(n_trials)>=len(image_inds_trn)
    
    n_prfs = len(prf_models)
    n_voxels = voxel_data_trn.shape[1]   

    # clear any stored features from feature loader's memory    
    feature_loader.clear_big_features()
    max_features = feature_loader.max_features 
    # max features for any pRF - can be different actual number for diff pRFs (due to PCA)

    # Decide whether to do any "partial" versions of the models (leaving out subsets of features)
    # Purpose is for variance partition
    masks, partial_version_names = feature_loader.get_partial_versions()
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

    # Additional params that are optional
    if add_bias:
        best_w_params = np.concatenate([best_w_params, np.zeros(shape=(n_voxels,1,n_partial_versions), dtype=dtype)], axis=1)

    if zscore:        
        features_mean = np.zeros(shape=(n_prfs, max_features), dtype=dtype)
        features_std  = np.zeros(shape=(n_prfs, max_features), dtype=dtype)
        print('will z-score each column')
    else:
        features_mean = None
        features_std = None
        print('will not z-score')

    start_time = time.time()
    vox_loop_time = 0

    print ('---------------------------------------\n')
    
    with torch.no_grad(): # make sure local gradients are off to save memory
        
        # Looping over prf_model
        for m,(x,y,sigma) in enumerate(prf_models):
            if debug and m>1:
                break
 
            print('\nGetting features for prf %d: [x,y,sigma] is [%.2f %.2f %.4f]'%(m, \
                                                        prf_models[m,0],  prf_models[m,1],  prf_models[m,2]))

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

            t = time.time()            

            # Get features for the desired pRF, across all trn set image  
            # Features is size [ntrials x nfeatures]
            # nfeatures may be less than max_features; max_features is the largest number possible for any pRF.
            # feature_inds_defined is length max_features, and indicates which of the features in max_features 
            # are included in features.
            features, feature_inds_defined = feature_loader.load(image_inds_concat, m, fitting_mode=True)
              
            elapsed = time.time() - t

            n_features_actual = features.shape[1]
            
            if zscore:                
                features_m = np.mean(features, axis=0, keepdims=True) #[:trn_size]
                features_s = np.std(features, axis=0, keepdims=True) + 1e-6          
                features -= features_m
                features /= features_s   
                # saving these for later so we can exactly reproduce this normalization
                # when doing validation pass...
                features_mean[m,feature_inds_defined] = features_m
                features_std[m,feature_inds_defined] = features_s
            if add_bias:
                features = np.concatenate([features, np.ones(shape=(len(features), 1), dtype=dtype)], axis=1)
                feature_inds_defined = np.concatenate((feature_inds_defined, [True]), axis=0)
                
            trn_features = features[trninds,:]
            out_features = features[valinds,:]
            
            if trials_use_each_prf_trn is not None:
                # select subset of trials to work with
                trials_use_trn = trials_use_each_prf_trn[:,m]
                trn_features = trn_features[trials_use_trn,:]
                trn_data_use = voxel_data_trn[trials_use_trn,:]
            else:
                trn_data_use = voxel_data_trn
                
            if trials_use_each_prf_holdout is not None:
                # select subset of trials to work with
                trials_use_holdout = trials_use_each_prf_holdout[:,m]
                out_features = out_features[trials_use_holdout,:]
                out_data_use = voxel_data_holdout[trials_use_holdout,:]
            else:
                out_data_use = voxel_data_holdout
                
            if trn_data_use.shape[0]==0 or out_data_use.shape[0]==0:
                # if insufficient trials to work with this pRF - skip it.
                # this only happens when the trial counts are subsampled in various control analyses.
                assert(best_model_each_voxel is not None) # this can never happen when we are actually fitting pRFs.
                print('prf %d - not enough trials to fit. skipping voxels with this pRF!'%(m))
                continue
                
            print('prf %d - using %d training trials and %d held-out trials'\
                      %(m, trn_data_use.shape[0], out_data_use.shape[0]))

            # Looping over versions of model w different features set to zero (variance partition)
            for pp in range(n_partial_versions):

                print('\nFitting version %d of %d: %s, '%(pp, n_partial_versions, partial_version_names[pp]))

                # nonzero_inds_full is length max_features (or max_features+1 if bias=True)
                # same size as the final params matrices will be.
                nonzero_inds_full = np.logical_and(masks[:,pp], feature_inds_defined)             
                # nonzero_inds_full is restricted to just indices that are defined for this prf 
                # (same size as features)
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
                    _vtrn = torch_utils._to_torch(trn_data_use[:,rv], device=device)
                    _vout = torch_utils._to_torch(out_data_use[:,rv], device=device)

                    # Here is where optimization happens - matrix math inside loss fn.
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

                            # for the partial models we will always defer to which pRF is best for the full model. 
                            # This makes the partial/full models more directly comparable.
                            imp = full_model_improved[rv]

                    else:
                        
                        imp = np.ones((lv,))==1

                    if np.sum(imp)>0:

                        # For whichever voxels had improvement relative to previous prf_models, save parameters now
                        arv = np.array(rv)[imp]

                        lambda_inds = lambda_index[imp]
                        best_lambdas[arv,pp] = lambda_inds
                        best_losses[arv,pp] = loss_values[imp]                        
                        best_prf_models[arv,pp] = m

                        # taking the weights associated with the best lambda value
                        # remember that they won't fill entire matrix, rest of values stay at zero
                        best_w_tmp = copy.deepcopy(best_w_params[arv,:,pp])
                        best_w_tmp[:,nonzero_inds_full] = numpy_utils.select_along_axis(betas[:,:,imp], lambda_inds, \
                                                                                        run_axis=2, choice_axis=0).T
                        best_w_tmp[:,~nonzero_inds_full] = 0.0 # make sure to fill zeros here
                        best_w_params[arv,:,pp] = best_w_tmp
                        
                       
                vox_loop_time += (time.time() - vox_start)
                elapsed = (time.time() - vox_start)
                sys.stdout.flush()

    # Print information about how fitting went...
    total_time = time.time() - start_time
    inv_time = total_time - vox_loop_time
    best_weights = best_w_params[:,0:max_features,:]  
    if add_bias:
        best_biases = best_w_params[:,-1,:]       
    else: 
        best_biases = None
    print ('\n---------------------------------------')
    print ('total time = %fs' % total_time)
    print ('total throughput = %fs/voxel' % (total_time / n_voxels))
    print ('voxel throughput = %fs/voxel' % (vox_loop_time / n_voxels))
    print ('setup throughput = %fs/model' % (inv_time / n_prfs))
    
    # This step clears the big feature maps for training data from feature extractor (no longer needed)
    feature_loader.clear_big_features()

    sys.stdout.flush()

    return best_losses, best_lambdas, best_weights, best_biases, best_prf_models, features_mean, features_std




def _cofactor_fn_cpu(_x, lambdas):
    '''
    Generating a matrix needed to solve ridge regression model for each lambda value.
    Ridge regression (Tikhonov) solution is :
    w = (X^T*X + I*lambda)^-1 * X^T * Y
    This func will return (X^T*X + I*lambda)^-1 * X^T. 
    So once we have that, can just multiply by training data (Y) to get weights.
    returned size is [nLambdas x nFeatures x nTrials]
    This version makes sure that the torch inverse operation is done on the cpu, and in 
    floating point-64 precision. 
    Otherwise it gives a numerically different result than numpy operations.
    
    '''
    device_orig = _x.device
    type_orig = _x.dtype
    # switch to this specific format which works with inverse
    _x = _x.to('cpu').to(torch.float64)
    
    try: 
        _f = torch.stack([(torch.mm(torch.t(_x), _x) + \
                           torch.eye(_x.size()[1], device='cpu', dtype=torch.float64) * l).inverse() \
                          for l in lambdas], axis=0) 
        
    except RuntimeError:
        # problem with inverse - print some info to help diagnose the problem.
        # usually due to zero columns or duplicate columns.
        print('WARNING: Problem with inverse in _cofactor_fn_cpu.')
        print('Size of _x (trials x features):')
        print(_x.shape)
        print('Rank of _x:')
        print(torch.matrix_rank(_x))
        # to prevent a crash, replace 0 with a small lambda value, just temporarily
        lambdas_adjusted = copy.deepcopy(lambdas)
        lambdas_adjusted[lambdas_adjusted==0] = 10e-9
        print('Trying again with these lambda values:')
        print(lambdas_adjusted)
        _f = torch.stack([(torch.mm(torch.t(_x), _x) + \
                           torch.eye(_x.size()[1], device='cpu', dtype=torch.float64) * l).inverse() \
                          for l in lambdas_adjusted], axis=0) 

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


