from __future__ import division
import sys
import time
import numpy as np
import copy
import torch
from cvxopt import matrix, solvers

from utils import numpy_utils, torch_utils, stats_utils


def validate_fwrf_model(best_params, prf_models, voxel_data, images, _feature_extractor, \
                                   sample_batch_size=100, voxel_batch_size=100, debug=False, dtype=np.float32):
    
    """ 
    Evaluate trained model, leaving out a subset of features at a time.
    """
    
    params = best_params
    device = _feature_extractor.device
    
    n_trials, n_voxels = len(images), len(params[0])
    n_prfs = prf_models.shape[0]
    n_features = params[1].shape[1]  
    n_voxels = np.shape(voxel_data)[1]

    best_models, weights, bias, features_mt, features_st, best_model_inds = params
    masks, partial_version_names = _feature_extractor.get_partial_versions()
    masks = np.transpose(masks)    
    n_features_max = _feature_extractor.max_features
    n_partial_versions = len(partial_version_names)
    
    # val_cc is the correlation coefficient bw real and predicted responses across trials, for each voxel.
    val_cc  = np.zeros(shape=(n_voxels, n_partial_versions), dtype=dtype)
    val_r2 = np.zeros(shape=(n_voxels, n_partial_versions), dtype=dtype)

    pred_models = np.full(fill_value=0, shape=(n_trials, n_features_max, n_prfs), dtype=dtype)
    feature_inds_defined_each_prf = np.full(fill_value=0, shape=(n_features_max, n_prfs), dtype=bool)
    
    # Saving full trial-by-trial predictions for each voxel, each partial model.
    # Need these for stacking.
    pred_voxel_data = np.full(fill_value=0, shape=(n_trials, n_voxels, n_partial_versions), dtype=dtype)
    
    start_time = time.time()    
    with torch.no_grad(): # make sure local gradients are off to save memory
        
        # First gather texture features for all pRFs.
        
        _feature_extractor.clear_big_features()
        
        for mm in range(n_prfs):
            if mm>1 and debug:
                break
            print('Getting features for prf %d: [x,y,sigma] is [%.2f %.2f %.4f]'%(mm, prf_models[mm,0],  prf_models[mm,1],  prf_models[mm,2] ))
            # all_feat_concat is size [ntrials x nfeatures]
            # nfeatures may be less than n_features_max, because n_features_max is the largest number possible for any pRF.
            # feature_inds_defined is length max_features, and tells which of the features in max_features are includes in features.
            all_feat_concat, feature_inds_defined = _feature_extractor(images, prf_models[mm,:], mm, fitting_mode=False)
            
            pred_models[:,feature_inds_defined,mm] = torch_utils.get_value(all_feat_concat)
            feature_inds_defined_each_prf[:,mm] = feature_inds_defined
            
        _feature_extractor.clear_big_features()
        
        vv=-1
        ## Looping over voxels here in batches, will eventually go through all.
        for rv, lv in numpy_utils.iterate_range(0, n_voxels, voxel_batch_size):
            vv=vv+1
            print('Getting predictions for voxels [%d-%d] of %d'%(rv[0],rv[-1],n_voxels))

            if vv>1 and debug:
                break

            # Looping over versions of model w different features set to zero (variance partition)
            for pp in range(n_partial_versions):

                print('\nEvaluating version %d of %d: %s'%(pp, n_partial_versions, partial_version_names[pp]))

                # masks describes the indices of the features that are included in this partial model
                # n_features_max in length
                features_to_use = masks[:,pp]==1
                print('Includes %d features'%np.sum(features_to_use))

                # [trials x features x voxels]
                features_full = pred_models[:,:,best_model_inds[rv,pp]]
                # Take out the relevant features now
                features_full = features_full[:,features_to_use,:]
                # Note there may be some zeros in this matrix, if we used fewer than the max number of features.
                # But they are zero in weight matrix too, so turns out ok.

                _weights = torch_utils._to_torch(weights[rv,:,pp], device=device)   
                _weights = _weights[:, features_to_use]
                _bias = torch_utils._to_torch(bias[rv,pp], device=device)

                print('number of zeros:')
                print(np.sum(features_full[0,:,0]==0))

                print('size of weights is:')
                print(_weights.shape)

                if features_mt is not None:
                    _features_m = torch_utils._to_torch(features_mt[rv,:], device=device)
                    _features_m = _features_m[:,features_to_use]
                if features_st is not None:
                    _features_s = torch_utils._to_torch(features_st[rv,:], device=device)
                    _features_s = _features_s[:,features_to_use]

                pred_block = np.full(fill_value=0, shape=(n_trials, lv), dtype=dtype)

                # Now looping over validation set trials in batches
                for rt, lt in numpy_utils.iterate_range(0, n_trials, sample_batch_size):

                    _features = torch_utils._to_torch(features_full[rt,:], device=device) # trials x features
                    if features_mt is not None:    
                        # features_m is [nvoxels x nfeatures] - need [trials x features x voxels]
                        _features = _features - torch.tile(torch.unsqueeze(_features_m, dim=0), [_features.shape[0], 1, 1]).moveaxis([1],[2])

                    if features_st is not None:
                        _features = _features/torch.tile(torch.unsqueeze(_features_s, dim=0), [_features.shape[0], 1, 1]).moveaxis([1],[2])
                        # if any entries in std are zero or nan, this gives bad result - fix these now.
                        # these bad entries will also be zero in weights, so doesn't matter. just want to avoid nans.
                        _features[torch.isnan(_features)] = 0.0 
                        _features[torch.isinf(_features)] = 0.0
                        
                    # features is [#samples, #features, #voxels] - swap dims to [#voxels, #samples, features]
                    _features = torch.transpose(torch.transpose(_features, 0, 2), 1, 2)
                    # weights is [#voxels, #features]
                    # _r will be [#voxels, #samples, 1] - then [#samples, #voxels]

                    _r = torch.squeeze(torch.bmm(_features, torch.unsqueeze(_weights, 2)), dim=2).t() 

                    if _bias is not None:
                        _r = _r + torch.tile(torch.unsqueeze(_bias, 0), [_r.shape[0],1])

                    pred_block[rt] = torch_utils.get_value(_r) 
                
                # Making sure to save these so that we can get stacking performance later.
                pred_voxel_data[:,rv,pp] = pred_block
                
                # Now for this batch of voxels and this partial version of the model, measure performance.
                val_cc[rv,pp] = stats_utils.get_corrcoef(voxel_data[:,rv], pred_block)
                val_r2[rv,pp] = stats_utils.get_r2(voxel_data[:,rv], pred_block)

                sys.stdout.flush()

    # any nans become zeros here.
    val_cc = np.nan_to_num(val_cc)
    val_r2 = np.nan_to_num(val_r2) 
    
    return val_cc, val_r2, pred_voxel_data


def run_stacking(_feature_extractor, trn_holdout_voxel_data, val_voxel_data, trn_holdout_voxel_data_pred, val_voxel_data_pred, debug=False):   
    """
    Get data organized to run stacking code (stacked_core)
    """
    n_voxels = trn_holdout_voxel_data.shape[1]

    # To get the "features" to use for stacking - i'm using the "partial models" that are defined 
    # by the feature extractor. The first one is the full model, so we don't want to use that - just want 
    # the ones that include just a subset of the full feature space.
    partial_masks, partial_version_names = _feature_extractor.get_partial_versions()
    partial_models_use = []
    hasattr(_feature_extractor, 'module_names')
    for mm in range(len(_feature_extractor.module_names)):
        this_module = np.where([(_feature_extractor.module_names[mm] in pp) for pp in partial_version_names])[0]
        if len(this_module)>1:
            # this means there are 'subsets' of features within this module that we will want to consider separately.
            # so finding just the ones that we want here.
            partial_models_use += list(np.where([(_feature_extractor.module_names[mm] in pp and '_just' in pp and '_no_other_modules' in pp) \
                         for pp in partial_version_names])[0])
        else:
            partial_models_use += list(this_module)
    print('Subsets of features that are going into the stacking analysis:')
    print([partial_version_names[pp] for pp in partial_models_use])

    n_feature_groups = len(partial_models_use)

    # Creating a list where each element is predictions for one of the partial models - these will be 
    # the 'features' elements input to stacking code.
    preds_train = [trn_holdout_voxel_data_pred[:,:,pp].T for pp in partial_models_use]
    preds_val = [val_voxel_data_pred[:,:,pp] for pp in partial_models_use]
    # Compute trial-wise training errors
    # each element of err is [ntrials x nvoxels]
    train_err = [trn_holdout_voxel_data - trn_holdout_voxel_data_pred[:,:,pp].T for pp in partial_models_use]

    # Also computing the performance of each of the partial versions on training set data.
    # this is sort of a sanity check that things are working, since the performance of the partial models
    # should roughly (?)  predict what the stacking weights will be.
    train_r2 = np.array([stats_utils.get_r2(trn_holdout_voxel_data, trn_holdout_voxel_data_pred[:,:,pp].T) \
                         for pp in range(len(partial_version_names))]).T
    train_r2 = np.nan_to_num(train_r2)
    train_cc = np.array([stats_utils.get_corrcoef(trn_holdout_voxel_data, trn_holdout_voxel_data_pred[:,:,pp].T) \
                         for pp in range(len(partial_version_names))]).T
    train_cc = np.nan_to_num(train_cc)

    # First running stacking w all features included
    feat_use = np.arange(0,n_feature_groups)
    # Stack result will be a tuple including the stacking weights, performance.
    stack_result = stacked_core(feat_use, train_err, train_data=trn_holdout_voxel_data,\
                     val_data = val_voxel_data, preds_train = preds_train, preds_val = preds_val,\
                     debug=debug);

    # Then going to repeat it leaving out one feature group at a time
    # This will only make sense to do there are more than 2 feature groups, otherwise it's just single models.
    if n_feature_groups>2:   
        stack_result_lo = dict()
        for leave_one in range(n_feature_groups):
            feat_use_lo = list(copy.deepcopy(feat_use))
            feat_use_lo.remove(leave_one)
            tmp = stacked_core(feat_use_lo, train_err, train_data=trn_holdout_voxel_data,\
                             val_data = val_voxel_data, preds_train = preds_train, preds_val = preds_val,\
                             debug=debug);
            stack_result_lo[leave_one] = tmp
    else:       
        stack_result_lo = None


    return stack_result, stack_result_lo, partial_models_use, train_r2, train_cc


def stacked_core(feat_use, train_err, train_data, val_data, preds_train, preds_val, debug=False):
    """
    Compute weights for stacking models (linearly combining predictions of multiple encoding models).
    Outputs weights and performance of the stacked model.
    Code from Ruogu Lin (modified slightly for this project).
    """
    
    solvers.options["show_progress"] = False

    print('Running stacking, feat_use is:')
    print(feat_use)
    n_voxels = train_data.shape[1]
    n_feature_groups = len(feat_use) # feat use is the sub-set of feature groups to stack.
    n_trials_train = train_data.shape[0]
    n_trials_val = preds_val[0].shape[0]
        
    dtype = train_data.dtype
    stacked_pred_train = np.full(fill_value=0, shape=(n_trials_train, n_voxels), dtype=dtype)
    stacked_pred_val = np.full(fill_value=0, shape=(n_trials_val, n_voxels), dtype=dtype)

    # calculate error matrix for stacking
    P = np.zeros((n_voxels, n_feature_groups, n_feature_groups))
    idI = 0
    for i in feat_use:
        idJ = 0
        for j in feat_use:
            # err is the trialwise, voxelwise, error for each model.
            # P will store the summed products of the error for each pair of models 
            # (if i=j, then it's the summed squared error).
            P[:, idI, idJ] = np.mean(train_err[i] * train_err[j], 0)
            idJ += 1
        idI += 1

    idI = 0
    idJ = 0

    # PROGRAMATICALLY SET THIS FROM THE NUMBER OF FEATURES
    q = matrix(np.zeros((n_feature_groups)))
    G = matrix(-np.eye(n_feature_groups, n_feature_groups))
    h = matrix(np.zeros(n_feature_groups))
    A = matrix(np.ones((1, n_feature_groups)))
    b = matrix(np.ones(1))

    # Stacking weights will be stored here
    S = np.zeros((n_voxels, n_feature_groups))

    for vv in range(0, n_voxels):
        if debug and vv>1:
            continue
            
        print('Solving for stacking weights for voxel %d of %d'%(vv, n_voxels))
        PP = matrix(P[vv])
        # solve for stacking weights for every voxel
        # This essentially is minimizing the quantity x.T @ PP @ x, subject to the constraint that
        # the elements of x have to be positive, and have to sum to 1. 
        # x will be the weights for the stacking model.
        # Weights will be dependent on the error of each model individually (this is contained in PP).
        S[vv, :] = np.array(solvers.qp(PP, q, G, h, A, b)["x"]).reshape(n_feature_groups,)
        if vv==0:
            print('Stacking weights matrix is size:')
            print(S.shape)
            
        # Combine the predictions from the individual feature spaces for voxel i
        z = np.array([preds_val[feature_j][:, vv] for feature_j in feat_use])
        # multiply the predictions by S[vv,:]
        stacked_pred_val[:, vv] = np.dot(S[vv, :], z)
        
        # Same thing for the training trials
        z = np.array([preds_train[feature_j][:, vv] for feature_j in feat_use])
        stacked_pred_train[:, vv] = np.dot(S[vv, :], z)
        
        sys.stdout.flush()
        
    print('Computing performance of stacked models')
    # Compute r2 of the stacked model for training data
    stacked_r2_train = stats_utils.get_r2(train_data, stacked_pred_train)
    stacked_cc_train = stats_utils.get_corrcoef(train_data, stacked_pred_train)
    stacked_r2_train = np.nan_to_num(stacked_r2_train)
    stacked_cc_train = np.nan_to_num(stacked_cc_train) 
    
    # And for validation data
    stacked_r2_val = stats_utils.get_r2(val_data, stacked_pred_val)
    stacked_cc_val = stats_utils.get_corrcoef(val_data, stacked_pred_val)
    stacked_r2_val = np.nan_to_num(stacked_r2_val)
    stacked_cc_val = np.nan_to_num(stacked_cc_val) 
    
    return S, stacked_r2_train, stacked_cc_train, stacked_r2_val, stacked_cc_val

