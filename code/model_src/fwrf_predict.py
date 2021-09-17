import sys
import os
import struct
import time
import numpy as np
import tqdm
import copy

import torch

from utils import numpy_utils, torch_utils


def get_r2(actual,predicted):
  
    """
    This computes the coefficient of determination (R2).
    For OLS, this is a good measure of variance explained. 
    Not necessarily true for ridge regression - can use signed correlation coefficient^2 instead.
    With OLS & when train/test sets are identical, R2 = correlation coefficient^2.
    """
    
    # calculate r2 for this fit.
    ssres = np.sum(np.power((predicted - actual),2));
    sstot = np.sum(np.power((actual - np.mean(actual)),2));
    r2 = 1-(ssres/sstot)
    
    return r2
 
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
    
    start_time = time.time()    
    with torch.no_grad(): # make sure local gradients are off to save memory
        
        # First gather texture features for all pRFs.
        
        _feature_extractor.clear_maps()
        
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
            
        _feature_extractor.clear_maps()
        
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

                # Now for this batch of voxels and this partial version of the model, measure performance.
        #                 print('\nEvaluating correlation coefficient on validation set...\n')
                for vi in range(lv):   
                    val_cc[rv[vi],pp] = np.corrcoef(voxel_data[:,rv[vi]], pred_block[:,vi])[0,1]  
                    val_r2[rv[vi],pp] = get_r2(voxel_data[:,rv[vi]], pred_block[:,vi])

                sys.stdout.flush()

    val_cc = np.nan_to_num(val_cc)
    val_r2 = np.nan_to_num(val_r2) 
    
    return val_cc, val_r2
