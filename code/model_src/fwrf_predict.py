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
 

def validate_texture_model_varpart(best_params, prf_models, val_voxel_single_trial_data, val_stim_single_trial_data, _texture_fn, sample_batch_size=100, voxel_batch_size=100, debug=False, dtype=np.float32):
    
    """ 
    Evaluate trained model, leaving out a subset of features at a time.
    """
    images = val_stim_single_trial_data
    params = best_params
    dtype = images.dtype.type
    device = _texture_fn.device
    
    n_trials, n_voxels = len(images), len(params[0])
    n_prfs = prf_models.shape[0]
    n_features = params[1].shape[1]  

    best_models, weights, bias, features_mt, features_st, best_model_inds, partial_version_names = params
    
    # val_cc is the correlation coefficient bw real and predicted responses across trials, for each voxel.
    n_voxels = np.shape(val_voxel_single_trial_data)[1]
    n_features_total = _texture_fn.n_features_total
    n_feature_types = len(_texture_fn.feature_types_include)
    n_partial_versions = len(partial_version_names)
    if n_partial_versions>1:
        masks = np.concatenate([np.expand_dims(np.array(_texture_fn.feature_column_labels!=ff).astype('int'), axis=0) for ff in np.arange(-1,n_feature_types)], axis=0)
    else:
        masks = np.ones([1,n_features_total])
    # "partial versions" will be listed as: [full model, leave out first set of features, leave out second set of features...]

    masks = np.transpose(masks)

    val_cc  = np.zeros(shape=(n_voxels, n_partial_versions), dtype=dtype)
    val_r2 = np.zeros(shape=(n_voxels, n_partial_versions), dtype=dtype)

    pred_models = np.full(fill_value=0, shape=(n_trials, n_features, n_prfs), dtype=dtype)
    
    start_time = time.time()    
    with torch.no_grad():
        
        # First gather texture features for all pRFs.
        
        _texture_fn.clear_maps()
        
        for mm in range(n_prfs):
            if mm>1 and debug:
                break
            print('Getting features for prf %d: [x,y,sigma] is [%.2f %.2f %.4f]'%(mm, prf_models[mm,0],  prf_models[mm,1],  prf_models[mm,2] ))
            all_feat_concat, feature_info = _texture_fn(images,prf_models[mm,:])
            
            pred_models[:,:,mm] = torch_utils.get_value(all_feat_concat)
        
        _texture_fn.clear_maps()
    
        vv=-1
        ## Looping over voxels here in batches, will eventually go through all.
        for rv, lv in numpy_utils.iterate_range(0, n_voxels, voxel_batch_size):
            vv=vv+1
            print('Getting predictions for voxels [%d-%d] of %d'%(rv[0],rv[-1],n_voxels))

            if vv>1 and debug:
                break
            
            # Looping over versions of model w different features set to zero (variance partition)
            for pp in range(n_partial_versions):
                
                print('Evaluating version %d of %d: %s'%(pp, n_partial_versions, partial_version_names[pp]))
   
                # [trials x features x voxels]
                features_full = pred_models[:,:,best_model_inds[rv,pp]]
                           
                nonzero_inds = masks[:,pp]==1
                
                features = features_full[:,nonzero_inds,:]

                # making sure to gather only the columns for features included in this partial model
                _weights = torch_utils._to_torch(weights[rv,:,pp][:,nonzero_inds]) 
                
                _bias = torch_utils._to_torch(bias[rv,pp])

            
                if features_mt is not None:
                    _features_m = torch_utils._to_torch(features_mt[rv,:][:,nonzero_inds])
                if features_st is not None:
                    _features_s = torch_utils._to_torch(features_st[rv,:][:,nonzero_inds])
                
                pred_block = np.full(fill_value=0, shape=(n_trials, lv), dtype=dtype)
                
                
                # Now looping over validation set trials in batches
                for rt, lt in numpy_utils.iterate_range(0, n_trials, sample_batch_size):

                    _features = torch_utils._to_torch(features[rt,:,:]) # trials x features x voxels
                    if features_mt is not None:    
                        # features_m is [nvoxels x nfeatures] - need [trials x features x voxels]
                        _features = _features - torch.tile(torch.unsqueeze(_features_m, dim=0), [_features.shape[0], 1, 1]).moveaxis([1],[2])

                    if features_st is not None:
                        _features = _features/torch.tile(torch.unsqueeze(_features_s, dim=0), [_features.shape[0], 1, 1]).moveaxis([1],[2])
                        _features[torch.isnan(_features)] = 0.0 # this applies in the pca case when last few columns of features are missing

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
                    val_cc[rv[vi],pp] = np.corrcoef(val_voxel_single_trial_data[:,rv[vi]], pred_block[:,vi])[0,1]  
                    val_r2[rv[vi],pp] = get_r2(val_voxel_single_trial_data[:,rv[vi]], pred_block[:,vi])
                
                sys.stdout.flush()
        
    val_cc = np.nan_to_num(val_cc)
    val_r2 = np.nan_to_num(val_r2) 
    
    return val_cc, val_r2






def validate_texture_model(best_params, prf_models, val_voxel_single_trial_data, val_stim_single_trial_data, _texture_fn, sample_batch_size=100, voxel_batch_size=100, debug=False, dtype=np.float32):
    
    # EVALUATE PERFORMANCE ON VALIDATION SET

    print('\nGetting model predictions on validation set...\n')
    val_voxel_pred = get_predictions_texture_model(val_stim_single_trial_data, _texture_fn, best_params, prf_models, sample_batch_size=sample_batch_size, voxel_batch_size=voxel_batch_size, debug=debug)

    # val_cc is the correlation coefficient bw real and predicted responses across trials, for each voxel.
    n_voxels = np.shape(val_voxel_single_trial_data)[1]
    val_cc  = np.zeros(shape=(n_voxels), dtype=dtype)
    val_r2 = np.zeros(shape=(n_voxels), dtype=dtype)
    
    print('\nEvaluating correlation coefficient on validation set...\n')
    for v in tqdm(range(n_voxels)):    
        val_cc[v] = np.corrcoef(val_voxel_single_trial_data[:,v], val_voxel_pred[:,v])[0,1]  
        val_r2[v] = get_r2(val_voxel_single_trial_data[:,v], val_voxel_pred[:,v])
        
    val_cc = np.nan_to_num(val_cc)
    val_r2 = np.nan_to_num(val_r2)    
    
    return val_cc, val_r2


def validate_texture_model_partial(best_params, prf_models, val_voxel_single_trial_data, val_stim_single_trial_data, _texture_fn, sample_batch_size=100, voxel_batch_size=100, debug=False, dtype=np.float32):
    
    """ 
    Evaluate trained model, leaving out a subset of features at a time.
    """
    
    # val_cc is the correlation coefficient bw real and predicted responses across trials, for each voxel.
    n_voxels = np.shape(val_voxel_single_trial_data)[1]
    n_feature_types = len(_texture_fn.feature_types_include)
    val_cc  = np.zeros(shape=(n_voxels, n_feature_types), dtype=dtype)
    val_r2 = np.zeros(shape=(n_voxels, n_feature_types), dtype=dtype)

    orig_feature_column_labels = _texture_fn.feature_column_labels
    orig_excluded_features = _texture_fn.feature_types_exclude

    for ff, feat_name in enumerate(_texture_fn.feature_types_include):

        print('\nVariance partition, leaving out: %s'%feat_name)
        _texture_fn.update_feature_list(orig_excluded_features+[feat_name])
        print('Remaining features are:')
        print(_texture_fn.feature_types_include)

        # Choose columns of interest here, leaving out weights for one feature at a time
        params_to_use = copy.deepcopy(best_params)
        columns_to_use = np.where(orig_feature_column_labels!=ff)[0]
        print(columns_to_use)
        params_to_use[1] = params_to_use[1][:,columns_to_use]
        if best_params[3] is not None:
            params_to_use[3] = params_to_use[3][:,columns_to_use]
            params_to_use[4] = params_to_use[4][:,columns_to_use]

        print(best_params[1].shape)
        print(params_to_use[1].shape)
        print('\nGetting model predictions on validation set...\n')
#         val_voxel_pred = get_predictions_texture_model(val_stim_single_trial_data, _texture_fn, best_params, prf_models, sample_batch_size=sample_batch_size, voxel_batch_size=voxel_batch_size, debug=debug)
        val_voxel_pred = get_predictions_texture_model(val_stim_single_trial_data, _texture_fn, params_to_use, prf_models, sample_batch_size=sample_batch_size, voxel_batch_size=voxel_batch_size, debug=debug)
        print('\nEvaluating correlation coefficient on validation set...\n')
        for v in range(n_voxels):    
            val_cc[v,ff] = np.corrcoef(val_voxel_single_trial_data[:,v], val_voxel_pred[:,v])[0,1]  
            val_r2[v,ff] = get_r2(val_voxel_single_trial_data[:,v], val_voxel_pred[:,v])

    val_cc = np.nan_to_num(val_cc)
    val_r2 = np.nan_to_num(val_r2) 
    
    return val_cc, val_r2



def get_predictions_texture_model(images, _texture_fn, params, prf_models, sample_batch_size=100, voxel_batch_size=100, debug=False):
   
    dtype = images.dtype.type
    device = _texture_fn.device

    best_models, weights, bias, features_mt, features_st, best_model_inds = params
        
    n_trials, n_voxels = len(images), len(params[0])
    n_prfs = prf_models.shape[0]
    n_features = params[1].shape[1]    
    
    pred = np.full(fill_value=0, shape=(n_trials, n_voxels), dtype=dtype)
    pred_models = np.full(fill_value=0, shape=(n_trials, n_features, n_prfs), dtype=dtype)
    
    start_time = time.time()    
    with torch.no_grad():
        
        # First gather texture features for all pRFs.
        for mm in range(n_prfs):
            if mm>1 and debug:
                break
            print('Getting features for prf %d: [x,y,sigma] is [%.2f %.2f %.4f]'%(mm, prf_models[mm,0],  prf_models[mm,1],  prf_models[mm,2] ))
            all_feat_concat, feature_info = _texture_fn(images,prf_models[mm,:])
            
            pred_models[:,:,mm] = torch_utils.get_value(all_feat_concat)
        
        vv=-1
        ## Looping over voxels here in batches, will eventually go through all.
        for rv, lv in numpy_utils.iterate_range(0, n_voxels, voxel_batch_size):
            vv=vv+1
            print('Getting predictions for voxels [%d-%d] of %d'%(rv[0],rv[-1],n_voxels))

            if vv>1 and debug:
                break
                
            # [trials x features x voxels]
            features = pred_models[:,:,best_model_inds[rv]]

            pred_block = np.full(fill_value=0, shape=(n_trials, lv), dtype=dtype)
            if features_mt is not None:
                _features_m = torch_utils._to_torch(features_mt[rv,:])
            if features_st is not None:
                _features_s = torch_utils._to_torch(features_st[rv,:])
            _weights = torch_utils._to_torch(weights[rv,:])
            _bias = torch_utils._to_torch(bias[rv])
                
            # Now looping over validation set trials in batches
            for rt, lt in numpy_utils.iterate_range(0, n_trials, sample_batch_size):

                _features = torch_utils._to_torch(features[rt,:,:]) # trials x features x voxels
                if features_mt is not None:    
                    # features_m is [nvoxels x nfeatures] - need [trials x features x voxels]
                    _features = _features - torch.tile(torch.unsqueeze(_features_m, dim=0), [_features.shape[0], 1, 1]).moveaxis([1],[2])

                if features_st is not None:
                    _features = _features/torch.tile(torch.unsqueeze(_features_s, dim=0), [_features.shape[0], 1, 1]).moveaxis([1],[2])
                    _features[torch.isnan(_features)] = 0.0 # this applies in the pca case when last few columns of features are missing

                # features is [#samples, #features, #voxels] - swap dims to [#voxels, #samples, features]
                _features = torch.transpose(torch.transpose(_features, 0, 2), 1, 2)
                # weights is [#voxels, #features]
                # _r will be [#voxels, #samples, 1] - then [#samples, #voxels]

                _r = torch.squeeze(torch.bmm(_features, torch.unsqueeze(_weights, 2)), dim=2).t() 

                if _bias is not None:
                    _r = _r + torch.tile(torch.unsqueeze(_bias, 0), [_r.shape[0],1])

                pred_block[rt] = torch_utils.get_value(_r) 
                
            pred[:,rv] = pred_block
            sys.stdout.flush()
            
    total_time = time.time() - start_time
    print ('\n---------------------------------------')
    print ('total time = %fs' % total_time)
    print ('sample throughput = %fs/sample' % (total_time / n_trials))
    print ('voxel throughput = %fs/voxel' % (total_time / n_voxels))
    sys.stdout.flush()
    return pred

   
def validate_bdcn_model(best_params, prf_models, val_voxel_single_trial_data, \
                            val_stim_single_trial_data, _feature_extractor, pc=None, sample_batch_size=100, \
                            voxel_batch_size=100, debug=False, dtype=np.float32):
    
    """ 
    Evaluate trained model, leaving out a subset of features at a time.
    """
    print('starting validation function')
    sys.stdout.flush()
        
    images = val_stim_single_trial_data
    params = best_params
    dtype = images.dtype.type
    device = _feature_extractor.device
    
    n_trials, n_voxels = len(images), len(params[0])
    n_prfs = prf_models.shape[0]
    n_features = params[1].shape[1]  

    best_models, weights, bias, features_mt, features_st, best_model_inds = params
    if pc is not None:
        pca_wts, pct_var_expl, min_pct_var, n_comp_needed, pca_pre_mean = pc
    
    n_voxels = np.shape(val_voxel_single_trial_data)[1]
    print('starting to initialize big arrays')
    sys.stdout.flush()
        
    val_cc  = np.zeros(shape=(n_voxels, 1), dtype=dtype)
    val_r2 = np.zeros(shape=(n_voxels, 1), dtype=dtype)
    n_features_actual = np.zeros(shape=(n_prfs,), dtype=int)

    pred_models = np.full(fill_value=0, shape=(n_trials, n_features, n_prfs), dtype=dtype)
    
    print('about to start loop')
    sys.stdout.flush()
        
    start_time = time.time()    
    with torch.no_grad():
        
        # First gather features for all pRFs.
        
        _feature_extractor.clear_maps()
        
        sys.stdout.flush()
        
        
        for mm in range(n_prfs):
            if mm>1 and debug:
                break
            print('Getting features for prf %d: [x,y,sigma] is [%.2f %.2f %.4f]'%(mm, prf_models[mm,0],  prf_models[mm,1],  prf_models[mm,2] ))
            
            features = _feature_extractor(images, prf_models[mm,:]).detach().cpu().numpy()   

            if pc is not None:
                print('Applying pre-computed PCA matrix')
                # Apply the PCA transformation, just as it was done during training
                nfeat = features.shape[1]
                features_submean = features - np.tile(np.expand_dims(pca_pre_mean[mm][0:nfeat], axis=0), [n_trials, 1])
                features_reduced = features_submean @ np.transpose(pca_wts[mm][0:n_comp_needed[mm],0:nfeat])                                               
                features = features_reduced

            n_features_actual[mm] = features.shape[1]

            pred_models[:,0:n_features_actual[mm],mm] = features
            
            sys.stdout.flush()
        
                
        _feature_extractor.clear_maps()
        
        sys.stdout.flush()
        
    
        vv=-1
        ## Looping over voxels here in batches, will eventually go through all.
        for rv, lv in numpy_utils.iterate_range(0, n_voxels, voxel_batch_size):
            vv=vv+1
            print('Getting predictions for voxels [%d-%d] of %d'%(rv[0],rv[-1],n_voxels))

            if vv>1 and debug:
                break
            
            # [trials x features x voxels]
            # to keep this from being huge, just keep the maximum number of features needed for any voxel
            # there will be some zeros still, but they are also zero in the weights so not a problem.
            feat2use = np.max(n_features_actual[best_model_inds[rv]])
            features = pred_models[:,0:feat2use,best_model_inds[rv]]

            print('size of feature matrix to use is:')
            print(features.shape)
            
            _weights = torch_utils._to_torch(weights[rv,0:feat2use]) 
            _bias = torch_utils._to_torch(bias[rv])

            if features_mt is not None:
                _features_m = torch_utils._to_torch(features_mt[rv,0:feat2use])
            if features_st is not None:
                _features_s = torch_utils._to_torch(features_st[rv,0:feat2use])

            pred_block = np.full(fill_value=0, shape=(n_trials, lv), dtype=dtype)

            # Now looping over validation set trials in batches
            for rt, lt in numpy_utils.iterate_range(0, n_trials, sample_batch_size):

                _features = torch_utils._to_torch(features[rt,:,:]) # trials x features x voxels
                if features_mt is not None:    
                    # features_m is [nvoxels x nfeatures] - need [trials x features x voxels]
                    _features = _features - torch.tile(torch.unsqueeze(_features_m, dim=0), [_features.shape[0], 1, 1]).moveaxis([1],[2])

                if features_st is not None:
                    _features = _features/torch.tile(torch.unsqueeze(_features_s, dim=0), [_features.shape[0], 1, 1]).moveaxis([1],[2])
                    _features[torch.isnan(_features)] = 0.0 # this applies in the pca case when last few columns of features are missing

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
                val_cc[rv[vi],0] = np.corrcoef(val_voxel_single_trial_data[:,rv[vi]], pred_block[:,vi])[0,1]  
                val_r2[rv[vi],0] = get_r2(val_voxel_single_trial_data[:,rv[vi]], pred_block[:,vi])

            sys.stdout.flush()
        
    val_cc = np.nan_to_num(val_cc)
    val_r2 = np.nan_to_num(val_r2) 
    
    return val_cc, val_r2

    
