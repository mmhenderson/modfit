from __future__ import division
import sys
import time
import numpy as np
import torch
import scipy.stats

from utils import numpy_utils, torch_utils, stats_utils


def validate_fwrf_model(best_params, prf_models, voxel_data, images, _feature_extractor, zscore=False,\
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
    
    if zscore:       
        print('will z-score each column')
    else:
        print('will not z-score')
    
    # val_cc is the correlation coefficient bw real and predicted responses across trials, for each voxel.
    val_cc  = np.zeros(shape=(n_voxels, n_partial_versions), dtype=dtype)
    val_r2 = np.zeros(shape=(n_voxels, n_partial_versions), dtype=dtype)

    features_each_prf = np.full(fill_value=0, shape=(n_trials, n_features_max, n_prfs), dtype=dtype)
    feature_inds_defined_each_prf = np.full(fill_value=0, shape=(n_features_max, n_prfs), dtype=bool)
    
    # Saving full trial-by-trial predictions for each voxel, each partial model.
    # Need these for stacking.
    pred_voxel_data = np.full(fill_value=0, shape=(n_trials, n_voxels, n_partial_versions), dtype=dtype)
    
    start_time = time.time()    
    with torch.no_grad(): # make sure local gradients are off to save memory
        
        # First gather features for all pRFs. There are fewer pRFs than voxels, so it is faster
        # to loop over pRFs first, then voxels.
        
        _feature_extractor.clear_big_features()
        
        for mm in range(n_prfs):
            if mm>1 and debug:
                break
            print('Getting features for prf %d: [x,y,sigma] is [%.2f %.2f %.4f]'%(mm, prf_models[mm,0],  prf_models[mm,1],  prf_models[mm,2] ))
            # all_feat_concat is size [ntrials x nfeatures]
            # nfeatures may be less than n_features_max, because n_features_max is the largest number possible for any pRF.
            # feature_inds_defined is length max_features, and tells which of the features in max_features are includes in features.
            all_feat_concat, feature_inds_defined = _feature_extractor(images, prf_models[mm,:], mm, fitting_mode=False)
            
            all_feat_concat = torch_utils.get_value(all_feat_concat)
            if zscore:
                # using mean and std that were computed on training set during fitting - keeping 
                # these pars constant here seems to improve fits. 
                tiled_mean = np.tile(features_mt[mm,feature_inds_defined], [n_trials, 1])
                tiled_std = np.tile(features_st[mm,feature_inds_defined], [n_trials, 1])
                all_feat_concat = (all_feat_concat - tiled_mean)/tiled_std
                # if any entries in std are zero or nan, this gives bad result - fix these now.
                # these bad entries will also be zero in weights, so doesn't matter. 
                # just want to avoid nans.
                all_feat_concat[np.isnan(all_feat_concat)] = 0.0 
                all_feat_concat[np.isinf(all_feat_concat)] = 0.0 
                        
            features_each_prf[:,feature_inds_defined,mm] = all_feat_concat
            feature_inds_defined_each_prf[:,mm] = feature_inds_defined
            
        _feature_extractor.clear_big_features()
        
        # Next looping over all voxels in batches.
        
        vv=-1
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

                
                # here is where we choose the right set of features for each voxel, based
                # on its fitted prf.
                # [trials x features x voxels]
                features_full = features_each_prf[:,:,best_model_inds[rv,pp]]
                
                # Take out the relevant features now
                features_full = features_full[:,features_to_use,:]
                # Note there may be some zeros in this matrix, if we used fewer than the 
                # max number of features.
                # But they are zero in weight matrix too, so turns out ok.

                _weights = torch_utils._to_torch(weights[rv,:,pp], device=device)   
                _weights = _weights[:, features_to_use]
                _bias = torch_utils._to_torch(bias[rv,pp], device=device)

                print('number of zeros:')
                print(np.sum(features_full[0,:,0]==0))

                print('size of weights is:')
                print(_weights.shape)

                pred_block = np.full(fill_value=0, shape=(n_trials, lv), dtype=dtype)

                # Now looping over validation set trials in batches
                for rt, lt in numpy_utils.iterate_range(0, n_trials, sample_batch_size):

                    _features = torch_utils._to_torch(features_full[rt,:,:], device=device)
                    # features is [#samples, #features, #voxels]
                    # swap dims to [#voxels, #samples, features]
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
    
    return val_cc, val_r2, pred_voxel_data, features_each_prf

def get_feature_tuning(best_params, features_each_prf, val_voxel_data_pred, debug=False):
   
    """
    Get an approximate measure of voxels' tuning for particular features, based on how correlated 
    the predicted responses of the encoding model are with the activation in each feature channel.
    """
    
    best_models, weights, bias, features_mt, features_st, best_model_inds = best_params
    n_voxels = val_voxel_data_pred.shape[1]
    n_features = features_each_prf.shape[1]
    corr_each_feature = np.zeros((n_voxels, n_features))
   
    for vv in range(n_voxels):
        
        if debug and (vv>1):
            continue
        print('computing feature tuning for voxel %d of %d, prf %d\n'%(vv, n_voxels, best_model_inds[vv,0]))
        
        # voxel's predicted response to validation set trials, based on encoding model.
        resp = val_voxel_data_pred[:,vv,0]
        # activation in the feature of interest on each trial.
        feat_act = features_each_prf[:,:,best_model_inds[vv,0]]
        
        for ff in range(n_features):
            if np.var(feat_act[:,ff])>0:
                corr_each_feature[vv,ff] = stats_utils.numpy_corrcoef_warn(resp, feat_act[:,ff])[0,1]
            else:
                corr_each_feature[vv,ff] = np.nan
                
    return corr_each_feature


def get_semantic_discrim(best_params, labels_all, unique_labels_each, val_voxel_data_pred, debug=False):
   
    """
    Measure how well voxels' predicted responses distinguish between image patches with 
    different semantic content (compute an F-statistic).
    """
    
    best_models, weights, bias, features_mt, features_st, best_model_inds = best_params
    n_voxels = val_voxel_data_pred.shape[1]

    n_sem_axes = labels_all.shape[1]
    sem_discrim_each_axis = np.zeros((n_voxels, n_sem_axes))
    sem_corr_each_axis = np.zeros((n_voxels, n_sem_axes))
    
    for vv in range(n_voxels):
        
        if debug and (vv>1):
            continue
        print('computing semantic discriminability for voxel %d of %d, prf %d\n'%(vv, n_voxels, best_model_inds[vv,0]))
        
        resp = val_voxel_data_pred[:,vv,0]
        
        for aa in range(n_sem_axes):
            
            labels = labels_all[:,aa,best_model_inds[vv,0]]
            inds2use = ~np.isnan(labels)
            
            unique_labels_actual = np.unique(labels[inds2use])
            
            if np.all(np.isin(unique_labels_each[aa], unique_labels_actual)):
                group_inds = [(labels==ll) & inds2use for ll in unique_labels_actual]
                groups = [resp[gi] for gi in group_inds]                
                fstat = stats_utils.anova_oneway_warn(groups).statistic               
                sem_discrim_each_axis[vv,aa] = fstat                
            else:                
                sem_discrim_each_axis[vv,aa] = np.nan
                
            # just for the binary categories, also getting a correlation coefficient 
            # (includes direction/sign)   
            if (len(unique_labels_each[aa])==2) and (len(unique_labels_actual)==2):                
                sem_corr_each_axis[vv,aa] = stats_utils.numpy_corrcoef_warn(\
                                        resp[inds2use],labels[inds2use])[0,1]
            else:
                sem_corr_each_axis[vv,aa] = np.nan

    return sem_discrim_each_axis, sem_corr_each_axis

