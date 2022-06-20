from __future__ import division
import sys
import time
import numpy as np
import torch
import scipy.stats

from utils import numpy_utils, torch_utils, stats_utils


def get_feature_tuning(best_prf_inds, features_each_prf, val_voxel_data_pred, \
                       trials_use_each_prf = None, debug=False):
   
    """
    Get an approximate measure of voxels' tuning for particular features, based on how correlated 
    the predicted responses of the encoding model are with the activation in each feature channel.
    """
    
    n_trials = val_voxel_data_pred.shape[0]
    n_voxels = val_voxel_data_pred.shape[1]
    n_features = features_each_prf.shape[1]
    corr_each_feature = np.zeros((n_voxels, n_features))
   
    for vv in range(n_voxels):
        
        if debug and (vv>1):
            continue
        print('computing feature tuning for voxel %d of %d, prf %d\n'%(vv, n_voxels, best_prf_inds[vv,0]))
           
        # voxel's predicted response to validation set trials, based on encoding model.
        resp = val_voxel_data_pred[:,vv,0]
        # activation in the feature of interest on each trial.
        feat_act = features_each_prf[:,:,best_prf_inds[vv,0]]
        
        if trials_use_each_prf is not None:
            # select subset of trials to work with
            trials_use = trials_use_each_prf[:,best_prf_inds[vv,0]]
            resp = resp[trials_use]
            feat_act = feat_act[trials_use]
            if np.sum(trials_use)==0:
                print('voxel %d: no trials are included here, skipping it'%vv)
                corr_each_feature[vv,:] = np.nan
                continue
        
        for ff in range(n_features):
            if np.var(feat_act[:,ff])>0:
                corr_each_feature[vv,ff] = stats_utils.numpy_corrcoef_warn(resp, feat_act[:,ff])[0,1]
            else:
                corr_each_feature[vv,ff] = np.nan
                
    return corr_each_feature


def get_features_each_prf(prf_models, image_inds_val, \
                        feature_loader, zscore=False, debug=False,
                        dtype=np.float32):
    
    """ 
    Just loads the features in each pRF on each trial. 
    Only used in special case when evaluating feature "tuning" of raw voxel data.
    """
    
    params = best_params
  
    n_trials = len(image_inds_val)
    
    n_prfs = prf_models.shape[0]
    
    n_features_max = feature_loader.max_features
   
    features_each_prf = np.full(fill_value=0, shape=(n_trials, n_features_max, n_prfs), dtype=dtype)
    
    start_time = time.time()    
    with torch.no_grad(): # make sure local gradients are off to save memory
        
        # First looping over pRFs - there are fewer pRFs than voxels, so this will be faster 
        # than looping over voxels first would be.
        feature_loader.clear_big_features()
        
        for mm in range(n_prfs):
            if mm>1 and debug:
                break
          
            # all_feat_concat is size [ntrials x nfeatures] (where nfeatures can be <max_features)
            # feature_inds_defined is [max_features]
            all_feat_concat, feature_inds_defined = feature_loader.load(image_inds_val, mm, fitting_mode=False)
            
            if zscore:
                m = np.mean(all_feat_concat, axis=0)
                s = np.std(all_feat_concat, axis=0)
                all_feat_concat = (all_feat_concat - m)/s
                assert(not np.any(np.isnan(all_feat_concat)) and not np.any(np.isinf(all_feat_concat)))
                  
            # saving all these features for use later on
            features_each_prf[:,feature_inds_defined,mm] = all_feat_concat
            
                    
    return features_each_prf
