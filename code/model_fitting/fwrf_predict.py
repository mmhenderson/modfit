from __future__ import division
import sys
import time
import numpy as np
import torch
import scipy.stats

from utils import numpy_utils, torch_utils, stats_utils


def validate_fwrf_model(best_params, prf_models, voxel_data, image_inds_val, \
                        feature_loader, zscore=False, sample_batch_size=100, \
                        voxel_batch_size=100, debug=False, \
                        trials_use_each_prf = None,
                        dtype=np.float32, device=None):
    
    """ 
    Evaluate trained FWRF model, using best fit pRF params and feature weights.
    Returns the trial-by-trial predictions of the encoding model for the voxels in 
    voxel_data, which can be used for further analyses (get_feature_tuning etc.),
    and computes overall val_cc and val_r2 (prediction accuracy of model).
    Also completes the variance partition analysis, by evaluating the model with 
    different held out sets of features at a time.
    """
    
    params = best_params
    if device is None:
        device=torch.device('cpu:0')
    
    n_trials, n_voxels = len(image_inds_val), len(params[0])
    n_prfs = prf_models.shape[0]
    n_features = params[1].shape[1]  
    n_voxels = np.shape(voxel_data)[1]

    best_models, weights, bias, features_mt, features_st, best_model_inds = params
    masks, partial_version_names = feature_loader.get_partial_versions()
    masks = np.transpose(masks)    
    n_features_max = feature_loader.max_features
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
        
        # First looping over pRFs - there are fewer pRFs than voxels, so this will be faster 
        # than looping over voxels first would be.
        feature_loader.clear_big_features()
        
        for mm in range(n_prfs):
            if mm>1 and debug:
                break
            print('Getting features for prf %d: [x,y,sigma] is [%.2f %.2f %.4f]'%\
                  (mm, prf_models[mm,0],  prf_models[mm,1],  prf_models[mm,2] ))
            if not np.any(best_model_inds==mm):
                print('No voxels have this pRF as their best model, skipping it.')
                continue
            else:
                voxels_to_do = np.where(best_model_inds==mm)[0]
                n_voxels_to_do = len(voxels_to_do)            
                n_voxel_batches = int(np.ceil(n_voxels_to_do/voxel_batch_size))

            # all_feat_concat is size [ntrials x nfeatures] (where nfeatures can be <max_features)
            # feature_inds_defined is [max_features]
            all_feat_concat, feature_inds_defined = feature_loader.load(image_inds_val, mm, fitting_mode=False)
            
            if zscore:
                # using mean and std that were computed on training set during fitting - keeping 
                # these pars constant here seems to improve fits. 
                tiled_mean = np.tile(features_mt[mm,feature_inds_defined], [n_trials, 1])
                tiled_std = np.tile(features_st[mm,feature_inds_defined], [n_trials, 1])
                all_feat_concat = (all_feat_concat - tiled_mean)/tiled_std
                assert(not np.any(np.isnan(all_feat_concat)) and not np.any(np.isinf(all_feat_concat)))
                        
            # saving all these features for use later on
            features_each_prf[:,feature_inds_defined,mm] = all_feat_concat
            feature_inds_defined_each_prf[:,mm] = feature_inds_defined
            
            feature_loader.clear_big_features()

            # get data ready to do validation
            features_full = features_each_prf[:,:,mm:mm+1]

            if trials_use_each_prf is not None:
                # select subset of trials to work with
                trials_use = trials_use_each_prf[:,mm]
                features_full = features_full[trials_use,:,:]
                voxel_data_use = voxel_data[trials_use,:]
                if np.sum(trials_use)==0:
                    print('prf %d: no trials are included here, skipping validation for all voxels with this pRF!'%mm)
                    val_cc[voxels_to_do,:] = np.nan
                    val_r2[voxels_to_do,:] = np.nan
                    continue
            else:
                trials_use = np.ones((n_trials,),dtype=bool)
                voxel_data_use = voxel_data
                
            n_trials_use = np.sum(trials_use)
            
                
            print(n_trials_use)
            print('prf %d: using %d validation set trials'%(mm, voxel_data_use.shape[0]))
            
            # Next looping over all voxels with this same pRF, in batches        
            for vv in range(n_voxel_batches):
                    
                vinds = np.arange(voxel_batch_size*vv, np.min([voxel_batch_size*(vv+1), n_voxels_to_do]))
                rv = voxels_to_do[vinds]
                lv = len(vinds)
                
                # double check the pRF estimates are same
                assert(np.all(best_model_inds[rv,:]==mm))

                print('Getting predictions for voxels [%d-%d], batch %d of %d'%(rv[0],rv[-1],vv, n_voxel_batches))

                if vv>1 and debug:
                    break

                # Looping over versions of model w different features set to zero (variance partition)
                for pp in range(n_partial_versions):

                    print('\nEvaluating version %d of %d: %s'%(pp, n_partial_versions, partial_version_names[pp]))

                    # masks describes the indices of the features that are included in this partial model
                    # n_features_max in length
                    features_to_use = masks[:,pp]==1
                    print('Includes %d features'%np.sum(features_to_use))
      
                    # Take out the relevant features now
                    features = np.tile(features_full[:,features_to_use,:], [1,1,lv])
                    # Note there may be some zeros in this matrix, if we used fewer than the 
                    # max number of features.
                    # But they are zero in weight matrix too, so turns out ok.

                    _weights = torch_utils._to_torch(weights[rv,:,pp], device=device)   
                    _weights = _weights[:, features_to_use]
                    _bias = torch_utils._to_torch(bias[rv,pp], device=device)

                    print('number of zeros:')
                    print(np.sum(features[0,:,0]==0))

                    print('size of weights is:')
                    print(_weights.shape)

                    pred_block = np.full(fill_value=0, shape=(n_trials_use, lv), dtype=dtype)
                    
                    # Now looping over validation set trials in batches
                    for rt, lt in numpy_utils.iterate_range(0, n_trials_use, sample_batch_size):

                        _features = torch_utils._to_torch(features[rt,:,:], device=device)
                        # features is [#samples, #features, #voxels]
                        # swap dims to [#voxels, #samples, features]
                        _features = torch.transpose(torch.transpose(_features, 0, 2), 1, 2)
                        # weights is [#voxels, #features]
                        # _r will be [#voxels, #samples, 1] - then [#samples, #voxels]
                        _r = torch.squeeze(torch.bmm(_features, torch.unsqueeze(_weights, 2)), dim=2).t() 

                        if _bias is not None:
                            _r = _r + torch.tile(torch.unsqueeze(_bias, 0), [_r.shape[0],1])

                        pred_block[rt] = torch_utils.get_value(_r) 

                    
                    # Making sure to save these for use in analyses later
                    pred_these_trials = pred_voxel_data[trials_use,:,pp]
                    pred_these_trials[:,rv] = pred_block
                    pred_voxel_data[trials_use,:,pp] = pred_these_trials

                    # Now for this batch of voxels and this partial version of the model, measure performance.
                    val_cc[rv,pp] = stats_utils.get_corrcoef(voxel_data_use[:,rv], pred_block)
                    val_r2[rv,pp] = stats_utils.get_r2(voxel_data_use[:,rv], pred_block)

                    sys.stdout.flush()

                    
    return val_cc, val_r2, pred_voxel_data, features_each_prf

def get_feature_tuning(best_params, features_each_prf, val_voxel_data_pred, \
                       trials_use_each_prf = None, debug=False):
   
    """
    Get an approximate measure of voxels' tuning for particular features, based on how correlated 
    the predicted responses of the encoding model are with the activation in each feature channel.
    """
    
    best_models, weights, bias, features_mt, features_st, best_model_inds = best_params
    n_trials = val_voxel_data_pred.shape[0]
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
        
        if trials_use_each_prf is not None:
            # select subset of trials to work with
            trials_use = trials_use_each_prf[:,best_model_inds[vv,0]]
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


def get_semantic_discrim(best_params, labels_all, unique_labels_each, val_voxel_data_pred, \
                         trials_use_each_prf = None, debug=False):
   
    """
    Measure how well voxels' predicted responses distinguish between image patches with 
    different semantic content.
    """
    
    best_models, weights, bias, features_mt, features_st, best_model_inds = best_params
    n_voxels = val_voxel_data_pred.shape[1]

    n_sem_axes = labels_all.shape[1]
    sem_discrim_each_axis = np.zeros((n_voxels, n_sem_axes))
    sem_corr_each_axis = np.zeros((n_voxels, n_sem_axes))
    
    max_categ = np.max([len(un) for un in unique_labels_each])
    n_samp_each_axis = np.zeros((n_voxels, n_sem_axes, max_categ),dtype=np.float32)
    mean_each_sem_level = np.zeros((n_voxels, n_sem_axes, max_categ),dtype=np.float32)
 
    for vv in range(n_voxels):
        
        if debug and (vv>1):
            continue
        print('computing semantic discriminability for voxel %d of %d, prf %d\n'%(vv, n_voxels, best_model_inds[vv,0]))
        
        resp = val_voxel_data_pred[:,vv,0]
        
        if trials_use_each_prf is not None:
            # select subset of trials to work with
            trials_use = trials_use_each_prf[:,best_model_inds[vv,0]]
            resp = resp[trials_use]
            labels_use = labels_all[trials_use,:,:]
            if np.sum(trials_use)==0:
                print('voxel %d: no trials are included here, skipping it'%vv)
                sem_discrim_each_axis[vv,:] = np.nan
                sem_corr_each_axis[vv,:] = np.nan
                continue
        else:
            labels_use = labels_all
       
        for aa in range(n_sem_axes):
            
            labels = labels_use[:,aa,best_model_inds[vv,0]]
            
            inds2use = ~np.isnan(labels)
            
            unique_labels_actual = np.unique(labels[inds2use])
            
            if np.all(np.isin(unique_labels_each[aa], unique_labels_actual)):
                
                # separate trials into those with the different labels for this semantic axis.
                group_inds = [(labels==ll) & inds2use for ll in unique_labels_actual]
                groups = [resp[gi] for gi in group_inds]
                
                if len(unique_labels_actual)==2:
                    # use t-statistic as a measure of discriminability
                    # larger pos value means resp[label==1] > resp[label==0]
                    sem_discrim_each_axis[vv,aa] = stats_utils.ttest_warn(groups[1], groups[0]).statistic
                else:
                    # if more than 2 classes, computing an F statistic 
                    sem_discrim_each_axis[vv,aa] = stats_utils.anova_oneway_warn(groups).statistic
                # also computing a correlation coefficient between semantic label/voxel response
                # sign is consistent with t-statistic
                sem_corr_each_axis[vv,aa] = stats_utils.numpy_corrcoef_warn(\
                                        resp[inds2use],labels[inds2use])[0,1]
                for gi, gg in enumerate(groups):
                    n_samp_each_axis[vv,aa,gi] = len(gg)
                    # mean within each label group 
                    mean_each_sem_level[vv,aa,gi] = np.mean(gg)
            else:                
                # at least one category is missing for this voxel's pRF and this semantic axis.
                # skip it and put nans in the arrays.               
                sem_discrim_each_axis[vv,aa] = np.nan
                sem_corr_each_axis[vv,aa] = np.nan
                n_samp_each_axis[vv,aa,:] = np.nan
                mean_each_sem_level[vv,aa,:] = np.nan
                
    return sem_discrim_each_axis, sem_corr_each_axis, n_samp_each_axis, mean_each_sem_level


def get_semantic_partial_corrs(best_params, labels_all, axes_to_do, \
                               unique_labels_each, val_voxel_data_pred, \
                               trials_use_each_prf = None, debug=False):   
    """
    Measure how well voxels' predicted responses distinguish between image patches with 
    different semantic content.
    Computing partial correlation coefficients here - to disentangle contributions of different 
    (possibly correlated) semantic features.
    """

    best_models, weights, bias, features_mt, features_st, best_model_inds = best_params
    n_trials, n_voxels = val_voxel_data_pred.shape[0:2]

    n_sem_axes = len(axes_to_do)
    labels_all = labels_all[:,axes_to_do,:]
    
    partial_corr_each_axis = np.zeros((n_voxels, n_sem_axes))
    
    max_categ = np.max([len(unique_labels_each[aa]) for aa in axes_to_do])
    n_samp_each_axis = np.zeros((n_voxels, n_sem_axes, max_categ),dtype=np.float32)
   
    for vv in range(n_voxels):
        
        if debug and (vv>1):
            continue
        print('computing partial correlations for voxel %d of %d, prf %d\n'%(vv, n_voxels, best_model_inds[vv,0]))
        
        resp = val_voxel_data_pred[:,vv,0]
        
        if trials_use_each_prf is not None:
            # select subset of trials to work with
            trials_use = trials_use_each_prf[:,best_model_inds[vv,0]]
            resp = resp[trials_use]
            labels_use = labels_all[trials_use,:,:]
            if np.sum(trials_use)==0:
                print('voxel %d: no trials are included here, skipping it'%vv)
                partial_corr_each_axis[vv,:] = np.nan
                continue
        else:
            labels_use = labels_all
        
        inds2use = (np.sum(np.isnan(labels_use[:,:,best_model_inds[vv,0]]), axis=1)==0)
        
        for aa in range(n_sem_axes):
            
            other_axes = ~np.isin(np.arange(n_sem_axes), aa)
            
            # going to compute information about the current axis of interest, while
            # partialling out the other axes. 
            labels_main_axis = labels_use[:,aa,best_model_inds[vv,0]]
            labels_other_axes = labels_use[:,other_axes,best_model_inds[vv,0]]

            unique_labels_actual = np.unique(labels_main_axis[inds2use])
            
            if np.all(np.isin(unique_labels_each[aa], unique_labels_actual)):
                
                partial_corr = stats_utils.compute_partial_corr(x=labels_main_axis[inds2use], \
                                                                y=resp[inds2use], \
                                                                c=labels_other_axes[inds2use,:])
                partial_corr_each_axis[vv,aa] = partial_corr
                
                for ui, uu in enumerate(unique_labels_actual):
                    n_samp_each_axis[vv,aa,ui] = np.sum(labels_main_axis[inds2use]==uu)
                    
            else:                
                # at least one category is missing for this voxel's pRF and this semantic axis.
                # skip it and put nans in the arrays.               
                partial_corr_each_axis[vv,aa] = np.nan
                n_samp_each_axis[vv,aa,:] = np.nan
               
    return partial_corr_each_axis, n_samp_each_axis




def get_features_each_prf(best_params, prf_models, image_inds_val, \
                        feature_loader, zscore=False, sample_batch_size=100, \
                        voxel_batch_size=100, debug=False, \
                        trials_use_each_prf = None,
                        dtype=np.float32):
    
    """ 
    Just loads the features in each pRF on each trial. 
    Only used in special case when evaluating feature "tuning" of raw voxel data.
    """
    
    params = best_params
  
    n_trials = len(image_inds_val)
    
    n_prfs = prf_models.shape[0]
    
    best_models, weights, bias, features_mt, features_st, best_model_inds = params
    n_voxels = len(best_model_inds)
    
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
