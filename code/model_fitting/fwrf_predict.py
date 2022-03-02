from __future__ import division
import sys
import time
import numpy as np
import torch
import scipy.stats

from utils import numpy_utils, torch_utils, stats_utils


def validate_fwrf_model(best_params, prf_models, voxel_data, images, \
                        feature_loader, zscore=False, sample_batch_size=100, \
                        voxel_batch_size=100, debug=False, \
                        dtype=np.float32, device=None):
    
    """ 
    Evaluate trained FWRF model, using best fit pRF params and feature weights.
    Returns the trial-by-trial predictions of the encoding model for the voxels in 
    voxel_data, which can be used for further analyses (get_feature_tuning etc.),
    and computes overall val_cc and val_r2 (prediction accuracy of model).
    Also performs variance partition analysis, by evaluating the model with different 
    held out sets of features at a time.
    """
    
    params = best_params
    if device is None:
        device=torch.device('cpu:0')
    
    n_trials, n_voxels = len(images), len(params[0])
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
        
        # First gather features for all pRFs. There are fewer pRFs than voxels, so it is faster
        # to loop over pRFs first, then voxels.
        
        feature_loader.clear_big_features()
        
        for mm in range(n_prfs):
            if mm>1 and debug:
                break
            print('Getting features for prf %d: [x,y,sigma] is [%.2f %.2f %.4f]'%\
                  (mm, prf_models[mm,0],  prf_models[mm,1],  prf_models[mm,2] ))
            if not np.any(best_model_inds==mm):
                print('No voxels have this pRF as their best model, skipping it.')
                continue
            # all_feat_concat is size [ntrials x nfeatures] (where nfeatures can be <max_features)
            # feature_inds_defined is [max_features]
            all_feat_concat, feature_inds_defined = feature_loader.load(images, mm, fitting_mode=False)
            
            if zscore:
                # using mean and std that were computed on training set during fitting - keeping 
                # these pars constant here seems to improve fits. 
                tiled_mean = np.tile(features_mt[mm,feature_inds_defined], [n_trials, 1])
                tiled_std = np.tile(features_st[mm,feature_inds_defined], [n_trials, 1])
                all_feat_concat = (all_feat_concat - tiled_mean)/tiled_std
                assert(not np.any(np.isnan(all_feat_concat)) and not np.any(np.isinf(all_feat_concat)))
                        
            features_each_prf[:,feature_inds_defined,mm] = all_feat_concat
            feature_inds_defined_each_prf[:,mm] = feature_inds_defined
            
        feature_loader.clear_big_features()
        
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

#     # any nans become zeros here.
#     val_cc = np.nan_to_num(val_cc)
#     val_r2 = np.nan_to_num(val_r2) 
    
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
    
#     all categories must be binary.
#     assert(np.all([len(un)==2 for un in unique_labels_each]))
    
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


def get_semantic_discrim_balanced(best_params, labels_all, axes_to_balance, unique_labels_each, \
                                  val_voxel_data_pred, n_samp_iters=1000, debug=False):
   
    """
    Measure how well voxels' predicted responses distinguish between image patches with 
    different semantic content.
    Balance the number of trials across levels of every column in labels_all.
    """
    
    best_models, weights, bias, features_mt, features_st, best_model_inds = best_params
    n_voxels = val_voxel_data_pred.shape[1]

    _, _, n_prfs = labels_all.shape
    
    assert(len(axes_to_balance[0])==2)
    n_axes_pairs = len(axes_to_balance)
    n_sem_axes = 2;
    
    sem_discrim_each_axis = np.zeros((n_voxels, n_sem_axes, n_axes_pairs))
    sem_corr_each_axis = np.zeros((n_voxels, n_sem_axes, n_axes_pairs))
    mean_each_sem_level = np.zeros((n_voxels, n_sem_axes, 2, n_axes_pairs))
    min_samp = np.zeros((n_voxels, n_axes_pairs))

    # elements of axes_to_balance are pairs of axes that we will do resampling to balance.
    # looping over the pairs now.
    for pi, axes in enumerate(axes_to_balance):

        print('Balancing over axes:')
        print(axes)
        
        # only going to work with the axes specified in axes here
        labels_balance = labels_all[:,axes,:]
        unique_labels_balance = [unique_labels_each[aa] for aa in axes]

        # must both be binary for this to work
        assert(np.all([len(un)==2 for un in unique_labels_balance]))
        assert(np.all([(un==[0,1]) for un in unique_labels_balance]))

        # define the groups of labels we are balancing over
        n_bal_groups = 2**len(axes)
        combs = np.array([np.repeat([0,1],2),np.tile([0,1],2)]).T
        
        # first find out how many trials of each label combination we have here    
        counts_all = np.zeros((n_prfs,n_bal_groups))
        for mm in range(n_prfs):
            counts = [len(np.where(np.all(labels_balance[:,:,mm]==combs[cc,:], axis=1))[0]) \
                              for cc in range(n_bal_groups)]
            counts_all[mm,:] = counts
        # print some summary stats as a check (median across pRFs)
        print('median counts each group:')
        print(np.median(counts_all, axis=0))
        print('number pRFs with count<10:')
        print(np.sum(counts_all<10, axis=0))

        # Now looping over pRFs, doing balancing separately in each pRF
        for mm in range(n_prfs):

            if debug and mm>1:
                continue

            print('Computing balanced semantic discriminability, processing pRF %d of %d'%(mm, n_prfs))

            # which voxels had this as their pRF?    
            vox2do = np.where(best_model_inds[:,0]==mm)[0]
            if len(vox2do)==0:
                continue

            min_count = int(np.min(counts_all[mm,:]))
            if min_count==0:
                sem_discrim_each_axis[vox2do,:] = np.nan
                sem_corr_each_axis[vox2do,:] = np.nan
                mean_each_sem_level[vox2do,:,:] = np.nan
                min_samp[vox2do] = 0
                continue

            min_samp[vox2do] = min_count

            # define a set of trial indices to use for re-sampling
            trial_inds_resample = np.zeros((n_samp_iters, min_count*n_bal_groups),dtype=int)
            for gg in range(n_bal_groups):
                # find actual list of trials with this label combination
                trial_inds_this_comb = np.where(np.all(labels_balance[:,:,mm]==combs[gg,:], axis=1))[0]
                samp_inds = np.arange(gg*min_count, (gg+1)*min_count)
                for ii in range(n_samp_iters):
                    # sample without replacement from the full set of trials.
                    # if this is the smallest group, this means taking all the trials.
                    # otherwise it is a sub-set of the trials.
                    trial_inds_resample[ii,samp_inds] = np.random.choice(trial_inds_this_comb, \
                                                                         min_count, \
                                                                         replace=False)

            # loop over every voxel, resample the data, and compute discrimination measures
            for vi, vv in enumerate(vox2do):

                resamp_discrim = np.zeros((n_samp_iters, n_sem_axes))
                resamp_corrs = np.zeros((n_samp_iters, n_sem_axes))
                resamp_means = np.zeros((n_samp_iters, n_sem_axes,2))

                resp = val_voxel_data_pred[:,vv,0] 

                for ii in range(n_samp_iters):          

                    # get the re-sampled response for this voxel
                    resp_resamp = resp[trial_inds_resample[ii,:]]
                    # and the corresponding re-sampled labels
                    labels_resamp = labels_balance[trial_inds_resample[ii,:],:,mm]

                    # double check to make sure counts are right
                    check_counts = [len(np.where(np.all(labels_resamp==combs[cc,:], axis=1))[0]) \
                                    for cc in range(n_bal_groups)]
                    assert(np.all(np.array(check_counts)==min_count))

                    for aa in range(n_sem_axes):

                        labels = labels_resamp[:,aa]
                        assert(not np.any(np.isnan(labels)))
                        assert(np.sum(labels==0)==min_count*2 and np.sum(labels==1)==min_count*2)

                        # separate trials into those with the different labels for this semantic axis.
                        group_inds = [(labels==ll) for ll in [0,1]]
                        groups = [resp_resamp[gi] for gi in group_inds]

                        # use t-statistic as a measure of discriminability
                        # larger pos value means resp[label==1] > resp[label==0]
                        resamp_discrim[ii,aa] = stats_utils.ttest_warn(groups[1],groups[0]).statistic

                        # also computing a correlation coefficient between semantic label/voxel response
                        # sign is consistent with t-statistic
                        resamp_corrs[ii,aa] = stats_utils.numpy_corrcoef_warn(resp_resamp,labels)[0,1]
                        
                        for gi, gg in enumerate(groups):
                            # mean within each label group 
                            resamp_means[ii,aa,gi] = np.mean(gg)

                # averaging over the iterations of resampling, to get a final measure.
                sem_discrim_each_axis[vi,:,pi] = np.mean(resamp_discrim, axis=0)
                sem_corr_each_axis[vi,:,pi] = np.mean(resamp_corrs, axis=0)
                mean_each_sem_level[vi,:,:,pi] = np.mean(resamp_means, axis=0)

    return sem_discrim_each_axis, sem_corr_each_axis, min_samp, mean_each_sem_level

