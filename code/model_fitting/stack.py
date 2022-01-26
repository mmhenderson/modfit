from __future__ import division
import sys
import numpy as np
import copy
from cvxopt import matrix, solvers

from utils import stats_utils



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

