import sys
import os
import struct
import time
import numpy as np
import h5py
from tqdm import tqdm
import pickle
import math
import sklearn
from sklearn import decomposition
import gc
import copy

import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.functional as F
import torch.optim as optim

from utils import numpy_utility, torch_utils
from model_src import fwrf_fit as fwrf_fit
from model_src import texture_statistics_gabor as texture_statistics

    
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
        for rv, lv in numpy_utility.iterate_range(0, n_voxels, voxel_batch_size):
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
            for rt, lt in numpy_utility.iterate_range(0, n_trials, sample_batch_size):

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







def get_voxel_texture_feature_corrs(best_params, models, _fmaps_fn_complex, _fmaps_fn_simple, val_voxel_single_trial_data, 
                                    val_stim_single_trial_data, sample_batch_size, include_autocorrs=True,  include_crosscorrs=True, autocorr_output_pix=5, n_prf_sd_out=2, 
                                    aperture=1.0, device=None, debug=False, dtype=np.float32):
    
    # Directly compute linear correlation bw each voxel's response and value of each feature channel.

    print('\nComputing activation in each feature channel on validation set trials...\n')
    n_features = best_params[1].shape[1]
    n_trials_val, n_voxels = np.shape(val_voxel_single_trial_data)
    features_each_model_val = None
    
    ### COMPUTE CORRELATION OF VALIDATION SET VOXEL RESP WITH FEATURE ACTIVATIONS ###########
    # this will serve as a measure of "tuning"
    print('\nComputing voxel/feature correlations for validation set trials...\n')
    voxel_feature_correlations_val = np.zeros((n_voxels, n_features),dtype=dtype)
    best_models = best_params[0]
    best_model_inds = best_params[5]

    for vv in range(n_voxels):
        if debug and vv>1:
            break 
        sys.stdout.write('\rvoxel %d of %d'%(vv,n_voxels))
        
        # figure out for this voxel, which pRF estimate was best.
        best_model_ind = best_model_inds[vv]

        gc.collect()
        torch.cuda.empty_cache()
        
        all_feat_concat, feature_info = texture_statistics.compute_all_texture_features(_fmaps_fn_complex, _fmaps_fn_simple, val_stim_single_trial_data, 
                                                                         models[best_model_ind,:], sample_batch_size, include_autocorrs, include_crosscorrs, autocorr_output_pix, 
                                                                         n_prf_sd_out, aperture, device=device)
        # get correlations
        for ff in range(n_features):        
            voxel_feature_correlations_val[vv,ff] = np.corrcoef(all_feat_concat[:,ff], val_voxel_single_trial_data[:,vv])[0,1]
           
    return features_each_model_val, voxel_feature_correlations_val

   
def validate_model(best_params, val_voxel_single_trial_data, val_stim_single_trial_data, _fmaps_fn, sample_batch_size, voxel_batch_size, aperture, dtype=np.float32, pc=None, combs_zstats = None):
    
    # EVALUATE PERFORMANCE ON VALIDATION SET
    
    print('\nInitializing model for validation...\n')
    param_batch = [p[:voxel_batch_size] if p is not None else None for p in best_params]
    # To initialize this module for prediction, need to take just first batch of voxels.
    # Will eventually pass all voxels through in batches.
    _fwrf_fn = Torch_fwRF_voxel_block(_fmaps_fn, param_batch, input_shape=val_stim_single_trial_data.shape, aperture=aperture, pc=pc, combs_zstats=combs_zstats)
    
    print('\nGetting model predictions on validation set...\n')
    val_voxel_pred = get_predictions(val_stim_single_trial_data, _fmaps_fn, _fwrf_fn, best_params, sample_batch_size=sample_batch_size)
    
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

def get_voxel_feature_corrs(best_params, models, _fmaps_fn, val_voxel_single_trial_data, val_stim_single_trial_data, sample_batch_size, aperture, device, debug=False, dtype=np.float32, pc=None, combs_zstats=None):
    
    # Directly compute linear correlation bw each voxel's response and value of each feature channel.
    
    ### GET ACTUAL FEATURE VALUES FOR EACH TRIAL IN TESTING SET ########
    # will use to compute tuning etc based on voxel responses in validation set.
    # looping over every model here; there are fewer models than voxels so this is faster than doing each voxel separately.

    print('\nComputing activation in each feature channel on validation set trials...\n')
    n_features = best_params[1].shape[1]
    n_prfs = models.shape[0]
    n_trials_val, n_voxels = np.shape(val_voxel_single_trial_data)
    features_each_model_val = np.zeros(shape=(n_trials_val, n_features, n_prfs),dtype=dtype)
    
    features_pca_each_model_val = None
    voxel_pca_feature_correlations_val = None
    if pc is not None:
        print('\nUsing pca features for validation set..\n')
        pca_wts = pc[0]
        n_comp_needed = pc[3]
        pca_pre_mean = pc[4]
        features_pca_each_model_val = np.zeros(shape=(n_trials_val, n_features, n_prfs),dtype=dtype)
        voxel_pca_feature_correlations_val = np.zeros((n_voxels, n_features),dtype=dtype)
        
    for mm in range(n_prfs):
        if debug and mm>1:
            break 
        sys.stdout.write('\rmodel %d of %d'%(mm,n_prfs))
        
        if combs_zstats is not None:
            features, zstats_ignore = fwrf_fit.get_features_in_prf_combinations(models[mm,:], _fmaps_fn, val_stim_single_trial_data, sample_batch_size, aperture, device, combs_zstats=combs_zstats[mm,:])
        else:
            features = fwrf_fit.get_features_in_prf(models[mm,:], _fmaps_fn, val_stim_single_trial_data, sample_batch_size, aperture, device)   
        features_each_model_val[:,:,mm] = features

        if pc is not None:
            # project into pca space
            features_submean = features - np.tile(np.expand_dims(pca_pre_mean[:,mm], axis=0), [np.shape(features)[0], 1])
            features_reduced = features_submean @ np.transpose(pca_wts[0:n_comp_needed[mm],:,mm]) # subtract mean in same way as for original training set features                                                       
            features_pca_each_model_val[:,0:n_comp_needed[mm],mm] = features_reduced[:,0:n_comp_needed[mm]] # multiply by weight matrix

        
    ### COMPUTE CORRELATION OF VALIDATION SET VOXEL RESP WITH FEATURE ACTIVATIONS ###########
    # this will serve as a measure of "tuning"
    print('\nComputing voxel/feature correlations for validation set trials...\n')
    voxel_feature_correlations_val = np.zeros((n_voxels, n_features),dtype=dtype)
    best_models = best_params[0]
    best_model_inds = best_params[5]
    
    for vv in range(n_voxels):
        if debug and vv>1:
            break 
        sys.stdout.write('\rvoxel %d of %d'%(vv,n_voxels))
        
        # figure out for this voxel, which pRF estimate was best.
        best_model_ind = best_model_inds[vv]
        # taking features for the validation set images, within this voxel's fitted RF
        features2use = features_each_model_val[:,:,best_model_ind]
        if pc is not None:
            features2use_pca = features_pca_each_model_val[:,:,best_model_ind]

        for ff in range(n_features):        
            voxel_feature_correlations_val[vv,ff] = np.corrcoef(features2use[:,ff], val_voxel_single_trial_data[:,vv])[0,1]
            if pc is not None:
                if ff<n_comp_needed[best_model_ind]:                                               
                    voxel_pca_feature_correlations_val[vv,ff] = np.corrcoef(features2use_pca[:,ff], val_voxel_single_trial_data[:,vv])[0,1]
                else:
                    voxel_pca_feature_correlations_val[vv,ff] = None
    
    return features_each_model_val, voxel_feature_correlations_val, features_pca_each_model_val, voxel_pca_feature_correlations_val




# class texture_model(torch.nn.Module):
    
#     """
#     Module that predicts voxel responses based on texture features and encoding model weights.
#     Texture features are computed in the module specified by '_texture_fn'.
#     Currently written to work with just 1 voxel at a time. This is because the texture features are pRF-specific, 
#     and have to be computed 1 pRF at a time. Could probably batch >1 voxel if they had same pRF params, though.
#     """
    
#     def __init__(self, _texture_fn, params, input_shape = (1,3,227,227)):
        
#         super(texture_model, self).__init__()        
# #         print('Creating FWRF texture model...')
        
#         self.texture_fn = _texture_fn       
#         self.voxel_batch_size = 1 # because of how this model is set up, can only do for one voxel at a time! slow.
#         device = _texture_fn.device
      
#         models, weights, bias, features_mt, features_st, best_model_inds = params
# #         _x = torch.empty((1,)+input_shape[1:], device=device).uniform_(0, 1)
# #         _fmaps = _fmaps_fn_complex(_x)
# #         n_features_complex, self.fmaps_rez = fwrf_fit.get_fmaps_sizes(_fmaps_fn_complex, _x, device)    
        
#         self.models = nn.Parameter(torch.from_numpy(models).to(device), requires_grad=False)
        
#         self.weights = nn.Parameter(torch.from_numpy(weights).to(device), requires_grad=False)
#         self.bias = None
#         if bias is not None:
#             self.bias = nn.Parameter(torch.from_numpy(bias).to(device), requires_grad=False)
      
#         self.features_m = None
#         self.features_s = None
#         if features_mt is not None:
#             self.features_m = nn.Parameter(torch.from_numpy(features_mt.T).to(device), requires_grad=False)
#         if features_st is not None:
#             self.features_s = nn.Parameter(torch.from_numpy(features_st.T).to(device), requires_grad=False)
       
#     def load_voxel_block(self, *params):
#         # This takes a given set of parameters for the voxel batch of interest, and puts them 
#         # into the right fields of the module so we can use them in a forward pass.
#         models = params[0]
#         assert(models.shape[0]==self.voxel_batch_size)
        
#         torch_utils.set_value(self.models, models)

#         for _p,p in zip([self.weights, self.bias], params[1:3]):
#             if _p is not None:
#                 if len(p)<_p.size()[0]:
#                     pp = np.zeros(shape=_p.size(), dtype=p.dtype)
#                     pp[:len(p)] = p
#                     torch_utils.set_value(_p, pp)
#                 else:
#                     torch_utils.set_value(_p, p)
                    
#         for _p,p in zip([self.features_m, self.features_s], params[3:]):
#             if _p is not None:
#                 if len(p)<_p.size()[1]:
#                     pp = np.zeros(shape=(_p.size()[1], _p.size()[0]), dtype=p.dtype)
#                     pp[:len(p)] = p
#                     torch_utils.set_value(_p, pp.T)
#                 else:
#                     torch_utils.set_value(_p, p.T)
                    
        
#     def forward(self, image_batch):
        
#         all_feat_concat, feature_info = self.texture_fn(image_batch,self.models)
        
#         _features = all_feat_concat.view([all_feat_concat.shape[0],-1,1]) # trials x features x 1
# #         _features = torch_utils._to_torch(all_feat_concat, device=self.weights.device).view([all_feat_concat.shape[0],-1,1]) # trials x features x 1
       
#         if self.features_m is not None:    
#             # features_m is [nfeatures x nvoxels]
#             _features = _features - torch.tile(torch.unsqueeze(self.features_m, dim=0), [_features.shape[0], 1, 1])

#         if self.features_s is not None:
#             _features = _features/torch.tile(torch.unsqueeze(self.features_s, dim=0), [_features.shape[0], 1, 1])
#             _features[torch.isnan(_features)] = 0.0 # this applies in the pca case when last few columns of features are missing

#         # features is [#samples, #features, #voxels] - swap dims to [#voxels, #samples, features]
#         _features = torch.transpose(torch.transpose(_features, 0, 2), 1, 2)
#         # weights is [#voxels, #features]
#         # _r will be [#voxels, #samples, 1] - then [#samples, #voxels]
#         _r = torch.squeeze(torch.bmm(_features, torch.unsqueeze(self.weights, 2)), dim=2).t() 
  
#         if self.bias is not None:
#             _r = _r + torch.tile(torch.unsqueeze(self.bias, 0), [_r.shape[0],1])
            
#         return _r
