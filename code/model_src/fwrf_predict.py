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

import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.functional as F
import torch.optim as optim

from utils import numpy_utility, torch_utils
from model_src import fwrf_fit, texture_statistics

    
def get_r2(actual,predicted):
  
    # calculate r2 for this fit.
    ssres = np.sum(np.power((predicted - actual),2));
#     print(ssres)
    sstot = np.sum(np.power((actual - np.mean(actual)),2));
#     print(sstot)
    r2 = 1-(ssres/sstot)
    
    return r2
    
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




def validate_texture_model(best_params, val_voxel_single_trial_data, val_stim_single_trial_data, _fmaps_fn_complex, 
                           _fmaps_fn_simple, sample_batch_size, include_autocorrs=True, include_crosscorrs=True, autocorr_output_pix=5, n_prf_sd_out=2, 
                           aperture=1.0, debug=False, dtype=np.float32):
    
    # EVALUATE PERFORMANCE ON VALIDATION SET
    
    print('\nInitializing model for validation...\n')
    param_batch = [p[0:1] if p is not None else None for p in best_params]
    # To initialize this module for prediction, need to take just first batch of voxels.
    # Will eventually pass all voxels through in batches.
    _fwd_model = texture_model(_fmaps_fn_complex, _fmaps_fn_simple, param_batch, 
                   input_shape=val_stim_single_trial_data.shape, include_autocorrs=include_autocorrs, include_crosscorrs=include_crosscorrs, autocorr_output_pix=5, n_prf_sd_out=2, aperture=1.0)

    print('\nGetting model predictions on validation set...\n')
    val_voxel_pred = get_predictions_texture_model(val_stim_single_trial_data, _fwd_model, best_params, sample_batch_size=sample_batch_size, debug=debug)

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



class Torch_fwRF_voxel_block(nn.Module):
    '''
    This is the module that maps from feature maps to voxel predictions according to weights.
    This works for a batch of voxels at a time. 
    Initialize with one set of voxels, but can use load_voxel_block to run w different batches
    '''

    def __init__(self, _fmaps_fn, params, input_shape=(1,3,227,227), aperture=1.0, pc=None, combs_zstats=None):
        super(Torch_fwRF_voxel_block, self).__init__()
        print('Making fwrf module...')
        self.aperture = aperture
        models, weights, bias, features_mt, features_st, best_model_inds = params
        device = next(_fmaps_fn.parameters()).device
        _x =torch.empty((1,)+input_shape[1:], device=device).uniform_(0, 1)
        _fmaps = _fmaps_fn(_x)
        self.fmaps_rez = []
        for k,_fm in enumerate(_fmaps):
            assert _fm.size()[2]==_fm.size()[3], 'All feature maps need to be square'
            self.fmaps_rez += [_fm.size()[2],]
        
        self.prfs = []
        for k,n_pix in enumerate(self.fmaps_rez):
            prf = numpy_utility.make_gaussian_mass_stack(models[:,0], models[:,1], models[:,2], n_pix, size=aperture, dtype=np.float32)[2]
            self.prfs += [nn.Parameter(torch.from_numpy(prf).to(device), requires_grad=False),]
            self.register_parameter('prfs%d'%k, self.prfs[-1])
            
        self.weights = nn.Parameter(torch.from_numpy(weights).to(device), requires_grad=False)
        self.bias = None
        if bias is not None:
            self.bias = nn.Parameter(torch.from_numpy(bias).to(device), requires_grad=False)
            
        self.features_m = None
        self.features_s = None
        if features_mt is not None:
            self.features_m = nn.Parameter(torch.from_numpy(features_mt.T).to(device), requires_grad=False)
        if features_st is not None:
            self.features_s = nn.Parameter(torch.from_numpy(features_st.T).to(device), requires_grad=False)
       
        # add in params related to pca on training features, if this was done. otherwise ignore.
        self.pca_wts = None
        self.n_comp_needed = None
        self.pca_pre_mean = None
        if pc is not None:
            self.pca_wts = pc[0]
            self.n_comp_needed = pc[3]
            self.pca_pre_mean = pc[4]
            self.n_comp_this_batch = nn.Parameter(torch.from_numpy(self.n_comp_needed[best_model_inds]).to(device), requires_grad=False)
            self.pca_wts_this_batch = nn.Parameter(torch.from_numpy(self.pca_wts[:,:,best_model_inds]).to(device), requires_grad=False)
            self.pca_premean_this_batch = nn.Parameter(torch.from_numpy(self.pca_pre_mean[:,best_model_inds]).to(device), requires_grad=False)
        
        self.combs_zstats = None
        if combs_zstats is not None:
            self.combs_zstats = combs_zstats
            self.combs_zstats_this_batch = nn.Parameter(torch.from_numpy(self.combs_zstats[best_model_inds,:]).to(device), requires_grad=False)
            
    def load_voxel_block(self, *params):
        # This takes a given set of parameters for the voxel batch of interest, and puts them 
        # into the right fields of the module so we can use them in a forward pass.
        models = params[0]
                
        for _prfs,n_pix in zip(self.prfs, self.fmaps_rez):
            prfs = numpy_utility.make_gaussian_mass_stack(models[:,0], models[:,1], models[:,2], n_pix, size=self.aperture, dtype=np.float32)[2]
            if len(prfs)<_prfs.size()[0]:
                pp = np.zeros(shape=_prfs.size(), dtype=prfs.dtype)
                pp[:len(prfs)] = prfs
                torch_utils.set_value(_prfs, pp)
            else:
                torch_utils.set_value(_prfs, prfs)
                
        if self.combs_zstats is not None:
            best_model_inds = params[5]
            torch_utils.set_value(self.combs_zstats_this_batch, self.combs_zstats[best_model_inds,:])
            
        if self.pca_wts is not None:
            
            # figure out which pca parameters go with which voxels in this voxel batch
            best_model_inds = params[5]
#             print([self.pca_wts_this_batch.shape[0],len(best_model_inds)])
            if len(best_model_inds)<self.pca_wts_this_batch.shape[2]:
                
                # if this is a small batch of trials, pad it with zeros                
                pp1 = np.zeros(shape=self.pca_wts_this_batch.shape, dtype=self.pca_wts.dtype)
                pp1[:,:,0:len(best_model_inds)] = self.pca_wts[:,:,best_model_inds]
                
                pp2 = np.zeros(shape=self.n_comp_this_batch.shape, dtype=self.n_comp_needed.dtype)
                pp2[0:len(best_model_inds)] = self.n_comp_needed[best_model_inds]   
                
                pp3 = np.zeros(shape=self.pca_premean_this_batch.shape, dtype=self.pca_pre_mean.dtype)
                pp3[:,0:len(best_model_inds)] = self.pca_pre_mean[:,best_model_inds]
                
                
                torch_utils.set_value(self.pca_wts_this_batch,   pp1)
                torch_utils.set_value(self.n_comp_this_batch,   pp2)
                torch_utils.set_value(self.pca_premean_this_batch,   pp3)
            else:
                torch_utils.set_value(self.pca_wts_this_batch,   self.pca_wts[:,:,best_model_inds])
                torch_utils.set_value(self.n_comp_this_batch,   self.n_comp_needed[best_model_inds])
                torch_utils.set_value(self.pca_premean_this_batch, self.pca_pre_mean[:,best_model_inds])
                
        for _p,p in zip([self.weights, self.bias], params[1:3]):
            if _p is not None:
                if len(p)<_p.size()[0]:
                    pp = np.zeros(shape=_p.size(), dtype=p.dtype)
                    pp[:len(p)] = p
                    torch_utils.set_value(_p, pp)
                else:
                    torch_utils.set_value(_p, p)
                    
        for _p,p in zip([self.features_m, self.features_s], params[3:]):
            if _p is not None:
                if len(p)<_p.size()[1]:
                    pp = np.zeros(shape=(_p.size()[1], _p.size()[0]), dtype=p.dtype)
                    pp[:len(p)] = p
                    torch_utils.set_value(_p, pp.T)
                else:
                    torch_utils.set_value(_p, p.T)
 
    def forward(self, _fmaps):

        _features = torch.cat([torch.tensordot(_fm, _prf, dims=[[2,3], [1,2]]) for _fm,_prf in zip(_fmaps, self.prfs)], dim=1) # [#samples, #features, #voxels]
        
        if self.combs_zstats is not None:    
            # convert these first order features into a larger matrix that includes second-order combinations of features            
            f = torch_utils._to_torch(np.zeros(shape=(_features.shape[0], _features.shape[1]*_features.shape[1]+_features.shape[1], _features.shape[2]),dtype=self.combs_zstats.dtype), device=_features.device)
            for vv in range(_features.shape[2]):
                f_first = (_features[:,:,vv] - self.combs_zstats[vv,0])/self.combs_zstats[vv,1]
                f_second = torch.tile(_features[:,:,vv], [1,_features.shape[1]]) * torch.repeat_interleave(_features[:,:,vv], _features.shape[1], axis=1)
                f_second = (f_second - self.combs_zstats[vv,2])/self.combs_zstats[vv,3]
                f[:,:,vv] = torch.cat([f_first, f_second], axis=1)
                
            _features = f
#         print(_features.shape)
        
        if self.pca_wts is not None:            
        
            # apply the pca matrix to each voxel - to keep all features same length, put zeros for components past the desired number.
            features_pca = torch_utils._to_torch(np.zeros(shape=_features.shape, dtype=self.pca_wts.dtype), device=_features.device)
            
            # features is [#samples, #features, #voxels]
            for vv in range(_features.shape[2]):
#                 print([vv, self.n_comp_this_batch.shape, self.pca_wts_this_batch.shape, self.pca_premean_this_batch.shape])
                features_submean = _features[:,:,vv] - torch.tile(torch.unsqueeze(self.pca_premean_this_batch[:,vv], dim=0), [_features.shape[0],1])
                
                features_pca[:, 0:self.n_comp_this_batch[vv], vv] = torch.tensordot(features_submean, self.pca_wts_this_batch[0:self.n_comp_this_batch[vv],:,vv], dims=[[1],[1]]) 

            _features = features_pca

        if self.features_m is not None:    
            # features_m is [nfeatures x nvoxels]
            _features = _features - torch.tile(torch.unsqueeze(self.features_m, dim=0), [_features.shape[0], 1, 1])

        if self.features_s is not None:
            _features = _features/torch.tile(torch.unsqueeze(self.features_s, dim=0), [_features.shape[0], 1, 1])
            _features[torch.isnan(_features)] = 0.0 # this applies in the pca case when last few columns of features are missing

        # features is [#samples, #features, #voxels] - swap dims to [#voxels, #samples, features]
        _features = torch.transpose(torch.transpose(_features, 0, 2), 1, 2)
        # weights is [#voxels, #features]
        # _r will be [#voxels, #samples, 1] - then [#samples, #voxels]
        _r = torch.squeeze(torch.bmm(_features, torch.unsqueeze(self.weights, 2)), dim=2).t() 
  
        if self.bias is not None:
            _r = _r + torch.tile(torch.unsqueeze(self.bias, 0), [_r.shape[0],1])
            
        return _r

    
def get_predictions(images, _fmaps_fn, _fwrf_fn, params, sample_batch_size=100):
    """
    The predictive fwRF model for arbitrary input image.

    Parameters
    ----------
    images : ndarray, shape (#samples, #channels, x, y)
        Input image block.
    _fmaps_fn: Torch module
        Torch module that returns a list of torch tensors.
        This is defined previously, maps from images to feature maps.
    _fwrf_fn: Torch module
        Torch module that compute the fwrf model for one batch of voxels
        Defined in Torch_fwrf_voxel_block
    params: list including all of the following:
    [
        models : ndarray, shape (#voxels, 3)
            The RF model (x, y, sigma) associated with each voxel.
        weights : ndarray, shape (#voxels, #features)
            Tuning weights
        bias: Can contain a bias parameter of shape (#voxels) if add_bias is True.
           Tuning biases: None if there are no bias
        features_mean (optional): ndarray, shape (#voxels, #feature)
            None if zscore is False. Otherwise returns zscoring average per feature.
        features_std (optional): ndarray, shape (#voxels, #feature)
            None if zscore is False. Otherwise returns zscoring std.dev. per feature.
    ]
    sample_batch_size (default: 100)
        The sample batch size (used where appropriate)

    Returns
    -------
    pred : ndarray, shape (#samples, #voxels)
        The prediction of voxel activities for each voxels associated with the input images.
    """
    
    dtype = images.dtype.type
    device = next(_fmaps_fn.parameters()).device
    _params = [_p for _p in _fwrf_fn.parameters()]
    voxel_batch_size = _params[0].size()[0]    
    n_trials, n_voxels = len(images), len(params[0])

    pred = np.full(fill_value=0, shape=(n_trials, n_voxels), dtype=dtype)
    start_time = time.time()
    
    with torch.no_grad():
        
        ## Looping over voxels here in batches, will eventually go through all.
        for rv, lv in numpy_utility.iterate_range(0, n_voxels, voxel_batch_size):
            
            # for this voxel batch, put the right parameters into the _fwrf_fn module
            # so that we can do forward pass...
            _fwrf_fn.load_voxel_block(*[p[rv] if p is not None else None for p in params])
            pred_block = np.full(fill_value=0, shape=(n_trials, voxel_batch_size), dtype=dtype)
            
            # Now looping over validation set trials in batches
            for rt, lt in numpy_utility.iterate_range(0, n_trials, sample_batch_size):
                sys.stdout.write('\rsamples [%5d:%-5d] of %d, voxels [%6d:%-6d] of %d' % (rt[0], rt[-1], n_trials, rv[0], rv[-1], n_voxels))
                # Get predictions for this set of trials.
                pred_block[rt] = torch_utils.get_value(_fwrf_fn(_fmaps_fn(torch_utils._to_torch(images[rt], device)))) 
                
            pred[:,rv] = pred_block[:,:lv]
            
    total_time = time.time() - start_time
    print ('\n---------------------------------------')
    print ('total time = %fs' % total_time)
    print ('sample throughput = %fs/sample' % (total_time / n_trials))
    print ('voxel throughput = %fs/voxel' % (total_time / n_voxels))
    sys.stdout.flush()
    return pred

class texture_model(torch.nn.Module):
    
    def __init__(self, _fmaps_fn_complex, _fmaps_fn_simple, params, input_shape = (1,3,227,227), sample_batch_size=100, include_autocorrs=True,  include_crosscorrs=True, autocorr_output_pix=3, n_prf_sd_out=2, aperture=1.0):
        
        super(texture_model, self).__init__()        
        print('Creating FWRF texture model...')
        
        self._fmaps_fn_complex = _fmaps_fn_complex
        self._fmaps_fn_simple = _fmaps_fn_simple
        self.aperture=aperture
        self.include_autocorrs = include_autocorrs
        self.include_crosscorrs = include_crosscorrs
        self.autocorr_output_pix = autocorr_output_pix
        self.n_prf_sd_out = n_prf_sd_out
        self.sample_batch_size=sample_batch_size
        self.voxel_batch_size=1 # because of how this model is set up, can only do for one voxel at a time! slow.
        device = next(_fmaps_fn_complex.parameters()).device
      
        models, weights, bias, features_mt, features_st, best_model_inds = params
        _x = torch.empty((1,)+input_shape[1:], device=device).uniform_(0, 1)
        _fmaps = _fmaps_fn_complex(_x)
        n_features_complex, self.fmaps_rez = fwrf_fit.get_fmaps_sizes(_fmaps_fn_complex, _x, device)    
        
        self.models = nn.Parameter(torch.from_numpy(models).to(device), requires_grad=False)
        
        self.weights = nn.Parameter(torch.from_numpy(weights).to(device), requires_grad=False)
        self.bias = None
        if bias is not None:
            self.bias = nn.Parameter(torch.from_numpy(bias).to(device), requires_grad=False)
      
        self.features_m = None
        self.features_s = None
        if features_mt is not None:
            self.features_m = nn.Parameter(torch.from_numpy(features_mt.T).to(device), requires_grad=False)
        if features_st is not None:
            self.features_s = nn.Parameter(torch.from_numpy(features_st.T).to(device), requires_grad=False)
       
    def load_voxel_block(self, *params):
        # This takes a given set of parameters for the voxel batch of interest, and puts them 
        # into the right fields of the module so we can use them in a forward pass.
        models = params[0]
        assert(models.shape[0]==self.voxel_batch_size)
        
        torch_utils.set_value(self.models, models)

        for _p,p in zip([self.weights, self.bias], params[1:3]):
            if _p is not None:
                if len(p)<_p.size()[0]:
                    pp = np.zeros(shape=_p.size(), dtype=p.dtype)
                    pp[:len(p)] = p
                    torch_utils.set_value(_p, pp)
                else:
                    torch_utils.set_value(_p, p)
                    
        for _p,p in zip([self.features_m, self.features_s], params[3:]):
            if _p is not None:
                if len(p)<_p.size()[1]:
                    pp = np.zeros(shape=(_p.size()[1], _p.size()[0]), dtype=p.dtype)
                    pp[:len(p)] = p
                    torch_utils.set_value(_p, pp.T)
                else:
                    torch_utils.set_value(_p, p.T)
                    
        
    def forward(self, image_batch):
        
        all_feat_concat, feature_info = texture_statistics.compute_all_texture_features(self._fmaps_fn_complex, self._fmaps_fn_simple, image_batch, 
                                                                         self.models, self.sample_batch_size, self.include_autocorrs, self.include_crosscorrs, self.autocorr_output_pix, 
                                                                         self.n_prf_sd_out, self.aperture, device=self.weights.device)
        _features = torch_utils._to_torch(all_feat_concat, device=self.weights.device).view([all_feat_concat.shape[0],-1,1]) # trials x features x 1
       
        if self.features_m is not None:    
            # features_m is [nfeatures x nvoxels]
            _features = _features - torch.tile(torch.unsqueeze(self.features_m, dim=0), [_features.shape[0], 1, 1])

        if self.features_s is not None:
            _features = _features/torch.tile(torch.unsqueeze(self.features_s, dim=0), [_features.shape[0], 1, 1])
            _features[torch.isnan(_features)] = 0.0 # this applies in the pca case when last few columns of features are missing

        # features is [#samples, #features, #voxels] - swap dims to [#voxels, #samples, features]
        _features = torch.transpose(torch.transpose(_features, 0, 2), 1, 2)
        # weights is [#voxels, #features]
        # _r will be [#voxels, #samples, 1] - then [#samples, #voxels]
        _r = torch.squeeze(torch.bmm(_features, torch.unsqueeze(self.weights, 2)), dim=2).t() 
  
        if self.bias is not None:
            _r = _r + torch.tile(torch.unsqueeze(self.bias, 0), [_r.shape[0],1])
            
        return _r

def get_predictions_texture_model(images, _fwd_model, params, sample_batch_size=100, debug=False):
   
    dtype = images.dtype.type
    device = _fwd_model.weights.device
    _params = [_p for _p in _fwd_model.parameters()]
    voxel_batch_size = _fwd_model.voxel_batch_size
    assert(voxel_batch_size==1) # this won't work with batches of >1 voxel
    n_trials, n_voxels = len(images), len(params[0])

    pred = np.full(fill_value=0, shape=(n_trials, n_voxels), dtype=dtype)
    start_time = time.time()
    
    with torch.no_grad():
        
        vv=-1
        ## Looping over voxels here in batches, will eventually go through all.
        for rv, lv in numpy_utility.iterate_range(0, n_voxels, voxel_batch_size):
            vv=vv+1
            if debug and vv>1:
                break
            # for this voxel batch, put the right parameters into the _fwrf_fn module
            # so that we can do forward pass...
            _fwd_model.load_voxel_block(*[p[rv] if p is not None else None for p in params])
            pred_block = np.full(fill_value=0, shape=(n_trials, voxel_batch_size), dtype=dtype)
            
            # Now looping over validation set trials in batches
            for rt, lt in numpy_utility.iterate_range(0, n_trials, sample_batch_size):
                sys.stdout.write('\rsamples [%5d:%-5d] of %d, voxels [%6d:%-6d] of %d' % (rt[0], rt[-1], n_trials, rv[0], rv[-1], n_voxels))
                # Get predictions for this set of trials.
               
                pred_block[rt] = torch_utils.get_value(_fwd_model(torch_utils._to_torch(images[rt], device))) 
                
            pred[:,rv] = pred_block[:,:lv]
            
    total_time = time.time() - start_time
    print ('\n---------------------------------------')
    print ('total time = %fs' % total_time)
    print ('sample throughput = %fs/sample' % (total_time / n_trials))
    print ('voxel throughput = %fs/voxel' % (total_time / n_voxels))
    sys.stdout.flush()
    return pred