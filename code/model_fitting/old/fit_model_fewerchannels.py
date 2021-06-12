"""
Run the model fitting for Gabor FWRF model.
"""

import sys
import os
import struct
import time
import numpy as np
import h5py
from scipy.io import loadmat
from scipy.stats import pearsonr
from tqdm import tqdm
# import pickle
import math 
import gc

fpX = np.float32

import torch as T
import torch.nn as L
import torch.nn.init as I
import torch.nn.functional as F
import torch.optim as optim

import torch

root_dir   = os.path.dirname(os.getcwd())
sys.path.append(os.path.join(root_dir, 'fwrf_code_from_osf'))
import src.numpy_utility as pnu
from src.file_utility import save_stuff, flatten_dict, embed_dict
from src.file_utility import load_mask_from_nii, view_data
from src.roi import roi_map, iterate_roi
from src.load_nsd import image_uncolorize_fn, data_split, load_betas
from src.torch_feature_space import get_tuning_masks
from src.rf_grid    import linspace, logspace, model_space, model_space_pyramid
from src.torch_fwrf import learn_params_ridge_regression, get_predictions, get_value, set_value, Torch_fwRF_voxel_block, get_r2, add_nonlinearity, get_features_in_prf
from src.gabor_feature_extractor import Gaborizer

#################################################################################################

## PATHS ##
nsd_root = "/lab_data/tarrlab/common/datasets/NSD/"
stim_root = '/user_data/mmhender/nsd_stimuli/stimuli/nsd/'    
beta_root = nsd_root + "nsddata_betas/ppdata/"
mask_root = nsd_root + "nsddata/ppdata/"

#### CUDA STUFF ######
print ('#device:', torch.cuda.device_count())
print ('device#:', torch.cuda.current_device())
print ('device name:', torch.cuda.get_device_name(torch.cuda.current_device()))

torch.manual_seed(time.time())
device = torch.device("cuda:0") #cuda
torch.backends.cudnn.enabled=True

print ('\ntorch:', torch.__version__)
print ('cuda: ', torch.version.cuda)
print ('cudnn:', torch.backends.cudnn.version())
print ('dtype:', torch.get_default_dtype())



def fit_gabor_fwrf(subject=1, roi='V1', up_to_sess=1, n_ori=36, n_sf=12, sample_batch_size=50, voxel_batch_size=100, zscore_features=True, normalize_fn=False, debug=False):
    
    ## BASIC SETUP FOR SAVING ##
    saveext = ".png"
    savearg = {'format':'png', 'dpi': 120, 'facecolor': None}
    timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
    model_name = 'gabor_fwrf'

    output_dir = os.path.join(root_dir, "gabor_model_fits/S%02d/%s_%s/" % (subject,model_name,timestamp) )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print ("Time Stamp: %s" % timestamp)    

    ## GATHERING IMAGES/EXPERIMENT INFO ##
    exp_design_file = nsd_root + "nsddata/experiments/nsd/nsd_expdesign.mat"
    exp_design = loadmat(exp_design_file)
    ordering = exp_design['masterordering'].flatten() - 1 # zero-indexed ordering of indices (matlab-like to python-like)
    # ordering is 30000 values in the range [0-9999], which provide a list of trials in order. 
    # the value in ordering[ii] tells the index into the subject-specific stimulus array that we would need to take to
    # get the image for that trial.

    image_data = {}
    image_data_set = h5py.File(stim_root + "S%d_stimuli_227.h5py"%subject, 'r')
    image_data = np.copy(image_data_set['stimuli'])
    image_data_set.close()
    
    ## DEFINE ROIS/VOXELS TO USE HERE ##
    group_names = ['V1', 'V2', 'V3', 'hV4', 'V3ab', 'LO', 'IPS', 'VO', 'PHC', 'MT', 'MST', 'other']
    group = [[1,2],[3,4],[5,6], [7], [16, 17], [14, 15], [18,19,20,21,22,23], [8, 9], [10,11], [13], [12], [24,25,0]]

    #voxel_mask_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/brainmask_inflated_1.0.nii"%subject)
    # voxel_mask_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/brainmask_vcventral_1.0.nii"%subject)

    # note we don't seem to have the vcventral mask  - using full brain mask.
    voxel_mask_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/brainmask.nii.gz"%subject)
    voxel_roi_full  = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/prf-visualrois.nii.gz"%subject)
    voxel_kast_full = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/Kastner2015.nii.gz"%(subject))
    general_mask_full  = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/nsdgeneral.nii.gz"%(subject))
    ncsnr_full = load_mask_from_nii(beta_root + "subj%02d/func1pt8mm/betas_fithrf_GLMdenoise_RR/ncsnr.nii.gz"%subject)
    ###
    brain_nii_shape = voxel_roi_full.shape
#     print (brain_nii_shape)
    ###
    voxel_roi_mask_full = (voxel_roi_full>0).flatten().astype(bool)
    voxel_joined_roi_full = np.copy(voxel_kast_full.flatten())  # load kastner rois
    voxel_joined_roi_full[voxel_roi_mask_full] = voxel_roi_full.flatten()[voxel_roi_mask_full] # overwrite with prf rois
    # the ROIs defined here are mostly based on pRF mapping - but any voxels which weren't given an ROI definition during the 
    # prf mapping, will be given a definition based on Kastner atlas.

    ###
    voxel_mask  = np.nan_to_num(voxel_mask_full).flatten().astype(bool)
    # taking out the voxels in "other" to reduce the size a bit
    voxel_mask_adj = np.copy(voxel_mask)
    voxel_mask_adj[np.isin(voxel_joined_roi_full, group[-1])] = False
    voxel_mask_adj[np.isin(voxel_joined_roi_full, [-1])] = False

    voxel_mask  = voxel_mask_adj

    voxel_idx   = np.arange(len(voxel_mask))[voxel_mask]
    voxel_roi   = voxel_joined_roi_full[voxel_mask]
    voxel_ncsnr = ncsnr_full.flatten()[voxel_mask]

#     print ('full mask length = %d'%len(voxel_mask))
#     print ('selection length = %d'%np.sum(voxel_mask))
    print('\nSizes of all defined ROIs in this subject:')
    vox_total = 0
    for roi_mask, roi_name in iterate_roi(group, voxel_roi, roi_map, group_name=group_names):
        vox_total = vox_total + np.sum(roi_mask)
        print ("%d \t: %s" % (np.sum(roi_mask), roi_name))

    print ("%d \t: Total" % (vox_total))

    ## DEFINE WHICH VOXEL GROUPS TO ANALYZE HERE #####
    voxel_mask_this_roi = np.zeros(np.shape(voxel_mask)).astype('bool')
    if roi==None:     
        voxel_mask_this_roi[np.isin(voxel_joined_roi_full,np.concatenate(group[0:11],axis=0))] = True
        roi2print = 'allROIs'        
    else:
        groupind = [ii for ii in range(len(group_names)) if group_names[ii]==roi]
        groupind = groupind[0]
        voxel_mask_this_roi[np.isin(voxel_joined_roi_full, group[groupind])] = True
        roi2print = roi
        
    print('\nRunning model for %s, %d voxels\n'%(roi2print, np.sum(voxel_mask_this_roi)))

    voxel_idx_this_roi = np.arange(len(voxel_mask_this_roi))[voxel_mask_this_roi]
    
    
    ### LOADING DATA ####
    print('Loading data for sessions 1:%d...'%up_to_sess)
    beta_subj = beta_root + "subj%02d/func1pt8mm/betas_fithrf_GLMdenoise_RR/" % (subject,)
   
    voxel_data, filenames = load_betas(folder_name=beta_subj, zscore=True, voxel_mask=voxel_mask_this_roi, up_to=up_to_sess, load_ext=".nii")
    print('\nSize of full data set [nTrials x nVoxels] is:')
    print(voxel_data.shape)
    
    data_size, n_voxels = voxel_data.shape 
    trn_stim_data, trn_voxel_data,\
    val_stim_single_trial_data, val_voxel_single_trial_data,\
    val_stim_multi_trial_data, val_voxel_multi_trial_data = \
        data_split(image_uncolorize_fn(image_data), voxel_data, ordering, imagewise=False)
    n_trials_val = val_stim_single_trial_data.shape[0]
    
    ### DEFINE THE FILTERS FOR THE MODEL ###
    
    # fitting the model w fewer channels than we'll ultimately use to assess tuning...trying to reduce channel-channel correlations.
    n_ori_trn = 8
    n_sf_trn = 12
    print('\nBuilding Gabor filter bank with %d orientations and %d spatial frequencies...'%(n_ori_trn, n_sf_trn))
    cyc_per_stim_trn = logspace(n_sf_trn)(3., 72.) # Which SF values are we sampling here?
    _gaborizer_trn = Gaborizer(num_orientations=n_ori_trn, cycles_per_stim=cyc_per_stim_trn,
              pix_per_cycle=4.13, cycles_per_radius=.7, 
              radii_per_filter=4, complex_cell=True, pad_type='half', 
              crop=False).to(device)
    
    if normalize_fn:
        # adding a nonlinearity to the filter activations
        _fmaps_fn_trn = add_nonlinearity(_gaborizer_trn, lambda x: torch.log(1+torch.sqrt(x)))
    else:
        _fmaps_fn_trn = _gaborizer_trn
        
    # pull out some relevant stuff from gaborizer object to save
    sf_tuning_masks_trn = _gaborizer_trn.sf_tuning_masks
    assert(np.all(_gaborizer_trn.cyc_per_stim==cyc_per_stim_trn))

    ori_tuning_masks_trn = _gaborizer_trn.ori_tuning_masks
    orients_deg_trn = _gaborizer_trn.orients_deg
    orient_filters_trn = _gaborizer_trn.orient_filters  
  

    #### CREATING A HIGHER RESOLUTION FILTER BANK FOR USE IN GETTING TUNING FUNCTIONS ##########
    n_ori_val = n_ori
    n_sf_val = n_sf
    print('\nBuilding Gabor filter bank with %d orientations and %d spatial frequencies...'%(n_ori_val, n_sf_val))
    cyc_per_stim_val = logspace(n_sf_val)(3., 72.) # Which SF values are we sampling here?
    _gaborizer_val = Gaborizer(num_orientations=n_ori_val, cycles_per_stim=cyc_per_stim_val,
              pix_per_cycle=4.13, cycles_per_radius=.7, 
              radii_per_filter=4, complex_cell=True, pad_type='half', 
              crop=False).to(device)
    
    if normalize_fn:
        # adding a nonlinearity to the filter activations
        _fmaps_fn_val = add_nonlinearity(_gaborizer_val, lambda x: torch.log(1+torch.sqrt(x)))
    else:
        _fmaps_fn_val = _gaborizer_val
        
    # pull out some relevant stuff from gaborizer object to save
    sf_tuning_masks_val = _gaborizer_val.sf_tuning_masks
    assert(np.all(_gaborizer_val.cyc_per_stim==cyc_per_stim_val))

    ori_tuning_masks_val = _gaborizer_val.ori_tuning_masks
    orients_deg_val = _gaborizer_val.orients_deg
    orient_filters_val = _gaborizer_val.orient_filters  

    
    ### PARAMS FOR THE RF CENTERS, SIZES ####
    aperture = np.float32(1)
    smin, smax = np.float32(0.04), np.float32(0.4)
    n_sizes = 8

    # models is three columns, x, y, sigma
    models = model_space_pyramid(logspace(n_sizes)(smin, smax), min_spacing=1.4, aperture=1.1*aperture)    

    ### PARAMS FOR RIDGE REGRESSION ####
    holdout_pct = 0.10
    holdout_size = int(np.ceil(np.shape(trn_voxel_data)[0]*holdout_pct))
#     lambdas = np.logspace(0.,5.,9, dtype=np.float32)
    lambdas = np.logspace(-6., 1., 9).astype(np.float64)
#     lambdas = np.array([0.0,0.0,0.0])
    print('\nPossible lambda values are:')
    print(lambdas)
    
    #### DO THE ACTUAL MODEL FITTING HERE ####
    gc.collect()
    torch.cuda.empty_cache()

    best_losses, best_lambdas, best_params, covar_each_model_training = learn_params_ridge_regression(
        trn_stim_data, trn_voxel_data, _fmaps_fn_trn, models, lambdas, \
        aperture=aperture, zscore=zscore_features, sample_batch_size=sample_batch_size, \
        voxel_batch_size=voxel_batch_size, holdout_size=holdout_size, shuffle=False, add_bias=True, debug=debug)
    print('\nDone with training\n')

    #### EVALUATE PERFORMANCE ON VALIDATION SET #####
    print('\nInitializing model for validation...\n')
    param_batch = [p[:voxel_batch_size] if p is not None else None for p in best_params]
    # To initialize this module for prediction, need to take just first batch of voxels.
    # Will eventually pass all voxels through in batches.
    _fwrf_fn = Torch_fwRF_voxel_block(_fmaps_fn_trn, param_batch, input_shape=val_stim_single_trial_data.shape, aperture=1.0)
    
    print('\nGetting model predictions on validation set...\n')
    val_voxel_pred = get_predictions(val_stim_single_trial_data, _fmaps_fn_trn, _fwrf_fn, best_params, sample_batch_size=sample_batch_size)
    
    # val_cc is the correlation coefficient bw real and predicted responses across trials, for each voxel.
    val_cc  = np.zeros(shape=(n_voxels), dtype=fpX)
    val_r2 = np.zeros(shape=(n_voxels), dtype=fpX)
    
    print('\nEvaluating correlation coefficient on validation set...\n')
    for v in tqdm(range(n_voxels)):    
        val_cc[v] = np.corrcoef(val_voxel_single_trial_data[:,v], val_voxel_pred[:,v])[0,1]  
        val_r2[v] = get_r2(val_voxel_single_trial_data[:,v], val_voxel_pred[:,v])
        
    val_cc = np.nan_to_num(val_cc)
    val_r2 = np.nan_to_num(val_r2)
    
    ### GET ACTUAL FEATURE VALUES FOR EACH TRIAL IN TESTING SET ########
    # will use to compute tuning etc based on voxel responses in validation set.
    # looping over every model here; there are fewer models than voxels so this is faster than doing each voxel separately.

    print('\nComputing activation in each feature channel on validation set trials...\n')
    n_features = int(n_ori_val*n_sf_val)
    n_prfs = models.shape[0]
    features_each_model_val = np.zeros(shape=(n_trials_val, n_features, n_prfs),dtype=fpX)

    for mm in range(n_prfs):
        if debug and mm>1:
            break 
        sys.stdout.write('\rmodel %d of %d'%(mm,n_prfs))
        
        features = get_features_in_prf(models[mm,:], _fmaps_fn_val, val_stim_single_trial_data, sample_batch_size, aperture, device)     
        features_each_model_val[:,:,mm] = features

        
    ### COMPUTE CORRELATION OF VALIDATION SET VOXEL RESP WITH FEATURE ACTIVATIONS ###########
    # this will serve as a measure of "tuning"
    print('\nComputing voxel/feature correlations for validation set trials...\n')
    voxel_feature_correlations_val = np.zeros((n_voxels, n_features),dtype=fpX)
    best_models = best_params[0]
    
    for vv in range(n_voxels):
        if debug and vv>1:
            break 
        sys.stdout.write('\rvoxel %d of %d'%(vv,n_voxels))
        
        # figure out for this voxel, which pRF estimate was best.
        best_model_ind = np.where(np.sum(models==best_models[vv],axis=1)==3)[0]
        assert(len(best_model_ind)==1)
        # taking features for the validation set images, within this voxel's fitted RF
        features2use = features_each_model_val[:,:,best_model_ind[0]]

        for ff in range(n_features):        
            voxel_feature_correlations_val[vv,ff] = np.corrcoef(features2use[:,ff], val_voxel_single_trial_data[:,vv])[0,1]

        
    ### SAVE THE RESULTS TO DISK #########
    
    fn2save = output_dir+'model_params_%s'%roi2print
    print('\nSaving result to %s'%fn2save)
    
    torch.save({
    'feature_table_trn': _gaborizer_trn.feature_table,
    'feature_table_val': _gaborizer_val.feature_table,
    'sf_tuning_masks_trn': sf_tuning_masks_trn,
    'ori_tuning_masks_trn': ori_tuning_masks_trn,
    'cyc_per_stim_trn': cyc_per_stim_trn,
    'orients_deg_trn': orients_deg_trn,
    'orient_filters_trn': orient_filters_trn,
     'sf_tuning_masks_val': sf_tuning_masks_val,
    'ori_tuning_masks_val': ori_tuning_masks_val,
    'cyc_per_stim_val': cyc_per_stim_val,
    'orients_deg_val': orients_deg_val,
    'orient_filters_val': orient_filters_val,
    'aperture': aperture,
    'models': models,
    'voxel_mask': voxel_mask_this_roi,
    'brain_nii_shape': np.array(brain_nii_shape),
    'image_order': ordering,
    'voxel_index': voxel_idx_this_roi,
    'voxel_roi': voxel_roi,
    'voxel_ncsnr': voxel_ncsnr, 
    'best_params': best_params,
    'lambdas': lambdas, 
    'best_lambdas': best_lambdas,
    'best_losses': best_losses,
    'val_cc': val_cc,
    'val_r2': val_r2,   
    'features_each_model_val': features_each_model_val,
    'covar_each_model_training': covar_each_model_training,
    'voxel_feature_correlations_val': voxel_feature_correlations_val,
    'zscore_features': zscore_features,
    'normalize_fn': normalize_fn,
    'debug': debug
    }, fn2save)
    

    
if __name__ == '__main__':
    
    subj = sys.argv[1] # number of the subject, 1-8
    roi = str(sys.argv[2]) # ROI name, in ['V1', 'V2', 'V3', 'hV4', 'V3ab', 'LO', 'IPS', 'VO', 'PHC', 'MT', 'MST']
    if roi=='None':
        roi = None
        
    up_to_sess = sys.argv[3] # analyze sessions 1-##
    
    if len(sys.argv)>4:
        n_ori = sys.argv[4] # number of orientation channels to use
    else:
        n_ori = 36
        
    if len(sys.argv)>5:
        n_sf = sys.argv[5] # number of spatial frequency channels to use
    else:
        n_sf = 12
    
    if len(sys.argv)>6:
        sample_batch_size = sys.argv[6] # number of trials to analyze at once when making features (smaller will help with out-of-memory errors)
    else:
        sample_batch_size = 50
        
    if len(sys.argv)>7:
        voxel_batch_size=sys.argv[7] # number of voxels to analyze at once when fitting weights (smaller will help with out-of-memory errors)
    else:
        voxel_batch_size=100
    
    if len(sys.argv)>8:
        zscore_features=str(sys.argv[8])
        if zscore_features=='False':
            zscore_features=False
            print('skipping z-scoring of features')
        else:
            zscore_features=True
            print('will perform z-scoring of features')
            
    if len(sys.argv)>9:
        normalize_fn=str(sys.argv[9])
        if normalize_fn=='False':
            normalize_fn=False
            print('skipping normalizing fn')
        else:
            normalize_fn=True
            print('will use log(1+sqrt(x)) as normalizing fn')
        
        
    if len(sys.argv)>10:
        debug=str(sys.argv[10])
        if debug=='True':
            debug=True
            print('USING DEBUG MODE...')
        else:
            debug=False
    else:
        debug=False
        
    fit_gabor_fwrf(int(subj), roi, int(up_to_sess), int(n_ori), int(n_sf), int(sample_batch_size), int(voxel_batch_size), zscore_features, normalize_fn, debug)