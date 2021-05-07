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
from src.torch_fwrf_orig import learn_params_ridge_regression, get_predictions, get_value, set_value, Torch_fwRF_voxel_block
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



def fit_gabor_fwrf(subject=1, roi='V1', up_to_sess=1, n_ori=36, n_sf=12, sample_batch_size=50, voxel_batch_size=100, debug=False):
    
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
    
    data_size, nv = voxel_data.shape 
    trn_stim_data, trn_voxel_data,\
    val_stim_single_trial_data, val_voxel_single_trial_data,\
    val_stim_multi_trial_data, val_voxel_multi_trial_data = \
        data_split(image_uncolorize_fn(image_data), voxel_data, ordering, imagewise=False)

    
    ### DEFINE THE FILTERS FOR THE MODEL ###
    print('\nBuilding Gabor filter bank with %d orientations and %d spatial frequencies...'%(n_ori, n_sf))
    cyc_per_stim = logspace(n_sf)(3., 72.) # Which SF values are we sampling here?
    _gaborizer = Gaborizer(num_orientations=n_ori, cycles_per_stim=cyc_per_stim,
              pix_per_cycle=4.13, cycles_per_radius=.7, 
              radii_per_filter=4, complex_cell=True, pad_type='half', 
              crop=False).to(device)
    
    # adding a nonlinearity to the filter activations
    _fmaps_fn = add_nonlinearity(_gaborizer, lambda x: torch.log(1+torch.sqrt(x)))
    
    # pull out some relevant stuff from gaborizer object to save
    sf_tuning_masks = _gaborizer.sf_tuning_masks
    assert(np.all(_gaborizer.cyc_per_stim==cyc_per_stim))

    ori_tuning_masks = _gaborizer.ori_tuning_masks
    orients_deg = _gaborizer.orients_deg
    orient_filters = _gaborizer.orient_filters  
  
    ### PARAMS FOR THE RF CENTERS, SIZES ####
    aperture = np.float32(1)
    smin, smax = np.float32(0.04), np.float32(0.4)
    ns = 8

    # models is three columns, x, y, sigma
    models = model_space_pyramid(logspace(ns)(smin, smax), min_spacing=1.4, aperture=1.1*aperture)    

    ### PARAMS FOR RIDGE REGRESSION ####
    holdout_pct = 0.10
    holdout_size = int(np.ceil(np.shape(trn_voxel_data)[0]*holdout_pct))
#     lambdas = np.logspace(0.,5.,9, dtype=np.float32)
    lambdas = np.logspace(0.,0.,9, dtype=np.float32)
    
    #### DO THE ACTUAL MODEL FITTING HERE ####
    gc.collect()
    torch.cuda.empty_cache()

    best_losses, best_lambdas, best_params = learn_params_ridge_regression(
        trn_stim_data, trn_voxel_data, _fmaps_fn, models, lambdas, \
        aperture=aperture, _nonlinearity=None, zscore=True, sample_batch_size=sample_batch_size, \
        voxel_batch_size=voxel_batch_size, holdout_size=holdout_size, shuffle=False, add_bias=True, debug=debug)
    print('done with training')
#     del trn_stim_data
#     del trn_voxel_data
#     gc.collect()
#     torch.cuda.empty_cache()

    #### EVALUATE PERFORMANCE ON VALIDATION SET #####
    print('initializing model for validation...')
    param_batch = [p[:voxel_batch_size] if p is not None else None for p in best_params]
    # To initialize this module for prediction, need to take just first batch of voxels.
    # Will eventually pass all voxels through in batches.
    _fwrf_fn = Torch_fwRF_voxel_block(_fmaps_fn, param_batch, _nonlinearity=None, input_shape=val_stim_single_trial_data.shape, aperture=1.0)
    
    print('\nEvaluating model on validation set...\n')
    val_voxel_pred = get_predictions(val_stim_single_trial_data, _fmaps_fn, _fwrf_fn, best_params, sample_batch_size=sample_batch_size)
    
    # val_cc is the correlation coefficient bw real and predicted responses across trials, for each voxel.
    val_cc  = np.zeros(shape=(nv), dtype=fpX)
    val_r2 = np.zeros(shape=(nv), dtype=fpX)
    
    for v in tqdm(range(nv)):    
        val_cc[v] = np.corrcoef(val_voxel_single_trial_data[:,v], val_voxel_pred[:,v])[0,1]  
        val_r2[v] = get_r2(val_voxel_single_trial_data[:,v], val_voxel_pred[:,v])
        
    val_cc = np.nan_to_num(val_cc)
    val_r2 = np.nan_to_num(val_r2)
    
    #### ASSESS TUNING FOR SPATIAL FREQUENCY #####
    # this is done by measuring correlation coefficient, using subset of features at a time.
    # idea is that if you can get good prediction with just the features at a single SF, the voxel is likely responsive to it.
    # probably other ways to do this as well.
    partition_cc_sf   = np.ndarray(shape=(len(sf_tuning_masks),)+(nv,), dtype=fpX)   
    partition_r2_sf   = np.ndarray(shape=(len(sf_tuning_masks),)+(nv,), dtype=fpX) 
#     for l,rl in enumerate(sf_tuning_masks):
#         if debug and l>1:
#             break
#         print('Measuring model performance with just the features having SF value %d\n'%l)
#         partition_params = [np.copy(p) for p in best_params]
#         partition_params[1][:,:]   = 0   # setting to zero the other params
#         partition_params[1][:, rl] = best_params[1][:, rl]
        
#         val_voxel_pred = get_predictions(val_stim_single_trial_data, _fmaps_fn, _fwrf_fn, partition_params, sample_batch_size=sample_batch_size)
        
#         for v in tqdm(range(nv)):    
#             partition_cc_sf[l,v] = np.corrcoef(val_voxel_single_trial_data[:,v], val_voxel_pred[:,v])[0,1]
#             partition_r2_sf[l,v] = get_r2(val_voxel_single_trial_data[:,v], val_voxel_pred[:,v])[0,1]

#     partition_cc_sf = np.nan_to_num(partition_cc_sf)
#     partition_r2_sf = np.nan_to_num(partition_r2_sf)
    
            
    #### ASSESS TUNING FOR ORIENTATION #####
    # analogous to frequency method
    partition_cc_orient   = np.ndarray(shape=(len(ori_tuning_masks),)+(nv,), dtype=fpX)    
    partition_r2_orient   = np.ndarray(shape=(len(ori_tuning_masks),)+(nv,), dtype=fpX)    
#     for l,rl in enumerate(ori_tuning_masks):
#         if debug and l>1:
#             break
#         print('Measuring model performance with just the features having orientation value %d\n'%l)
#         partition_params = [np.copy(p) for p in best_params]
#         partition_params[1][:,:]   = 0    # setting to zero the other params
#         partition_params[1][:, rl] = best_params[1][:, rl]
        
#         val_voxel_pred = get_predictions(val_stim_single_trial_data, _fmaps_fn, _fwrf_fn, partition_params, sample_batch_size=sample_batch_size)

#         for v in tqdm(range(nv)):    
#             partition_cc_orient[l,v] = np.corrcoef(val_voxel_single_trial_data[:,v], val_voxel_pred[:,v])[0,1]
#             partition_r2_orient[l,v] = get_r2(val_voxel_single_trial_data[:,v], val_voxel_pred[:,v])[0,1]

#     partition_cc_orient = np.nan_to_num(partition_cc_orient)
#     partition_r2_orient = np.nan_to_num(partition_r2_orient)
    
    
    ### SAVE THE RESULTS TO DISK #########
    
    fn2save = output_dir+'model_params_%s'%roi2print
    print('\nSaving result to %s'%fn2save)
    
    torch.save({
    'feature_table': _gaborizer.feature_table,
    'sf_tuning_masks': sf_tuning_masks,
    'ori_tuning_masks': ori_tuning_masks,
    'cyc_per_stim': cyc_per_stim,
    'orients_deg': orients_deg,
    'orient_filters': orient_filters,
    'aperture': aperture,
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
    'partition_cc_sf': partition_cc_sf,
    'partition_r2_sf': partition_r2_sf,
    'partition_cc_orient': partition_cc_orient,
    'partition_r2_orient': partition_r2_orient,
    }, fn2save)
    
    
def get_r2(actual,predicted):
  
    # calculate r2 for this fit.
    ssres = np.sum(np.power((predicted - actual),2));
    print(ssres)
    sstot = np.sum(np.power((actual - np.mean(actual)),2));
    print(sstot)
    r2 = 1-(ssres/sstot)
    
    return r2


class add_nonlinearity(L.Module):
    def __init__(self, _fmaps_fn, _nonlinearity):
        super(add_nonlinearity, self).__init__()
        self.fmaps_fn = _fmaps_fn
        self.nl_fn = _nonlinearity
    def forward(self, _x):
        return [self.nl_fn(_fm) for _fm in self.fmaps_fn(_x)]
    
    
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
        debug=str(sys.argv[8])
        if debug=='True':
            debug=True
            print('USING DEBUG MODE...')
        else:
            debug=False
    else:
        debug=False
        
    fit_gabor_fwrf(int(subj), roi, int(up_to_sess), int(n_ori), int(n_sf), int(sample_batch_size), int(voxel_batch_size), debug)