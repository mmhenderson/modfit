"""
These are all semi-general functions that are run before model fitting (load/split data,  make feature extractor etc.)
"""

import torch
import time
import os
from scipy.io import loadmat
import h5py
import numpy as np

# import custom modules
from model_src import rf_grid, gabor_feature_extractor
from utils import file_utility, roi_utils, load_nsd


def init_cuda():
    
    # Set up CUDA stuff
    
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

    return device
    
def get_paths():
    
    # these are hard coded to work with mind cluster
    
    nsd_root = "/lab_data/tarrlab/common/datasets/NSD/"
    stim_root = "/user_data/mmhender/nsd_stimuli/stimuli/nsd/"    
    beta_root = nsd_root + "nsddata_betas/ppdata/"
    mask_root = nsd_root + "nsddata/ppdata/"
    
    return nsd_root, stim_root, beta_root, mask_root

def get_save_path(root_dir, subject, model_name, shuffle_images, random_images, random_voxel_data, debug):
    
    # choose where to save results of fitting - always making a new file w current timestamp.
    
    # add these suffixes to the file name if it's one of the control analyses
    if shuffle_images==True:
        model_name = model_name + '_SHUFFLEIMAGES'
    if random_images==True:
        model_name = model_name + '_RANDOMIMAGES'
    if random_voxel_data==True:
        model_name = model_name + '_RANDOMVOXELDATA'
    
    timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
    print ("Time Stamp: %s" % timestamp)  
    root_dir = os.path.dirname(root_dir)
    if debug==True:
        output_dir = os.path.join(root_dir, "model_fits/S%02d/%s/%s_DEBUG/" % (subject,model_name,timestamp) )
    else:
        output_dir = os.path.join(root_dir, "model_fits/S%02d/%s/%s/" % (subject,model_name,timestamp) )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    fn2save = os.path.join(output_dir,'all_fit_params')
    print('\nWill save final output file to %s\n'%output_dir)
    
    return output_dir, fn2save
 
def get_fitting_pars(trn_voxel_data, zscore_features=True, ridge=True, holdout_pct=0.10):

    holdout_size = int(np.ceil(np.shape(trn_voxel_data)[0]*holdout_pct))

    if ridge==True:
        if zscore_features==True:
            lambdas = np.logspace(0.,5.,9, dtype=np.float32)
        else:
            lambdas = np.logspace(-6., 1., 9).astype(np.float64)
    else:
        # putting in two zeros because the code might break with a singleton dimension for lambdas.
        lambdas = np.array([0.0,0.0])
        
    print('\nPossible lambda values are:')
    print(lambdas)

    return holdout_size, lambdas


def get_prf_models():

    # sizes/centers; hard coded, taken from OSF code for fwrf fitting
    
    aperture = np.float32(1)
    smin, smax = np.float32(0.04), np.float32(0.4)
    n_sizes = 8

    # models is three columns, x, y, sigma
    models = rf_grid.model_space_pyramid(rf_grid.logspace(n_sizes)(smin, smax), min_spacing=1.4, aperture=1.1*aperture)    

    return aperture, models

def get_feature_map_fn(n_ori, n_sf, padding_mode, device, nonlin_fn):
    
    cyc_per_stim = rf_grid.logspace(n_sf)(3., 72.) # Which SF values are we sampling here?
    _gaborizer = gabor_feature_extractor.Gaborizer(num_orientations=n_ori, cycles_per_stim=cyc_per_stim,
              pix_per_cycle=4.13, cycles_per_radius=.7, 
              radii_per_filter=4, complex_cell=True, pad_type='half', padding_mode = padding_mode,
              crop=False).to(device)
    assert(np.all(_gaborizer.cyc_per_stim==cyc_per_stim))
    
    if nonlin_fn:
        # adding a nonlinearity to the filter activations
        print('\nAdding log(1+sqrt(x)) as nonlinearity fn...')
        _fmaps_fn = gabor_feature_extractor.add_nonlinearity(_gaborizer, lambda x: torch.log(1+torch.sqrt(x)))
    else:
        _fmaps_fn = _gaborizer
    
    return  _gaborizer, _fmaps_fn


def get_data_splits(nsd_root, beta_root, stim_root, subject, voxel_mask, up_to_sess, shuffle_images=False, random_images=False, random_voxel_data=False):
    
    # first load images and information abt them
    image_data, image_order = get_image_data(nsd_root, stim_root, subject, shuffle_images, random_images)
    
    # Now load voxel data (preprocessed beta weights for each trial)
    if random_voxel_data==False:
        # actual data loading here
        print('Loading data for sessions 1:%d...'%up_to_sess)
        beta_subj_folder = os.path.join(beta_root, 'subj%02d'%subject, 'func1pt8mm', 'betas_fithrf_GLMdenoise_RR')   
        print(beta_subj_folder)
        voxel_data, filenames = load_nsd.load_betas(folder_name=beta_subj_folder, zscore=True, voxel_mask=voxel_mask, up_to=up_to_sess, load_ext=".nii")
        print('\nSize of full data set [nTrials x nVoxels] is:')
        print(voxel_data.shape)
    else:
        print('\nCreating fake random normal data...')
        n_voxels = 11694 # hard coded to match expected nifti size
        voxel_data = np.random.normal(0,1,size=(750*up_to_sess, n_voxels))
        print('\nSize of full data set [nTrials x nVoxels] is:')
        print(voxel_data.shape)
    
    # split into train and validation sets, always leaving fixed set for validation (the shared 1000 images)
    data_size, n_voxels = voxel_data.shape 
    trn_stim_data, trn_voxel_data,\
    val_stim_single_trial_data, val_voxel_single_trial_data,\
    val_stim_multi_trial_data, val_voxel_multi_trial_data = \
        load_nsd.data_split(load_nsd.image_uncolorize_fn(image_data), voxel_data, image_order, imagewise=False)
    n_trials_val = val_stim_single_trial_data.shape[0]
    
    return trn_stim_data, trn_voxel_data, val_stim_single_trial_data, val_voxel_single_trial_data, n_voxels, n_trials_val, image_order


def get_image_data(nsd_root, stim_root, subject, shuffle_images=False, random_images=False):

    ## GATHERING IMAGES/EXPERIMENT INFO ##
    exp_design_file = nsd_root + "nsddata/experiments/nsd/nsd_expdesign.mat"
    exp_design = loadmat(exp_design_file)
    image_order = exp_design['masterordering'].flatten() - 1 # zero-indexed ordering of indices (matlab-like to python-like)
    # ordering is 30000 values in the range [0-9999], which provide a list of trials in order. 
    # the value in ordering[ii] tells the index into the subject-specific stimulus array that we would need to take to
    # get the image for that trial.

    if random_images==False:
        print('\nLoading images for subject %d\n'%subject)
        image_data = {}
        image_data_set = h5py.File(stim_root + "S%d_stimuli_227.h5py"%subject, 'r')
        image_data = np.copy(image_data_set['stimuli'])
        image_data_set.close()
    else:
        print('\nGenerating random gaussian noise images...\n')
        n_images = 10000
        image_data = (np.random.normal(0,1,[n_images, 3, 227, 227])*30+255/2).astype(np.uint8)
        image_data = np.maximum(np.minimum(image_data, 255),0)
        
    print ('image data size:', image_data.shape, ', dtype:', image_data.dtype, ', value range:',\
        np.min(image_data[0]), np.max(image_data[0]))

    if shuffle_images==True:
        
        shuff_order = np.arange(0,np.shape(image_data)[0])
        np.random.shuffle(shuff_order)
        print('\nShuffling image data...')
        print('\nShuffled order ranges from [%d to %d], first elements are:'%(np.min(shuff_order), np.max(shuff_order)))
        print(shuff_order[0:10])
        print('size of orig data matrix:')
        print(np.shape(image_data))
        image_data = image_data[shuff_order,:,:,:]
        print('size of shuffled data matrix:')
        print(np.shape(image_data))

        
    return image_data, image_order


def get_voxel_info(mask_root, beta_root, subject, roi):
       
    # Define which ROIs/set of voxels to use here. If roi=None, using all voxels.

    group_names = ['V1', 'V2', 'V3', 'hV4', 'V3ab', 'LO', 'IPS', 'VO', 'PHC', 'MT', 'MST', 'other']
    group = [[1,2],[3,4],[5,6], [7], [16, 17], [14, 15], [18,19,20,21,22,23], [8, 9], [10,11], [13], [12], [24,25,0]]

    # using full brain mask.
    voxel_mask_full = file_utility.load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/brainmask.nii.gz"%subject)
    voxel_roi_full  = file_utility.load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/prf-visualrois.nii.gz"%subject)
    voxel_kast_full = file_utility.load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/Kastner2015.nii.gz"%(subject))
    general_mask_full  = file_utility.load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/nsdgeneral.nii.gz"%(subject))
    ncsnr_full = file_utility.load_mask_from_nii(beta_root + "subj%02d/func1pt8mm/betas_fithrf_GLMdenoise_RR/ncsnr.nii.gz"%subject)
    brain_nii_shape = voxel_roi_full.shape

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

    print('\nSizes of all defined ROIs in this subject:')
    vox_total = 0
    for roi_mask, roi_name in roi_utils.iterate_roi(group, voxel_roi, roi_utils.roi_map, group_name=group_names):
        vox_total = vox_total + np.sum(roi_mask)
        print ("%d \t: %s" % (np.sum(roi_mask), roi_name))

    print ("%d \t: Total" % (vox_total))

    # Now decide if we want to use a single ROI, or all
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

    return voxel_mask_this_roi, voxel_idx_this_roi, voxel_roi, voxel_ncsnr, np.array(brain_nii_shape)

