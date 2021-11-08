
import os
import numpy as np
import h5py
from scipy.io import loadmat
import PIL.Image
import nibabel as nib
import pandas as pd
import pickle

from utils import default_paths, roi_utils

def get_paths():      
    return default_paths.nsd_root, default_paths.stim_root, default_paths.beta_root

nsd_root, stim_root, beta_root = get_paths()

trials_per_sess = 750
sess_per_subj = 40

def get_session_inds_full():
    session_inds = np.repeat(np.arange(0,sess_per_subj), trials_per_sess)
    return session_inds

def load_from_nii(nii_file):
    return nib.load(nii_file).get_fdata()
  
def load_from_mgz(mgz_file):
    return load_from_nii(mgz_file)
  
def load_from_hdf5(hdf5_file, keyname=None):
    data_set = h5py.File(hdf5_file, 'r')
    if keyname is None:
        keyname = list(data_set.keys())[0]
    values = np.copy(data_set[keyname])
    data_set.close()    
    return values

def image_uncolorize_fn(image):
    data = image.astype(np.float32) / 255
    return (0.2126*data[:,0:1]+ 0.7152*data[:,1:2]+ 0.0722*data[:,2:3])

def ncsnr_to_nc(ncsnr, n=1):    
    """
    From Allen, E. J., St-yves, G., Wu, Y., & Kay, K. N. (2021). A massive 7T fMRI dataset to bridge cognitive and computational neuroscience. BioRxiv, 1–70.
    Equation on page 47 of preprint.
    """   
    noise_ceiling = 100 * ncsnr**2 / (ncsnr**2 + 1/n)   
    return noise_ceiling
    
def get_image_data(subject, random_images=False, native=False):

    """
    Load the set of NSD images that were shown to a given subject.
    This loads a subject-specific array of images, see [get_subject_specific_images] for details.
    Can also choose to insert random noise instead of the images here.
    """
    
    if random_images==False:        
        print('\nLoading images for subject %d\n'%subject)
        if native:
            image_data = load_from_hdf5(os.path.join(stim_root, 'S%d_stimuli_native.h5py'%subject))     
        else:
            image_data = load_from_hdf5(os.path.join(stim_root, 'S%d_stimuli_240.h5py'%subject))        
    else:        
        print('\nGenerating random gaussian noise images...\n')
        n_images = 10000
        image_data = (np.random.normal(0,1,[n_images, 3, 240,240])*30+255/2).astype(np.uint8)
        image_data = np.maximum(np.minimum(image_data, 255),0)

    print ('image data size:', image_data.shape, ', dtype:', image_data.dtype, ', value range:',\
        np.min(image_data[0]), np.max(image_data[0]))

    return image_data

def get_master_image_order():    
    """
    Gather the "ordering" information for NSD images.
    masterordering gives zero-indexed ordering of indices (matlab-like to python-like), same for all subjects. 
    consists of 30000 values in the range [0-9999], which provide a list of trials in order. 
    The value in ordering[ii] tells the index into the subject-specific stimulus array that we would need to take to
    get the image for that trial.
    """
    exp_design_file = os.path.join(nsd_root, 'nsddata','experiments','nsd','nsd_expdesign.mat')
    exp_design = loadmat(exp_design_file)
    
    image_order = exp_design['masterordering'].flatten() - 1 
    
    return image_order
      

def load_betas(subject, sessions=[0], voxel_mask=None,  zscore_betas_within_sess=True, volume_space=True):

    """
    Load preprocessed voxel data for an NSD subject (beta weights).
    Always loading the betas with suffix 'fithrf_GLMdenoise_RR.
    Concatenate the values across multiple sessions.
    """
    
    if volume_space:
        beta_subj_folder = os.path.join(beta_root, 'subj%02d'%subject, 'func1pt8mm', 'betas_fithrf_GLMdenoise_RR')   
    else:
        beta_subj_folder = os.path.join(beta_root, 'subj%02d'%subject, 'nativesurface', 'betas_fithrf_GLMdenoise_RR')   

    print('Data is located in: %s...'%beta_subj_folder)

    n_trials = len(sessions)*trials_per_sess

    for ss, se in enumerate(sessions):

        if volume_space:

            # Load volume space nifti
            fn2load = os.path.join(beta_subj_folder, 'betas_session%02d.nii.gz'%(se+1))
            print('Loading from %s...'%fn2load)
            values = load_from_nii(fn2load).transpose((3,0,1,2))
            print('Raw data:')
            print(values.dtype, np.min(values), np.max(values), values.shape)

            betas = values.reshape((len(values), -1), order='C')

        else:
            # Surface space, concatenate the two hemispheres
            # Must be left then right to match ROI definitions.
            fn2load1 = os.path.join(beta_subj_folder, 'lh.betas_session%02d.hdf5'%(se+1))
            fn2load2 = os.path.join(beta_subj_folder, 'rh.betas_session%02d.hdf5'%(se+1))

            print('Loading from %s...'%fn2load1)        
            values1 = load_from_hdf5(fn2load1)
            print('Raw data:')
            print(values1.dtype, np.min(values1), np.max(values1), values1.shape)

            print('Loading from %s...'%fn2load2)        
            values2 = load_from_hdf5(fn2load2)
            print('Raw data:')
            print(values2.dtype, np.min(values2), np.max(values2), values2.shape)

            betas = np.concatenate((values1, values2), axis=1)

        # divide by 300 to convert back to percent signal change
        betas = betas.astype(np.float32) / 300

        print('Adjusted data (divided by 300):')
        print(betas.dtype, np.min(betas), np.max(betas), betas.shape)
        
        if voxel_mask is not None:        
            betas = betas[:,voxel_mask]

        if zscore_betas_within_sess: 
            print('z-scoring beta weights within this session...')
            mb = np.mean(betas, axis=0, keepdims=True)
            sb = np.std(betas, axis=0, keepdims=True)
            betas = np.nan_to_num((betas - mb) / (sb + 1e-6))
            print ("mean = %.3f, sigma = %.3f" % (np.mean(mb), np.mean(sb)))

        if ss==0:        
            n_vox = betas.shape[1]
            betas_full = np.zeros((n_trials, n_vox))   

        betas_full[ss*trials_per_sess : (ss+1)*trials_per_sess, :] = betas
        
    return betas_full
       

def get_data_splits(subject, sessions=[0], image_inds_only=False, voxel_mask=None, zscore_betas_within_sess=True, volume_space=True, \
                    shuffle_images=False, random_images=False, random_voxel_data=False):

    """
    Gather training/testing images and voxel data for one NSD subject.
    Always leaving out the "shared1000" image subset as my validation set, and training within the rest of the data.
    Can specify a list of sessions to work with (don't have to be contiguous).
    Can specify whether to work in volume or surface space (set volume_space to True or False).
    Can also choose to shuffle images, generate random images, or generate random voxel data at this stage.
    """
    
    # First load all the images, full brick of 10,000 images. Not in order yet.
    if not image_inds_only:
        image_data = get_image_data(subject, random_images)
        image_data = image_uncolorize_fn(image_data)
    else:
        image_data = None
    # Load the experiment design file that defines actual order.
    image_order = get_master_image_order()
    
    # Choosing which sessions we're analyzing now - same sessions as the voxel data that will be loaded.
    session_inds = get_session_inds_full()
    if np.isscalar(sessions):
        sessions = [sessions]
    sessions = np.array(sessions)
    inds2use = np.isin(session_inds, sessions)
    image_order = image_order[inds2use]
    
    if shuffle_images:
        shuff_order = np.arange(0,np.shape(image_order)[0])
        np.random.shuffle(shuff_order)
        print('\nShuffling image data...')
        print('\nShuffled order ranges from [%d to %d], first elements are:'%(np.min(shuff_order), \
                                                                              np.max(shuff_order)))
        print(shuff_order[0:10])        
        print('\nOrig image order ranges from [%d to %d], first elements are:'%(np.min(image_order), \
                                                                              np.max(image_order)))
        print(image_order[0:10])        
        image_order = image_order[shuff_order]
        print('\nNew image order ranges from [%d to %d], first elements are:'%(np.min(image_order), \
                                                                              np.max(image_order)))
        print(image_order[0:10])
        
    # Now re-ordering the image data into the real sequence, for just the sessions of interest.
    if not image_inds_only:
        image_data_ordered = image_data[image_order]
    
    # Now load voxel data (preprocessed beta weights for each trial)
    if random_voxel_data==False:
        # actual data loading here
        print('Loading data for sessions:')
        print(sessions+1)
        voxel_data = load_betas(subject, sessions, voxel_mask=voxel_mask, zscore_betas_within_sess=zscore_betas_within_sess, \
                                volume_space=volume_space)
        print('\nSize of full data set [nTrials x nVoxels] is:')
        print(voxel_data.shape)
    else:
        print('\nCreating fake random normal data...')
        n_voxels = 8000 # Just creating an array of random data
        voxel_data = np.random.normal(0,1,size=(trials_per_sess*len(sessions), n_voxels))
        print('\nSize of full data set [nTrials x nVoxels] is:')
        print(voxel_data.shape)

    # Split the data: the values in "image_order" < 1000 are the shared 1000 images, use these as validation set.
    shared_1000_inds = image_order<1000
   
    val_voxel_data = voxel_data[shared_1000_inds,:]
    trn_voxel_data = voxel_data[~shared_1000_inds,:]

    if not image_inds_only:
        val_stim_data = image_data_ordered[shared_1000_inds,:,:,:]
        trn_stim_data = image_data_ordered[~shared_1000_inds,:,:,:]
    else:
        val_stim_data = None
        trn_stim_data = None
        
    image_order_val = image_order[shared_1000_inds]
    image_order_trn = image_order[~shared_1000_inds]

    return trn_stim_data, trn_voxel_data, val_stim_data, val_voxel_data, image_order, image_order_trn, image_order_val

def resize_image_tensor(x, newsize):
        tt = x.transpose((0,2,3,1))
        r  = np.ndarray(shape=x.shape[:1]+newsize+(x.shape[1],), dtype=tt.dtype) 
        for i,t in enumerate(tt):
            r[i] = np.asarray(PIL.Image.fromarray(t).resize(newsize, resample=PIL.Image.BILINEAR))
        return r.transpose((0,3,1,2))   

def get_subject_specific_images(nsd_root, path_to_save, npix=227):

    """ 
    Load the big array of NSD images for all subjects.
    Downsample to a desired size, and select just those viewed by a given subject.
    Save a smaller array for each subject, at specified path.
    """
    
    stim_file_original = os.path.join(nsd_root,"nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5")
    exp_design_file = os.path.join(nsd_root,"nsddata/experiments/nsd/nsd_expdesign.mat")
    exp_design = loadmat(exp_design_file)
    subject_idx  = exp_design['subjectim']
    
    print ("Loading full block of images...")
    image_data_set = h5py.File(stim_file_original, 'r')
    print(image_data_set.keys())
    image_data = np.copy(image_data_set['imgBrick'])
    image_data_set.close()
    print(image_data.shape)

    for k,s_idx in enumerate(subject_idx):
        fn2save = os.path.join(path_to_save, 'S%d_stimuli_%d'%(k+1, npix))
        print('Will save to %s'%fn2save)       
        print('Resizing...')
        s_image_data = image_data[s_idx - 1]
        s_image_data = resize_image_tensor(s_image_data.transpose(0,3,1,2), newsize=(npix,npix))
        print(s_image_data.shape)        
        print('saving to %s'%fn2save)
        
        with h5py.File(fn2save + '.h5py', 'w') as hf:
            key='stimuli'
            val=s_image_data        
            hf.create_dataset(key,data=val)
            print ('saved %s in h5py file' %(key))


def get_subj_df(subject):
    """
    Get info about the 10,000 images that were shown to each subject.
    Note this is not the full ordered sequence of trials (which is 30,000 long)
    This is only the unique images 
    (matches what is in /user_data/mmhender/nsd_stimuli/stimuli/nsd/S1_stimuli....h5py)
    """
    exp_design_file = os.path.join(nsd_root,"nsddata/experiments/nsd/nsd_expdesign.mat")
    exp_design = loadmat(exp_design_file)
    subject_idx  = exp_design['subjectim']
    
    nsd_meta_file = os.path.join(nsd_root, 'nsddata/experiments/nsd/nsd_stim_info_merged.pkl')
    with open(nsd_meta_file,'rb') as f:
        stim_info = pickle.load(f,encoding="latin1")
    
    ss=subject-1
    subject_df = stim_info.loc[subject_idx[ss,:]-1]

    return subject_df


def load_prf_mapping_pars(subject, voxel_mask=None):
    
    """
    Load parameters of pRF fits for each voxel, obtained during independent pRF mapping expt.
    Stimuli are sweeping bars w objects, see:
    https://natural-scenes-dataset.s3-us-east-2.amazonaws.com/nsddata/experiments/prf/prf_screencapture.mp4
    """
    
    if voxel_mask is None:
        voxel_mask, _, _, _, _ = \
        roi_utils.get_voxel_roi_info(subject, volume_space=True, include_all=True, \
                           include_body=True,verbose=False)
        
    prf_path = os.path.join(default_paths.nsd_root, 'nsddata','ppdata','subj%02d'%subject,'func1pt8mm')

    angle = load_from_nii(os.path.join(prf_path, 'prf_angle.nii.gz')).flatten()[voxel_mask]
    eccen = load_from_nii(os.path.join(prf_path, 'prf_eccentricity.nii.gz')).flatten()[voxel_mask]
    size = load_from_nii(os.path.join(prf_path, 'prf_size.nii.gz')).flatten()[voxel_mask]
    exponent = load_from_nii(os.path.join(prf_path, 'prf_exponent.nii.gz')).flatten()[voxel_mask]
    gain = load_from_nii(os.path.join(prf_path, 'prf_gain.nii.gz')).flatten()[voxel_mask]
    rsq = load_from_nii(os.path.join(prf_path, 'prf_R2.nii.gz')).flatten()[voxel_mask]/100
            
    return angle, eccen, size, exponent, gain, rsq