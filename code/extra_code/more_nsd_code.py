""" 
OLD versions - may be useful in future
General code related to working with NSD data.
"""

import numpy as np
import pandas as pd
import pickle
import os
import time
import h5py

import torch
from torch.utils.data import Dataset

import nibabel as nib

def get_subj_stim_seq(subj,
                     nsd_meta_file = '/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl',
                     trials_per_sess = 750):

    with open(nsd_meta_file,'rb') as f:
        stim_info = pickle.load(f,encoding="latin1")
    
    # Making a smaller version of the big NSD data frame, re-sorted according to this subject's stim sequence in the expt.
    # Will be 30,000 rows
    # Takes in filename of the pd.dataframe containing NSD stim sequence info (nsd_stim_info_merged.pkl)
    
    inds_rep1 = np.array(stim_info['subject%d_rep0'%subj])
    inds_rep2 = np.array(stim_info['subject%d_rep1'%subj])
    inds_rep3 = np.array(stim_info['subject%d_rep2'%subj])
    all_inds = np.concatenate((inds_rep1,inds_rep2,inds_rep3),0)
    # check the lists of the repetition numbers shouldn't have overlap
    assert(np.intersect1d(inds_rep1,inds_rep2)==[0])
    assert(np.intersect1d(inds_rep1,inds_rep3)==[0])
    assert(np.intersect1d(inds_rep2,inds_rep3)==[0])
    # making sure these inds span the range of trial numbers we expect (plus a zero for empty space)
    assert(np.all(np.unique(all_inds)==np.arange(0,30000+1)))

    trial_inds = all_inds[all_inds>0]
    sort_order=np.argsort(trial_inds)

    subject_df = pd.DataFrame([])
    # trial number and sess num - note these are one-indexed
    subject_df['trial_num'] = trial_inds[sort_order]    
    subject_df['sess_num'] = (np.floor((trial_inds[sort_order] -1)/trials_per_sess)+1).astype('int')
    
    # loop over all fields in original DF (including those that are for other subjs too)
    for ii,kk in enumerate(stim_info.keys()):

        concatlist = np.concatenate((stim_info[kk][inds_rep1>0], 
                                    stim_info[kk][inds_rep2>0],
                                    stim_info[kk][inds_rep3>0]))

        subject_df[kk] = concatlist[sort_order]
        
    assert(subject_df.loc[0]['subject%d_rep0'%subj]==1)
    
    return subject_df


def get_coco_ids_by_subject(subj, trial_inds, 
                            nsd_meta_file = '/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl'):
    """ 
    Get the COCO IDs of images shown to a subject on any set of trials.
    """

    subject_df = get_subj_stim_seq(subj, nsd_meta_file)    
    
    coco_ids = np.array(subject_df.loc[trial_inds]['cocoId'])
    
    return coco_ids



def get_nsd_inds_by_subject(subj, trial_inds, 
                           nsd_meta_file = '/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl'):
    """
    For a given set of trials for a given subject, find NSD indices for the viewed images.
    These are indices 0-73000 into the big image brick.
    """
    with open(nsd_meta_file,'rb') as f:
        stim_info = pickle.load(f,encoding="latin1")

    subject_df = get_subj_stim_seq(subj, nsd_meta_file)
        
    nsd_inds = np.array(subject_df.loc[trial_inds]['nsdId'])
        
    # make sure the ordering is correct
    assert(np.all(np.array(stim_info.loc[nsd_inds]['nsdId'])==nsd_inds))

    return nsd_inds



def nsd_inds_each_sess(sess, trials_per_sess=750):
    """ 
    Return indices of all trial numbers within the desired sessions.
    Note that these indices will start from zero.
    Session numbers start from 1.
    """
    if not hasattr(sess, "__len__"):
        sess = [sess]
    sess = list(sess)   
    inds = []
    for se in sess:
        inds = np.concatenate((inds, np.arange((se-1)*trials_per_sess, se*trials_per_sess,1)))
        
    return inds
   
    
def get_roi_voxels(subj, roi_label=1, roi_filename='lh.prf-visualrois.nii.gz', nsd_dir = '/lab_data/tarrlab/common/datasets/NSD/'):
    """
    Return indices for the voxels belonging to each ROI.
    Use the ROI def specified in roi_filename (a volume which can have multiple ROIs labeled)
    Take all voxels with values of roi_ind.
    """
    subj_roi_dir = os.path.join(nsd_dir, 'nsddata','ppdata','subj%02d'%subj,
                       'func1pt8mm','roi')
    roi_fn = os.path.join(subj_roi_dir, roi_filename)
    roi_nii = nib.load(roi_fn)
    
    roi_vol = roi_nii.get_fdata()
    # want it to match dims of functional volume data
    roi_vol = np.moveaxis(roi_vol, [0,1,2],[2,1,0])
    
    voxel_inds = np.where(roi_vol==roi_label)
    
    return voxel_inds


class nsd_dataset(Dataset):
  
  """
  Custom dataset class that loads NSD data (voxels), and deals with corresponding image labels.
  Returns a Pytorch dataset object. Can access items with <object>[indices].
  Items are objects with fields 'data' [nTrials x nVoxels]
  and 'subject_df' which is a pandas dataframe listing everything about those trials. 
  """ 

  def __init__(self, subj, voxel_inds, trials_per_sess=750, num_sess = 40,  
               nsd_dir = '/lab_data/tarrlab/common/datasets/NSD/',
               nsd_meta_file = '/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl'):

    self.subj = subj;
    self.voxel_inds = voxel_inds;
    self.num_vox = np.shape(voxel_inds)[1]
   
    self.trials_per_sess=trials_per_sess
    self.num_sess=num_sess
    
    subj_data_dir = os.path.join(nsd_dir,
                 'nsddata_betas','ppdata','subj%02d'%subj,'func1pt8mm',
                             'betas_fithrf_GLMdenoise_RR')
    
    self.subj_data_dir = subj_data_dir
    self.nsd_meta_file = nsd_meta_file

  def __len__(self):

    if hasattr(self, 'data'):
        mylen = np.shape(self.data)[0]
    else:
        mylen = self.num_sess*self.trials_per_sess
        
    return mylen

  def __getitem__(self, trial_inds):

    if hasattr(trial_inds, '__len__')==True:
        nTrials = len(trial_inds)
    else:
        nTrials=1

    trial_inds = np.squeeze(np.array([trial_inds]))
    
    which_sess = np.floor(trial_inds/self.trials_per_sess)+1
    sess2load = np.unique(which_sess)
    data  = np.zeros([nTrials, self.num_vox])
    
    # looping over all the sessions we need data from
    for se in sess2load:
        
        trials_this_sess = (trial_inds[which_sess==se] - (se-1)*self.trials_per_sess).astype('int')        
#         print(trials_this_sess)
        
        # Load in the data for this session (slow)
        nii_fn = os.path.join(self.subj_data_dir,'betas_session%02d.hdf5'%se)
#         print('loading from %s'%nii_fn)
        
        if len(trials_this_sess)==1:
#             print('loading trial of interest')
            # if just one trial in this session, can make this a bit faster by indexing rather than loading all data.
            # otherwise will load whole file then choose trials of interest.
            t = time.time()
            with h5py.File(nii_fn, "r") as f:
                data_this_sess = np.array(f['betas'][trials_this_sess,:,:,:])
            elapsed = time.time() - t
            print('h5py loading time is %.2f'%elapsed)
        else:
#             print('loading whole file for this sess')
            t = time.time()
            with h5py.File(nii_fn, "r") as f:
                data_this_sess = np.array(f['betas'])
            elapsed = time.time() - t
#             print('h5py loading time is %.2f'%elapsed)

            # grab just the trials of interest
            data_this_sess = data_this_sess[trials_this_sess,:,:,:]

        # adjusting for the 300x scaling
        data_this_sess = data_this_sess/300
        
        # All NIFTI files in the prepared NSD data are in LPI ordering
        # (the first voxel is Left, Posterior, and Inferior)
        data_this_sess = data_this_sess[:,self.voxel_inds[0],self.voxel_inds[1],self.voxel_inds[2]]
        
#         print(which_sess)
        # putting data from this session into the correct rows of our big matrix
        data[which_sess==se, :] = data_this_sess
        
    assert(np.shape(data)[0]==nTrials)

    df = get_subj_stim_seq(self.subj).loc[trial_inds.astype(int)]
                            
    item = {'data': data, 'subject_df': df}

    return item 



def get_dataset_splits(dset, subset_trn = False, rndseed=None):
    """ 
    Given a NSD dataset object, split into training and validation sets.
    Can choose to also leave out a subset of training data for hyperparameter 
    (e.g. ridge lambda) tuning. Returns "data subset" objects.
    """

    nTotal = dset.trials_per_sess*dset.num_sess
    subject_df = get_subj_stim_seq(dset.subj)
    
    # Validation set is the shared 1000 images, makes it easy to later compare across subs
    val_inds = np.where(subject_df['shared1000']==True)[0]
    # Training set is remaining images 
    trn_inds = np.where(subject_df['shared1000']==False)[0]
    
    if subset_trn:
        # If we want to solve for ridge parameter, can take out an extra part of trn set.
        nTrnSubset= int(0.20*len(trn_inds))
        if rndseed==None:
            rndseed = int(str(time.time())[-6:-1])
        np.random.seed(rndseed)
        shuff_inds = np.random.permutation(trn_inds)
        subset_inds = shuff_inds[0:nTrnSubset]
        trn_inds = shuff_inds[nTrnSubset:]
    else:
        subset_inds = []

    # random_split will create 3 new datasets, each of which is a subset of shape_dataset - have same fields etc but only subset of items.
    trn = torch.utils.data.Subset(dset, trn_inds);
    val = torch.utils.data.Subset(dset, val_inds)
    trn_subset = torch.utils.data.Subset(dset, subset_inds)
    
    return trn, val, trn_subset



def nsd_dataloader(dset, batch_size, shuffle=True, rndseed=None):
    """
    Load batches of NSD data. Returns a "generator" object, to get the next item 
    in the sequence use <object>.__next__()
    """
#     print('inside custom dataloader')
    num_samples = int(len(dset))
    
    sample_order = np.arange(0,num_samples)
    if shuffle:
        if rndseed==None:
            rndseed = int(str(time.time())[-6:-1])
        np.random.seed(rndseed)
        sample_order = np.random.permutation(sample_order)
        
    num_batches = int(np.ceil(num_samples/batch_size))
    
    for bb in range(num_batches):
        
        batch_inds = sample_order[bb*batch_size:(bb+1)*batch_size]
        if np.any(batch_inds>num_samples):
            batch_inds = batch_inds[batch_inds<num_samples]
        
        yield dset[batch_inds]


def invertible_sort(sequence):
    """
    Sort a sequence and store the order needed to reverse sort.
    Based on np.argsort.
    """
    order2sort = np.argsort(sequence)
    order2reverse = np.argsort(order2sort)
    
    return order2sort, order2reverse