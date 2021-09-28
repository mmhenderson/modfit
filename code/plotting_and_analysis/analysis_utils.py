import numpy as np
import copy
import sys
import os
import torch

"""
General use functions for loading/getting basic properties of encoding model fit results.
Input to most of these functions is 'out', which is a dictionary containing 
fit results. Created by the model fitting code in model_fitting/fit_model.py
"""

from utils import roi_utils

def load_fit_results(subject, volume_space, fitting_type, n_from_end, root, verbose=True):
       
    if root is None:
        root = os.path.dirname(os.path.dirname(os.getcwd()))
    if volume_space:
        folder2load = os.path.join(root, 'model_fits','S%02d'%subject, fitting_type)
    else:
        folder2load = os.path.join(root, 'model_fits','S%02d_surface'%subject, fitting_type)
        
    # within this folder, assuming we want the most recent version that was saved
    files_in_dir = os.listdir(folder2load)
    from datetime import datetime
    my_dates = [f for f in files_in_dir if 'ipynb' not in f and 'DEBUG' not in f]
    try:
        my_dates.sort(key=lambda date: datetime.strptime(date, "%b-%d-%Y_%H%M_%S"))
    except:
        my_dates.sort(key=lambda date: datetime.strptime(date, "%b-%d-%Y_%H%M"))
    # if n from end is not zero, then going back further in time 
    most_recent_date = my_dates[-1-n_from_end]

    subfolder2load = os.path.join(folder2load, most_recent_date)
    file2load = os.listdir(subfolder2load)[0]
    fullfile2load = os.path.join(subfolder2load, file2load)

    if verbose:
        print('loading from %s\n'%fullfile2load)

    out = torch.load(fullfile2load)
    
    if verbose:
        print(out.keys())
        
    fig_save_folder = os.path.join(root,'figures','S%02d'%subject, fitting_type, most_recent_date)
    
    return out, fig_save_folder

def print_output_summary(out):
    """
    Print all the keys in the saved data file and a summary of each value.
    """
    for kk in out.keys():
        if out[kk] is not None:
            if np.isscalar(out[kk]):
                print('%s = %s'%(kk, out[kk]))
            elif isinstance(out[kk],tuple) or isinstance(out[kk],list):
                print('%s: len %s'%(kk, len(out[kk])))
            elif isinstance(out[kk],np.ndarray):
                print('%s: shape %s'%(kk, out[kk].shape))
            elif isinstance:
                print('%s: unknown'%kk)


def get_roi_info(subject, out, verbose=False):
    """
    Gather all information about roi definitions for the analyzed voxels.
    """
    voxel_roi = out['voxel_roi']
    voxel_idx = out['voxel_index'][0]
    
    assert(len(voxel_roi)==2)
    [roi_labels_retino, roi_labels_categ] = copy.deepcopy(voxel_roi)
    roi_labels_retino = roi_labels_retino[voxel_idx]
    roi_labels_categ = roi_labels_categ[voxel_idx]
    
    ret, face, place = roi_utils.load_roi_label_mapping(subject, verbose=verbose)
    
    max_ret_label = np.max(ret[0])
    face[0] = face[0]+max_ret_label
    max_face_label = np.max(face[0])
    place[0] = place[0]+max_face_label
    if verbose:
        print(face)
        print(place)
        print(np.unique(roi_labels_categ))

    ret_group_names = roi_utils.ret_group_names
    ret_group_inds =  roi_utils.ret_group_inds
    n_rois_ret = len(ret_group_names)

    categ_group_names = list(np.concatenate((face[1], place[1])))
    categ_group_inds =  list(np.concatenate((face[0], place[0])))
    n_rois_categ = len(categ_group_names)

    n_rois = n_rois_ret + n_rois_categ
    
    return roi_labels_retino, roi_labels_categ, ret_group_inds, categ_group_inds, ret_group_names, categ_group_names, \
                n_rois_ret, n_rois_categ, n_rois

def get_combined_rois(subject, out):
    
    roi_labels_retino, roi_labels_categ, ret_group_inds, categ_group_inds, ret_group_names, categ_group_names, \
        n_rois_ret, n_rois_categ, n_rois = get_roi_info(subject, out)

    retlabs = np.zeros(np.shape(roi_labels_retino))
    catlabs = np.zeros(np.shape(roi_labels_retino))

    for rr in range(n_rois_ret):   
        inds_this_roi = np.isin(roi_labels_retino, ret_group_inds[rr])
        retlabs[inds_this_roi] = rr+1

    for rr in range(n_rois_categ):   
        inds_this_roi = np.isin(roi_labels_categ, categ_group_inds[rr])
        catlabs[inds_this_roi] = rr+1

    return retlabs, catlabs, ret_group_names, categ_group_names


def get_r2(out):
    
#     val_cc = out['val_cc']
    # Note i'm NOT using the thing that actually is in the field val_r2, 
    # bc that is coefficient of determination which gives poor results for ridge regression.
    # instead using the signed squared correlation coefficient for r2/var explained.
#     val_r2 = np.sign(val_cc)*val_cc**2

    val_r2 = out['val_r2']

    return val_r2


# def process_two_way_var_part(out):
    
#     val_r2 = get_r2(out)
#     assert(val_r2.shape[1]==3)
#     assert(out['partial_version_names'][0]=='full_model')
#     assert(out['partial_version_names'][1].find('just_')==0)
#     assert(out['partial_version_names'][2].find('just_')==0)
#     name1 = out['partial_version_names'][1][5:]
#     name2 = out['partial_version_names'][2][5:]
#     names = ['full_model', 'unique: ' + name1, 'unique: ' + name2, 'shared']
#     shared_ab, unique_a, unique_b = get_shared_unique_var(val_r2[:,0], val_r2[:,1], val_r2[:,2])

#     var_expl = np.concatenate((val_r2[:,0:1], unique_a[:,np.newaxis], unique_b[:,np.newaxis],shared_ab[:,np.newaxis]),axis=1)
    
#     return var_expl, names


def get_shared_unique_var(combined, just_a, just_b):
    
    unique_a = combined - just_b
    unique_b = combined - just_a
    shared_ab = just_a + just_b - combined
   
    return shared_ab, unique_a, unique_b



