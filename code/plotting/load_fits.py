import numpy as np
import sys
import os
import torch

"""
General use functions for loading/getting basic properties of encoding model fit results.
Input to most of these functions is 'out', which is a dictionary containing 
fit results. Created by the model fitting code in model_fitting/fit_model.py
"""

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
