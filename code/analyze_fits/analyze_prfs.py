import os, sys
import numpy as np

from plotting import load_fits
from utils import roi_utils, default_paths, prf_utils

def compute_prf_coverage(subjects, fitting_type='alexnet_all_conv_pca', image_size=224, \
                        ignore_overlapping_voxels=True):
    
    n_subjects = len(subjects)
    out = [load_fits.load_fit_results(subject=ss, \
                                     fitting_type=fitting_type, \
                                     n_from_end=0, verbose=False, \
                                     return_filename=True) \
                                      for ss in subjects]
    filenames = [o[1] for o in out]
    out = [o[0] for o in out]
    
    prf_pars = np.concatenate([out[si]['best_params'][0][:,0,:] \
                               for si in range(n_subjects)], axis=0)
    
    x = prf_pars[:,0]
    y = prf_pars[:,1]
    sigma = prf_pars[:,2]
    
    # remove any very poorly fit voxels from this analysis
    r2_cutoff = 0.01
    val_r2 = np.concatenate([out[si]['val_r2'][:,0] for si in range(n_subjects)], axis=0)
    abv_thresh = val_r2>r2_cutoff

    if ignore_overlapping_voxels:
        roi_def = roi_utils.multi_subject_roi_def(subjects, remove_ret_overlap=True, remove_categ_overlap=True)
    else:
        roi_def = roi_utils.multi_subject_roi_def(subjects)
    roi_names =roi_def.roi_names
    n_rois = roi_def.n_rois
    
    n_vox_each_subj = [out[si]['best_params'][0].shape[0] for si in range(n_subjects)]
    subject_inds = np.concatenate([si*np.ones((n_vox_each_subj[si],),dtype=int) \
                               for si in range(n_subjects)], axis=0)
    
    # computing a single image for pRF coverage, for each ROI and subject
    all_mean_prfs = np.zeros((image_size, image_size, n_rois, n_subjects))
    all_max_prfs = np.zeros((image_size, image_size, n_rois, n_subjects))
    
    for si in range(n_subjects):

        for rr in range(n_rois):

            print('proc S%d, %s'%(si+1, roi_names[rr]))
            inds_this_roi = np.where(roi_def.get_indices(rr) & abv_thresh & (subject_inds==si))[0]

            roi_prfs = np.zeros((image_size, image_size, len(inds_this_roi)),dtype=float)

            for vi, vv in enumerate(inds_this_roi):

                # draw the pRF in image space
                prf = prf_utils.gauss_2d([x[vv], y[vv]], sigma[vv], image_size)
                roi_prfs[:,:,vi] = prf

            # trying two methods for combining across voxels, mean and max
            all_max_prfs[:,:,rr,si] = np.max(roi_prfs, axis=2)
            all_mean_prfs[:,:,rr,si] = np.mean(roi_prfs, axis=2)

    if ignore_overlapping_voxels:
        fn2save = os.path.join(default_paths.save_fits_path, 'prf_coverage', \
                               'All_pRFs_%s_no_roi_overlap_%dpix.npy'%(fitting_type, image_size))       
    else:
        fn2save = os.path.join(default_paths.save_fits_path, 'prf_coverage', \
                               'All_pRFs_%s_%dpix.npy'%(fitting_type, image_size))
    print('saving to %s'%fn2save)
    np.save(fn2save, {'all_max_prfs': all_max_prfs, \
                      'all_mean_prfs': all_mean_prfs, \
                      'filenames': filenames})
    
    
