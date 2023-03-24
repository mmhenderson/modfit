import os, sys
import numpy as np

from plotting import load_fits
from utils import roi_utils, default_paths, prf_utils
from analyze_fits import sem_voxel_groups

def compute_prf_coverage(subjects, \
                         which_hemis='concat', \
                         fitting_type='alexnet_all_conv_pca', \
                         image_size=224):
    
    n_subjects = len(subjects)
    out = [load_fits.load_fit_results(subject=ss, \
                                     fitting_type=fitting_type, \
                                     n_from_end=0, verbose=False, \
                                     return_filename=True) \
                                      for ss in subjects]
    filenames = [o[1] for o in out]
    out = [o[0] for o in out]
    
    val_r2 = np.concatenate([out[si]['val_r2'][:,0] for si in range(n_subjects)], axis=0)
    
    prf_pars = np.concatenate([out[si]['best_params'][0][:,0,:] \
                               for si in range(n_subjects)], axis=0)
    
    x = prf_pars[:,0]
    y = prf_pars[:,1]
    sigma = prf_pars[:,2]
    
    # remove any very poorly fit voxels from this analysis
    r2_cutoff = 0.01    
    abv_thresh = val_r2>r2_cutoff

    roi_def = roi_utils.multi_subject_roi_def(subjects, which_hemis=which_hemis, \
                                              remove_ret_overlap=True, remove_categ_overlap=True)
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

            inds_this_roi = np.where(roi_def.get_indices(rr) & abv_thresh & (subject_inds==si))[0]
           
            print('proc S%d, %s: %d vox'%(si+1, roi_names[rr], len(inds_this_roi)))
            
            roi_prfs = np.zeros((image_size, image_size, len(inds_this_roi)),dtype=float)

            for vi, vv in enumerate(inds_this_roi):

                # draw the pRF in image space
                prf = prf_utils.gauss_2d([x[vv], y[vv]], sigma[vv], image_size)
                roi_prfs[:,:,vi] = prf

            # trying two methods for combining across voxels, mean and max
            all_max_prfs[:,:,rr,si] = np.max(roi_prfs, axis=2)
            all_mean_prfs[:,:,rr,si] = np.mean(roi_prfs, axis=2)

    fn2save = os.path.join(default_paths.save_fits_path, 'prf_coverage', \
                               'All_pRFs_%s'%fitting_type)
    if not os.path.exists(os.path.join(default_paths.save_fits_path, 'prf_coverage')):
        os.makedirs(os.path.join(default_paths.save_fits_path, 'prf_coverage'))
        
    if which_hemis!='concat':
        fn2save += '_%s_only'%which_hemis
        
    fn2save+= '_%dpix.npy'%(image_size)
    
    print('saving to %s'%fn2save)
    np.save(fn2save, {'all_max_prfs': all_max_prfs, \
                      'all_mean_prfs': all_mean_prfs, \
                      'filenames': filenames})
    
def compute_prf_coverage_bigroigroups(subjects, \
                         fitting_type='alexnet_all_conv_pca', \
                         image_size=224):
    
    n_subjects = len(subjects)
    out = [load_fits.load_fit_results(subject=ss, \
                                     fitting_type=fitting_type, \
                                     n_from_end=0, verbose=False, \
                                     return_filename=True) \
                                      for ss in subjects]
    filenames = [o[1] for o in out]
    out = [o[0] for o in out]
    
    val_r2 = np.concatenate([out[si]['val_r2'][:,0] for si in range(n_subjects)], axis=0)
    
    prf_pars = np.concatenate([out[si]['best_params'][0][:,0,:] \
                               for si in range(n_subjects)], axis=0)
    
    x = prf_pars[:,0]
    y = prf_pars[:,1]
    sigma = prf_pars[:,2]
    
    # remove any very poorly fit voxels from this analysis
    r2_cutoff = 0.01    
    abv_thresh = val_r2>r2_cutoff

    roi_def = roi_utils.multi_subject_roi_def(subjects, remove_ret_overlap=True, remove_categ_overlap=True)
    roi_names =roi_def.roi_names
    n_rois = roi_def.n_rois
    
    roi_groups = [[0,1,2,3],[6,7,8],[9,10]]
    roi_group_names = ['V1-hV4','place','face']
    n_roi_groups = len(roi_groups)
    
    n_vox_each_subj = [out[si]['best_params'][0].shape[0] for si in range(n_subjects)]
    subject_inds = np.concatenate([si*np.ones((n_vox_each_subj[si],),dtype=int) \
                               for si in range(n_subjects)], axis=0)
    
    # computing a single image for pRF coverage, for each ROI and subject
    all_mean_prfs = np.zeros((image_size, image_size, n_roi_groups, n_subjects))
    all_max_prfs = np.zeros((image_size, image_size, n_roi_groups, n_subjects))
    
    for si in range(n_subjects):

        for rr in range(n_roi_groups):
            
            inds_this_group = np.any(np.array([roi_def.get_indices(ri) for ri in roi_groups[rr]]), axis=0)

            inds_this_roi = np.where(inds_this_group & abv_thresh & (subject_inds==si))[0]
           
            print('proc S%d, %s: %d vox'%(si+1, roi_group_names[rr], len(inds_this_roi)))
            if si==0:
                print([roi_names[ri] for ri in roi_groups[rr]])
            
            roi_prfs = np.zeros((image_size, image_size, len(inds_this_roi)),dtype=float)

            for vi, vv in enumerate(inds_this_roi):

                # draw the pRF in image space
                prf = prf_utils.gauss_2d([x[vv], y[vv]], sigma[vv], image_size)
                roi_prfs[:,:,vi] = prf

            # trying two methods for combining across voxels, mean and max
            all_max_prfs[:,:,rr,si] = np.max(roi_prfs, axis=2)
            all_mean_prfs[:,:,rr,si] = np.mean(roi_prfs, axis=2)

    fn2save = os.path.join(default_paths.save_fits_path, 'prf_coverage', \
                               'All_pRFs_bigroigroups_%s'%fitting_type)
    if not os.path.exists(os.path.join(default_paths.save_fits_path, 'prf_coverage')):
        os.makedirs(os.path.join(default_paths.save_fits_path, 'prf_coverage'))
     
    fn2save+= '_%dpix.npy'%(image_size)
    
    print('saving to %s'%fn2save)
    np.save(fn2save, {'all_max_prfs': all_max_prfs, \
                      'all_mean_prfs': all_mean_prfs, \
                      'filenames': filenames})
        
def compute_prf_coverage_sem_roigroups(subjects, \
                         fitting_type='alexnet_all_conv_pca', \
                         image_size=224):
    
    n_subjects = len(subjects)
    out = [load_fits.load_fit_results(subject=ss, \
                                     fitting_type=fitting_type, \
                                     n_from_end=0, verbose=False, \
                                     return_filename=True) \
                                      for ss in subjects]
    filenames = [o[1] for o in out]
    out = [o[0] for o in out]
    
    val_r2 = np.concatenate([out[si]['val_r2'][:,0] for si in range(n_subjects)], axis=0)
    
    prf_pars = np.concatenate([out[si]['best_params'][0][:,0,:] \
                               for si in range(n_subjects)], axis=0)
    
    x = prf_pars[:,0]
    y = prf_pars[:,1]
    sigma = prf_pars[:,2]
    
    # remove any very poorly fit voxels from this analysis
    r2_cutoff = 0.01    
    abv_thresh = val_r2>r2_cutoff

    sem_groups, sem_group_names = sem_voxel_groups.get_sem_voxel_groups()
    n_sem_groups = len(sem_group_names)

    n_vox_each_subj = [out[si]['best_params'][0].shape[0] for si in range(n_subjects)]
    subject_inds = np.concatenate([si*np.ones((n_vox_each_subj[si],),dtype=int) \
                               for si in range(n_subjects)], axis=0)
    
    # computing a single image for pRF coverage, for each ROI and subject
    all_mean_prfs = np.zeros((image_size, image_size, n_sem_groups, n_subjects))
    all_max_prfs = np.zeros((image_size, image_size, n_sem_groups, n_subjects))
    
    for si in range(n_subjects):

        for rr in range(n_sem_groups):
            
            inds_this_group = np.concatenate(sem_groups, axis=0)[:,rr]
            
            assert(not np.any(inds_this_group & ~abv_thresh))
            
            inds_this_roi = np.where(inds_this_group & abv_thresh & (subject_inds==si))[0]
           
            print('proc S%d, %s: %d vox'%(si+1, sem_group_names[rr], len(inds_this_roi)))
            
            roi_prfs = np.zeros((image_size, image_size, len(inds_this_roi)),dtype=float)

            for vi, vv in enumerate(inds_this_roi):

                # draw the pRF in image space
                prf = prf_utils.gauss_2d([x[vv], y[vv]], sigma[vv], image_size)
                roi_prfs[:,:,vi] = prf

            # trying two methods for combining across voxels, mean and max
            all_max_prfs[:,:,rr,si] = np.max(roi_prfs, axis=2)
            all_mean_prfs[:,:,rr,si] = np.mean(roi_prfs, axis=2)

    fn2save = os.path.join(default_paths.save_fits_path, 'prf_coverage', \
                               'All_pRFs_semroigroups_%s'%fitting_type)
    
    fn2save+= '_%dpix.npy'%(image_size)
    
    print('saving to %s'%fn2save)
    np.save(fn2save, {'all_max_prfs': all_max_prfs, \
                      'all_mean_prfs': all_mean_prfs, \
                      'filenames': filenames})
        