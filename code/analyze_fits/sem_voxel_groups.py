import os, sys
import numpy as np
from plotting import load_fits
from utils import roi_utils, default_paths, stats_utils, nsd_utils
import copy

def get_sem_voxel_groups(n_vox_in_group = 500):
    
    """
    Defining pseudo-roi definitions, based on the selectivity of voxels 
    for different high-level categories.
    """
    
    subjects = np.arange(1,9)
    n_subjects = len(subjects)
   
    sem_groups = [[] for ss in subjects]

    for si, ss in enumerate(subjects):

        # load the semantic selectivity values for all voxels
        fitting_type = 'semantic_discrim_raw_trnval_all'

        out_sem = load_fits.load_fit_results(subject=ss, fitting_type=fitting_type,\
                                             n_from_end=0,verbose=False)    
        
        categ_names = out_sem['discrim_type_list']
        assert(out_sem['sem_partial_corrs'].shape[1]==len(categ_names))
        n_categ = len(categ_names)

        sem_group_names = categ_names
        n_sem_groups = n_categ
        
        c_partial = copy.deepcopy(out_sem['sem_partial_corrs']).T
        
        # check that there were enough trials to reliably compute
        # the partial correlation coefficients
        assert not np.any(out_sem['sem_partial_n_samp']==0)
        assert not np.any(np.isnan(out_sem['sem_partial_n_samp']))

        # loading the results of gabor model fits, use to threshold which voxels 
        # to include in these region definitions
        fitting_type1 = 'gabor_solo_ridge_12ori_8sf'
        out = load_fits.load_fit_results(subject=ss, fitting_type=fitting_type1, \
                                         n_from_end=0,verbose=False)

        fitting_type2 = 'gabor_solo_ridge_12ori_8sf_permutation_test'
        out_shuff = load_fits.load_fit_results(subject=ss, fitting_type=fitting_type2, \
                                         n_from_end=0,verbose=False)
        
        # find voxels that were significant based on permutation test
        r2_real_orig = out['val_r2']
        r2_shuff_orig = out_shuff['val_r2']

        # for how many of the shuffle iterations did shuffle-R2 exceed real-R2?
        p_orig = np.mean(r2_real_orig[:,0,None]<=r2_shuff_orig[:,0,:], axis=1)
        _,pvals_fdr_orig = stats_utils.fdr_keepshape(p_orig, alpha=0.01, \
                                                       method='poscorr')
        sig_orig = pvals_fdr_orig<0.01
        
        # loading the results of gabor model fits, use to threshold which voxels 
        # to include in these region definitions
        fitting_type3 = 'gabor_solo_ridge_12ori_8sf_from_residuals'
        out_resid = load_fits.load_fit_results(subject=ss, fitting_type=fitting_type3, \
                                         n_from_end=0,verbose=False)

        fitting_type4 = 'gabor_solo_ridge_12ori_8sf_from_residuals_permutation_test'
        out_resid_shuff = load_fits.load_fit_results(subject=ss, fitting_type=fitting_type4, \
                                         n_from_end=0,verbose=False)
        
        # find voxels that were significant based on permutation test
        r2_real_resid = out_resid['val_r2']
        r2_shuff_resid = out_resid_shuff['val_r2']

        # for how many of the shuffle iterations did shuffle-R2 exceed real-R2?
        p_resid = np.mean(r2_real_resid[:,0,None]<=r2_shuff_resid[:,0,:], axis=1)
        _,pvals_fdr_resid = stats_utils.fdr_keepshape(p_resid, alpha=0.01, \
                                                       method='poscorr')
        sig_resid = pvals_fdr_resid<0.01
        
        # also threshold voxels based on performance of the alexnet model
        # (this model was used to fit pRFs)
        fitting_type5 = 'alexnet_all_conv_pca'
        out_alexnet = load_fits.load_fit_results(subject=ss, fitting_type=fitting_type5, \
                                                 n_from_end=0, verbose=False)

        val_r2_alexnet = out_alexnet['val_r2'][:,0]
        abv_thresh_alexnet = val_r2_alexnet>0.01
        
        # exclude any early visual voxels (V1,V2,V3,hV4) from these high-level definitions.
        roi_def = roi_utils.nsd_roi_def(subject=ss, remove_ret_overlap=True, \
                                          remove_categ_overlap=True)
        early_visual_mask = np.any(np.array([roi_def.get_indices(ii) for ii in range(4)]), axis=0)

        nc = nsd_utils.get_nc(ss)
        
        # final set of voxels to exclude
        vox_exclude = ~sig_orig | ~sig_resid | early_visual_mask | \
                        (nc<0.01) | ~abv_thresh_alexnet

        c_partial[:,vox_exclude] = 0 
        # setting the values of the "excluded" voxels to zero, so that we won't pick them.

        n_vox = c_partial.shape[1]
        assert(n_vox>=n_vox_in_group)
        
        sem_groups[si] = np.zeros((n_vox, n_sem_groups),dtype=bool)

        for cc in range(n_categ):
            
            vals = c_partial[cc,:]
            
            # make sure there are actually enough positive values to pick from
            assert(np.sum(vals>0)>=n_vox_in_group)

            top_values = np.flip(np.argsort(vals))[0:n_vox_in_group]
           
            # all the voxels we choose should have positive correlation
            assert(np.all(vals[top_values]>0))
            
            sem_groups[si][:,cc] = np.isin(np.arange(n_vox), top_values)

        assert not np.any(np.any(sem_groups[si], axis=1) & vox_exclude)
        sem_groups[si][vox_exclude,:] = False
        
        
    return sem_groups, sem_group_names  
        