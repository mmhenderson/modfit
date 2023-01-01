import os
import numpy as np

from utils import default_paths

"""
Code to load/organize visual feature statistics and feature-semantic correlations
(files saved by get_feature_stats, get_feature_sem_corrs)
"""

def load_feature_stats(feature_type, subject='all', which_prf_grid=5):
    
    if (subject is None) or (subject=='all'):
        subject='All_trn'
    else:
        subject='S%d'%subject
        
    if 'gabor' in feature_type:
        path_to_load = default_paths.gabor_texture_feat_path
    elif 'texture' in feature_type:
        path_to_load = default_paths.pyramid_texture_feat_path
    elif 'sketch_tokens' in feature_type:
        path_to_load = default_paths.sketch_token_feat_path
    elif 'alexnet' in feature_type:
        path_to_load = default_paths.alexnet_feat_path
    elif 'clip' in feature_type:
        path_to_load = default_paths.clip_feat_path
    path_to_load = os.path.join(path_to_load, 'feature_stats')

    
    fn1 = os.path.join(path_to_load, '%s_%s_mean_grid%d.npy'%(subject, feature_type, which_prf_grid))
    fn2 = os.path.join(path_to_load, '%s_%s_var_grid%d.npy'%(subject, feature_type, which_prf_grid))
    fn3 = os.path.join(path_to_load, '%s_%s_covar_grid%d.npy'%(subject, feature_type, which_prf_grid))

    mean = np.load(fn1,allow_pickle=True)
    var = np.load(fn2,allow_pickle=True)
    covar = np.load(fn3,allow_pickle=True)
    
    return mean, var, covar

def load_feature_semantic_corrs(feature_type, subject='all', which_prf_grid=5, \
                               which_axes_negate = [1,2], min_samp=10, verbose=False):
    
    if (subject is None) or (subject=='all'):
        subject='All_trn'
    else:
        subject='S%d'%subject
        
    if 'gabor' in feature_type:
        path_to_load = default_paths.gabor_texture_feat_path
    elif 'texture' in feature_type:
        path_to_load = default_paths.pyramid_texture_feat_path
    elif 'sketch_tokens' in feature_type:
        path_to_load = default_paths.sketch_token_feat_path
    elif 'alexnet' in feature_type:
        path_to_load = default_paths.alexnet_feat_path
    elif 'clip' in feature_type:
        path_to_load = default_paths.clip_feat_path
    path_to_load = os.path.join(path_to_load, 'feature_stats')

    fn1 = os.path.join(path_to_load, '%s_%s_semantic_corrs_grid%d.npy'%(subject, feature_type, which_prf_grid))
    fn2 = os.path.join(path_to_load, '%s_%s_semantic_discrim_tstat_grid%d.npy'%(subject, feature_type, which_prf_grid))
    fn3 = os.path.join(path_to_load, '%s_%s_nsamp_grid%d.npy'%(subject, feature_type, which_prf_grid))

    corr = np.load(fn1,allow_pickle=True)
    discrim = np.load(fn2,allow_pickle=True)
    nsamp = np.load(fn3,allow_pickle=True)

    names, signed_names, n_levels = load_discrim_names(which_axes_negate)
    
    if min_samp is not None:
        # excluding any pRFs where there weren't enough trials per label to get a good estimate of 
        # our discriminability measures. This happens for instance for small pRFs and rare labels.
        # should not happen ever for the most high-level axes (animacy etc)
        exclude_prfs = [np.any(nsamp[:,ll,0:n_levels[ll]]<min_samp, axis=1) for ll in range(len(names))]    
        for ll in range(len(names)):
            if np.sum(exclude_prfs[ll])>0: 
                if verbose:
                    print('excluding %d pRFs for %s'%(np.sum(exclude_prfs[ll]), names[ll]))
                corr[:,exclude_prfs[ll],ll] = np.nan
                discrim[:,exclude_prfs[ll],ll] = np.nan
            
    for aa, name in enumerate(names):        
        if aa in which_axes_negate:
            corr[:,:,aa] *= (-1)
            discrim[:,:,aa] *= (-1)
    
    return corr, discrim, nsamp, names, signed_names


def load_discrim_names(which_axes_negate = [], return_levels=False):
    
    save_name_groups = os.path.join(default_paths.stim_labels_root,'Highlevel_concat_labelgroupnames.npy')
    groups = np.load(save_name_groups, allow_pickle=True).item()
    names = groups['discrim_type_list']
    levels = groups['col_names_all']
    n_levels = [len(ll) for ll in levels]
    
    signed_names = []
    for aa, name in enumerate(names):
        if len(levels[aa])>2:
            name = ''
        else:
            if 'has' in levels[aa][0]:
                level1 = levels[aa][0].split('has_')[1]
                level2 = levels[aa][1].split('has_')[1]
            else:
                level1 = levels[aa][0]
                level2 = levels[aa][1]

            if aa in which_axes_negate:
                name = '%s > %s'%(level1, level2)
            else:
                name = '%s > %s'%(level2, level1)

        signed_names.append(name)
        
    to_return = names, signed_names, n_levels
    if return_levels:
        to_return += levels,
        
    return to_return
    
    
def load_feature_semantic_partial_corrs(feature_type, subject='all', which_prf_grid=5, \
                                        axes_to_do_partial = [1,2,3], \
                                        which_axes_negate = [1], \
                                        min_samp=10, verbose=False):
    
    if (subject is None) or (subject=='all'):
        subject='All_trn'
    else:
        subject='S%d'%subject
        
    if 'gabor' in feature_type:
        path_to_load = default_paths.gabor_texture_feat_path
    elif 'texture' in feature_type:
        path_to_load = default_paths.pyramid_texture_feat_path
    elif 'sketch_tokens' in feature_type:
        path_to_load = default_paths.sketch_token_feat_path
    elif 'alexnet' in feature_type:
        path_to_load = default_paths.alexnet_feat_path
    elif 'clip' in feature_type:
        path_to_load = default_paths.clip_feat_path
    path_to_load = os.path.join(path_to_load, 'feature_stats')

    fn1 = os.path.join(path_to_load, '%s_%s_semantic_partial_corrs_grid%d.npy'%(subject, feature_type, which_prf_grid))
    fn2 = os.path.join(path_to_load, '%s_%s_nsamp_partial_corrs_grid%d.npy'%(subject, feature_type, which_prf_grid))

    partial_corr = np.load(fn1,allow_pickle=True)
    nsamp = np.load(fn2,allow_pickle=True)

    names, signed_names, n_levels = load_partial_discrim_names(axes_to_do_partial, which_axes_negate);

    if min_samp is not None:
        # excluding any pRFs where there weren't enough trials per label to get a good estimate of 
        # our discriminability measures. This happens for instance for small pRFs and rare labels.
        # should not happen ever for the most high-level axes (animacy etc)
        exclude_prfs = [np.any(nsamp[:,ll,0:n_levels[ll]]<min_samp, axis=1) for ll in range(len(names))]    
        for ll in range(len(names)):
            if np.sum(exclude_prfs[ll])>0: 
                if verbose:
                    print('excluding %d pRFs for %s'%(np.sum(exclude_prfs[ll]), names[ll]))
                partial_corr[:,exclude_prfs[ll],ll] = np.nan
                
    for aa, name in enumerate(names):       
        if aa in which_axes_negate:
            partial_corr[:,:,aa] *= (-1)
      
    return partial_corr, nsamp, names, signed_names


def load_partial_discrim_names(axes_to_do_partial = [0,1,2,3], which_axes_negate = []):
    
    save_name_groups = os.path.join(default_paths.stim_labels_root,'Highlevel_concat_labelgroupnames.npy')
    groups = np.load(save_name_groups, allow_pickle=True).item()
    names = [groups['discrim_type_list'][aa] for aa in axes_to_do_partial]
    levels = [groups['col_names_all'][aa] for aa in axes_to_do_partial]
    n_levels = [len(ll) for ll in levels]
    
    signed_names = []
    for aa, name in enumerate(names):
        if len(levels[aa])>2:
            name = ''
        else:
            if 'has' in levels[aa][0]:
                level1 = levels[aa][0].split('has_')[1]
                level2 = levels[aa][1].split('has_')[1]
            else:
                level1 = levels[aa][0]
                level2 = levels[aa][1]

            if aa in which_axes_negate:
                name = '%s > %s'%(level1, level2)
            else:
                name = '%s > %s'%(level2, level1)

        signed_names.append(name)
        
    return names, signed_names, n_levels
    