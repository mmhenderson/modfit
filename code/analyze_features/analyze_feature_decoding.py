import os
import numpy as np

import scipy.stats
import pandas as pd
import statsmodels

from utils import default_paths
from model_fitting import initialize_fitting



def get_path(feature_type):
    
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
    path_to_load = os.path.join(path_to_load, 'feature_decoding')
    
    return path_to_load

def load_decoding(feature_type, subject=999, which_prf_grid=5, \
                  balanced=False, verbose=False):

    path_to_load = get_path(feature_type)

    if balanced:
        fn1 = os.path.join(path_to_load, 'S%d_%s_LDA_all_grid%d_balanced.npy'%(subject, feature_type, which_prf_grid))
    else:
        fn1 = os.path.join(path_to_load, 'S%d_%s_LDA_all_grid%d.npy'%(subject, feature_type, which_prf_grid))
    if verbose:
        print('loading from %s'%fn1)
    decoding = np.load(fn1,allow_pickle=True).item()
    
    names = decoding['discrim_type_list']
    acc = decoding['acc']
    dprime = decoding['dprime']
    
    return acc, dprime, names


def analyze_decoding_slopes(subject, feature_type, which_prf_grid=5, \
                            balanced=False, \
                            rndseed=309468, n_iter=10000):
    
    acc, dprime, names = load_decoding(feature_type, subject=subject, balanced=balanced)
    n_axes = len(names)
    
    models = initialize_fitting.get_prf_models(which_grid=5)
    n_prfs = len(models)
    
    x = models[:,0]*8.4; y = models[:,1]*8.4;
    angles = np.round(np.mod(np.arctan2(y,x)*180/np.pi, 360),1)
    ecc = np.round(np.sqrt(models[:,0]**2+models[:,1]**2)*8.4, 4)
    no_angle = ecc<10**(-2)
    angles[no_angle] = np.nan
    sizes = np.round(models[:,2]*8.4, 4)
    
    ecc_vals = np.unique(ecc)
    size_vals = np.unique(sizes)
    n_ecc = len(ecc_vals);
    n_angles = len(np.unique(angles))-1
    n_sizes = len(size_vals)

    # removing any pRFs that are very peripheral - for these, not all sizes/angles
    # are evenly represented so the grid is un-balanced.
    # left with 1280 pRFs (assuming grid=5)
    counts = np.array([np.sum(ecc==ecc_vals[ee]) for ee in range(n_ecc)])
    ecc_use = counts==(n_angles*n_sizes)
    prfs_use = np.isin(ecc,ecc_vals[ecc_use])
    print('using %d pRFs'%np.sum(prfs_use))
    
    pars = [sizes, ecc, x, y]
    n_pars = len(pars)
    par_names = ['size', 'eccen','xpos', 'ypos']

    slope_values = np.zeros((n_axes, n_pars), dtype=float)
    slope_values_shuffle = np.zeros((n_axes, n_pars, n_iter), dtype=float)
   
    np.random.seed(rndseed)

    inter_values = np.zeros((n_axes, n_pars), dtype=float)
    r_values = np.zeros((n_axes, n_pars), dtype=float)
    p_values_parametric = np.zeros((n_axes,n_pars), dtype=float)

    for ai in range(n_axes):

        vals = dprime[prfs_use,ai]

        for pi, par in enumerate(pars):

            x_vals = par[prfs_use]

            reg_result = scipy.stats.linregress(x_vals, vals)

            slope_values[ai,pi] = reg_result.slope
            inter_values[ai,pi] = reg_result.intercept
            r_values[ai,pi] = reg_result.rvalue
            p_values_parametric[ai,pi] = reg_result.pvalue

            # permutation test to get a non-parametric p-value
            for xx in range(n_iter):

                shuff_order = np.random.permutation(np.arange(len(x_vals)))
                shuff_x_vals = x_vals[shuff_order]
                reg_result = scipy.stats.linregress(shuff_x_vals, vals)
                slope_values_shuffle[ai,pi,xx] = reg_result.slope

    # compute a two-tailed p-value here:
    p_values = np.minimum(np.mean(slope_values_shuffle<=slope_values[:,:,np.newaxis], axis=2), \
                          np.mean(slope_values_shuffle>=slope_values[:,:,np.newaxis], axis=2))*2
                            
    # FDR correction
    orig_shape = p_values.shape
    mask_fdr, pvals_fdr = statsmodels.stats.multitest.fdrcorrection(p_values.ravel(), alpha=0.01)
    mask_fdr = np.reshape(mask_fdr, orig_shape)
    pvals_fdr = np.reshape(pvals_fdr, orig_shape)
    mask_fdr

    # make a summary table
                            
    slopes_table = pd.DataFrame({}, index=names)
    for pi in range(n_pars):
        slopes_table['%s slope'%par_names[pi]] = slope_values[:,pi]
        slopes_table['%s intercept'%par_names[pi]] = inter_values[:,pi]
        slopes_table['%s corr'%par_names[pi]] = r_values[:,pi]
        slopes_table['%s pval'%par_names[pi]] = pvals_fdr[:,pi]
        slopes_table['%s fdr sig'%par_names[pi]] = mask_fdr[:,pi]
    
    path_to_save = get_path(feature_type)
       
    if balanced:
        fn2save = os.path.join(path_to_save, 'prf_decoding_slopes_balanced.csv')
    else:
        fn2save = os.path.join(path_to_save, 'prf_decoding_slopes.csv')
    print('saving to %s'%fn2save)    
    slopes_table.to_csv(fn2save)
    
    return