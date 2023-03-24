import numpy as np
from utils import circ_utils
from feature_extraction import gabor_feature_extractor

"""
Code to analyze the estimated feature selectivity of voxele, based on encoding
model fit params (Gabor feature space).
"""

def get_gabor_feature_info(n_ori=12, n_sf=8, screen_eccen_deg=8.4):

    _gabor_ext_complex = gabor_feature_extractor.gabor_extractor_multi_scale(n_ori=n_ori, n_sf=n_sf)

    sf_cyc_per_stim = _gabor_ext_complex.feature_table['SF: cycles per stim']
    sf_cyc_per_deg = sf_cyc_per_stim/screen_eccen_deg
    sf_unique, sf_inds = np.unique(sf_cyc_per_deg, return_inverse=True)
   
    ori_deg = _gabor_ext_complex.feature_table['Orientation: degrees']
    ori_unique, orient_inds = np.unique(ori_deg, return_inverse=True)
    
    return sf_unique, ori_unique

def analyze_freq_peaks(est_tuning_curves, abv_thresh, peak_thresh=0.50,\
                         sf_unique=np.logspace(np.log10(3),np.log10(72),num = 12)/8.4 ):
    
    n_voxels, n_sf = est_tuning_curves.shape

    # First find all the local maxima
    # This can include the leftmost/rightmost point, if they are greater than
    # the points to their right/left.
    diffs = np.diff(np.concatenate([-2*np.ones((n_voxels,1)), \
                                est_tuning_curves, \
                                -2*np.ones((n_voxels,1))], axis=1), axis=1)
    is_peak = (diffs[:,0:n_sf]>0) & (diffs[:,1:n_sf+1]<0)
    peaks = [np.where(is_peak[vv,:])[0] for vv in range(n_voxels)]
   
    n_peaks = np.array([len(p) for p in peaks])
   
    max_n_peaks = np.max(n_peaks)
    
    # will only return results for voxels that are above the r2 threshold
    # return nans otherwise
    # abv_thresh = val_r2>r2_cutoff
    
    peak_ratios = np.full(shape=(n_voxels,max_n_peaks),fill_value=np.nan)
    n_peaks_nonneg = np.full(shape=(n_voxels,),fill_value=np.nan)
    n_peaks_abovethresh = np.full(shape=(n_voxels,),fill_value=np.nan)
    
    # listing the peaks for each voxel, sorted by value at the peak
    top_freqs = np.full(shape=(n_voxels,max_n_peaks),fill_value=np.nan)

    for vi in range(n_voxels):

        if not abv_thresh[vi]:
            continue

        peak_vals = est_tuning_curves[vi,peaks[vi]]
        min_val = np.min(est_tuning_curves[vi,:])
        max_val = np.max(est_tuning_curves[vi,:])
        
        # remove any peaks that are negative (shouldn't happen often)
        peak_vals_nonneg = peak_vals[peak_vals>=0]
        peak_inds_nonneg = peaks[vi][peak_vals>=0]

        n_peaks_nonneg[vi] = len(peak_vals_nonneg)

        if len(peak_vals_nonneg)>0:

            # sorting the peaks by the value at each peak
            sort_order = np.flip(np.argsort(peak_vals_nonneg))
            sorted_vals = peak_vals_nonneg[sort_order]
            sorted_inds = peak_inds_nonneg[sort_order]
            
            assert(sorted_vals[0]==max_val)

            # compare each secondary peak to the largest peak, compute a ratio
            sorted_heights = sorted_vals-min_val
            ratio = sorted_heights / sorted_heights[0]
#             ratio = sorted_vals / sorted_vals[0]
            peak_ratios[vi,0:len(peak_vals_nonneg)] = ratio

            # only keeping peaks that are some fraction relative to the biggest peak.
            n_peaks_count = np.sum(ratio>peak_thresh)
            n_peaks_abovethresh[vi] = n_peaks_count
            
            # find the actual orientations that corresponded to these peaks.
            top_freqs_count = sf_unique[sorted_inds[0:n_peaks_count]]
            top_freqs[vi,0:n_peaks_count] = top_freqs_count
            
        else:
            n_peaks_abovethresh[vi] = 0
            
    return n_peaks_abovethresh, top_freqs


def analyze_orient_peaks(est_tuning_curves, abv_thresh, r2_cutoff=0.10, peak_thresh=0.50,\
                         ori_unique = np.arange(0,165,12)):
    
    n_voxels = est_tuning_curves.shape[0]
    
    # first find all the local maxima and minima - just based on 
    # where the curve goes up/down.
    peaks = circ_utils.get_circ_peaks(est_tuning_curves)
    troughs = circ_utils.get_circ_troughs(est_tuning_curves)
    
    n_peaks = np.array([len(p) for p in peaks])
    n_troughs = np.array([len(t) for t in troughs])

    max_n_peaks = np.max(n_peaks)
    
    # will only return results for voxels that are above the r2 threshold
    # return nans otherwise
    # abv_thresh = val_r2>r2_cutoff
    
    peak_ratios = np.full(shape=(n_voxels,max_n_peaks),fill_value=np.nan)
    n_peaks_nonneg = np.full(shape=(n_voxels,),fill_value=np.nan)
    n_peaks_abovethresh = np.full(shape=(n_voxels,),fill_value=np.nan)
    
    # listing the peaks for each voxel, sorted by value at the peak
    top_orients = np.full(shape=(n_voxels,max_n_peaks),fill_value=np.nan)

    for vi in range(n_voxels):

        if not abv_thresh[vi]:
            continue

        peak_vals = est_tuning_curves[vi,peaks[vi]]
        min_val = np.min(est_tuning_curves[vi,:])
        max_val = np.max(est_tuning_curves[vi,:])
        
        # remove any peaks that are negative (shouldn't happen often)
        peak_vals_nonneg = peak_vals[peak_vals>=0]
        peak_inds_nonneg = peaks[vi][peak_vals>=0]

        n_peaks_nonneg[vi] = len(peak_vals_nonneg)

        if len(peak_vals_nonneg)>0:

            # sorting the peaks by the value at each peak
            sort_order = np.flip(np.argsort(peak_vals_nonneg))
            sorted_vals = peak_vals_nonneg[sort_order]
            sorted_inds = peak_inds_nonneg[sort_order]
            
            assert(sorted_vals[0]==max_val)

            # compare each secondary peak to the largest peak, compute a ratio
            sorted_heights = sorted_vals-min_val
            ratio = sorted_heights / sorted_heights[0]
#             ratio = sorted_vals / sorted_vals[0]
            peak_ratios[vi,0:len(peak_vals_nonneg)] = ratio

            # only keeping peaks that are some fraction relative to the biggest peak.
            n_peaks_count = np.sum(ratio>peak_thresh)
            n_peaks_abovethresh[vi] = n_peaks_count
            
            # find the actual orientations that corresponded to these peaks.
            top_orients_count = ori_unique[sorted_inds[0:n_peaks_count]]
            top_orients[vi,0:n_peaks_count] = top_orients_count
            
        else:
            n_peaks_abovethresh[vi] = 0
            
    return n_peaks_abovethresh, top_orients


def group_bimodal_voxels(n_peaks, top_orients, n_groups_use):
    
    has_two_peaks = n_peaks==2

    assert(not np.any(np.isnan(top_orients[has_two_peaks,0:2])))
    
    # for all the voxels that were bimodal, identify the unique sets of peaks
    # that the voxels had. Ignoring their rank here, assume the two peaks are the same.
    pairs = top_orients[has_two_peaks,0:2]
    pairs = np.array([np.sort(pairs[ii,:]) for ii in range(pairs.shape[0])])
    unpairs, counts = np.unique(pairs, axis=0, return_counts=True)

    # choose the most common sets of peaks among voxels.
    top_pairs = unpairs[np.flip(np.argsort(counts))[0:n_groups_use],:]
    pair_labels = ['peaks at %d/%d deg'%(np.round(top_pairs[ii,0]), \
                                         np.round(top_pairs[ii,1])) \
                                         for ii in range(n_groups_use)]
    # any voxels that don't fit in the most common groups will be in their own group
    pair_labels += ['Some other pair of orients']

    # label each bimodal voxel according to which group it falls into.
    which_group = np.full(shape=np.shape(n_peaks), fill_value=np.nan)
    
    for pp in range(n_groups_use):
        
        this_group_inds = has_two_peaks & \
                          np.all(np.isin(top_orients[:,0:2], top_pairs[pp,:]), axis=1)
        
        which_group[this_group_inds] = pp
        
    # label any voxels that are bimodal but didn't fall into one of the other groups
    which_group[has_two_peaks & np.isnan(which_group)] = pp+1
        
    return which_group, top_pairs, pair_labels

def group_trimodal_voxels(n_peaks, top_orients, n_groups_use):
    
    has_three_peaks = n_peaks==3

    assert(not np.any(np.isnan(top_orients[has_three_peaks,0:3])))
    
    # for all the voxels that were tri-modal, identify the unique sets of peaks
    # that the voxels had. Ignoring their rank here, assume the three peaks are the same.
    groups = top_orients[has_three_peaks,0:3]
    groups = np.array([np.sort(groups[ii,:]) for ii in range(groups.shape[0])])
    ungroups, counts = np.unique(groups, axis=0, return_counts=True)

    # choose the most common sets of peaks among voxels.
    top_groups = ungroups[np.flip(np.argsort(counts))[0:n_groups_use],:]
    group_labels = ['peaks at %d/%d/%d deg'%(np.round(top_groups[ii,0]), \
                                            np.round(top_groups[ii,1]), \
                                            np.round(top_groups[ii,2])) \
                                            for ii in range(n_groups_use)]
    # any voxels that don't fit in the most common groups will be in their own group
    group_labels += ['Some other orients']

    # label each tri-modal voxel according to which group it falls into.
    which_group = np.full(shape=np.shape(n_peaks), fill_value=np.nan)
    
    for pp in range(n_groups_use):
        
        this_group_inds = has_three_peaks & \
                          np.all(np.isin(top_orients[:,0:3], top_groups[pp,:]), axis=1)
        
        which_group[this_group_inds] = pp
        
    # label any voxels that are tri-modal but didn't fall into one of the other groups
    which_group[has_three_peaks & np.isnan(which_group)] = pp+1
        
    return which_group, top_groups, group_labels
