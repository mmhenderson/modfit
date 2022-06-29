import numpy as np
import os, h5py
from utils import default_paths

# Some useful functions for figuring out which columns in big texture statistics matrix go to which features.

feature_types_all = ['pixel_stats', 'mean_magnitudes', 'mean_realparts', \
                     'marginal_stats_lowpass_recons', 'variance_highpass_resid', \
                     'magnitude_feature_autocorrs', 'lowpass_recon_autocorrs', \
                     'highpass_resid_autocorrs', \
                     'magnitude_within_scale_crosscorrs', 'real_within_scale_crosscorrs', \
                     'magnitude_across_scale_crosscorrs', 'real_imag_across_scale_crosscorrs', \
                     'real_spatshift_within_scale_crosscorrs', 'real_spatshift_across_scale_crosscorrs']

feature_type_dims_all = [6,16,16,10,1,272,73,25,24,24,48,96,10,20]

feature_names_simple = ['marginal', 'mean-magnitude', 'mean-real', \
                        'marginal-lowpass', 'marginal-highpass', \
                        'magnitude-autocorr', 'lowpass-autocorr', 'highpass-autocorr', \
                        'magnitude-cross-orient', 'real-cross-orient', \
                        'magnitude-cross-scale', 'real-cross-scale', \
                        'lowpass-real-within-scale', 'lowpass-real-cross-scale']


def get_feature_inds():    
    # Numbers that define which feature types are in which columns of final output matrix
    feature_column_labels = np.squeeze(np.concatenate([fi*np.ones([1,feature_type_dims_all[fi]]) \
                                for fi in range(len(feature_type_dims_all))], axis=1).astype('int'))    
    return feature_column_labels, feature_types_all

def get_feature_inds_lowhigh():    
    
    # Numbers that define which feature types are in which columns of final output matrix
    n_total_features = np.sum(feature_type_dims_all)
    n_ll_features = np.sum(feature_type_dims_all[0:5])
    feature_column_labels = (np.arange(n_total_features)>=n_ll_features).astype(int)
    
    return feature_column_labels, ['lower-level', 'higher-level']

def is_low_level():
    
    is_ll = get_feature_inds_lowhigh()[0]==0
    
    return is_ll

def get_feature_inds_simplegroups():
    
    # These are the final 10 sets of features that we used.
    # They combine a few of the subsets that have similar kinds of features.
    features_to_groups = np.array([0,1,2,3,3,4,5,5,6,7,8,9,7,9])
    group_names = ['pixel','energy-mean','linear-mean','marginal',\
                   'energy-auto','linear-auto',\
                   'energy-cross-orient','linear-cross-orient',\
                   'energy-cross-scale','linear-cross-scale']
    feature_column_labels = get_feature_inds()[0]
    columns_new = np.zeros(np.shape(feature_column_labels))
    for gg in range(len(np.unique(features_to_groups))):
        feature_inds = np.where(features_to_groups==gg)[0]
        columns_new[np.isin(feature_column_labels,feature_inds)] = gg;
   
    return columns_new, group_names

def get_feature_inds_pca_concat(subject, n_ori=4, n_sf=4, which_prf_grid=5):

    # get column labels that correspond to files with "PCA_concat" in name
    columns_raw, feature_type_names = get_feature_inds_simplegroups()
    n_ll_feats = np.sum(is_low_level())
    running_count = n_ll_feats;
    
    n_feat_each = np.zeros((len(feature_type_names),),dtype=int)
    
    for fi, feature_type_name in enumerate(feature_type_names):

        if fi<4:
            # when pca wasn't done, the size is same as raw feature size
            n_feat_each[fi] = np.sum(columns_raw==fi)
        else:
            # when pca was done, it is smaller.
            # to know the size, need to load the file that contains original dims
            fn2load = os.path.join(default_paths.pyramid_texture_feat_path,'PCA', \
                                   'S%d_%dori_%dsf_PCA_%s_only_grid%d.h5py'\
                                   %(subject, n_ori, n_sf, feature_type_name, which_prf_grid))
            with h5py.File(fn2load, 'r') as file:
                n_feat_each[fi] = file['/features'].shape[1]
            running_count += n_feat_each[fi]
            
    
    feature_column_labels = np.squeeze(np.concatenate([fi*np.ones([1,n_feat_each[fi]]) \
                                for fi in range(len(n_feat_each))], axis=1).astype('int'))
    assert(len(feature_column_labels)==running_count)
    
    return feature_column_labels, feature_type_names
