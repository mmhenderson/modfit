import sys, os
import numpy as np
import time, h5py
codepath = '/user_data/mmhender/imStat/code'
sys.path.append(codepath)
from utils import default_paths, numpy_utils, nsd_utils
from model_fitting import initialize_fitting 
from feature_extraction import texture_statistics_pyramid
from sklearn import decomposition
import argparse

"""
Code to perform PCA on features within a given feature space (texture or contour etc).
PCA is done separately within each pRF position, and the results for all pRFs are saved in a single file.
"""

def run_pca_texture_pyramid(subject, n_ori=4, n_sf=4, max_pc_to_retain=None, debug=False, zscore_first=False, which_prf_grid=1):

    path_to_load = default_paths.pyramid_texture_feat_path

    print('\nusing prf grid %d\n'%(which_prf_grid))

    features_file = os.path.join(path_to_load, 'S%d_features_each_prf_%dori_%dsf_grid%d.h5py'%(subject,\
                                                                               n_ori, n_sf, which_prf_grid))
        
    if not os.path.exists(features_file):
        raise RuntimeError('Looking at %s for precomputed features, not found.'%features_file)   
    path_to_save = os.path.join(path_to_load, 'PCA')
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    # Params for the spatial aspect of the model (possible pRFs)
    aperture_rf_range = 1.1
    aperture, models = initialize_fitting.get_prf_models(aperture_rf_range=aperture_rf_range, \
                                                         which_grid = which_prf_grid)    

    prf_batch_size = 50 # batching prfs for loading, because it is a bit faster
    n_prfs = models.shape[0]
    n_prf_batches = int(np.ceil(n_prfs/prf_batch_size))          
    prf_batch_inds = [np.arange(prf_batch_size*bb, np.min([prf_batch_size*(bb+1), n_prfs])) for bb in range(n_prf_batches)]

    # Set up the pyramid feature extractor (just to get dims of diff feature types, not using it for real here)
    _fmaps_fn = texture_statistics_pyramid.steerable_pyramid_extractor(pyr_height = n_sf, n_ori = n_ori)
    _feature_extractor = texture_statistics_pyramid.texture_feature_extractor(_fmaps_fn, subject=subject, \
                                      sample_batch_size=None, feature_types_exclude=[], n_prf_sd_out=2, \
                                      aperture=aperture, do_varpart = False, compute_features=False, \
                                      group_all_hl_feats = True, device='cpu:0', which_prf_grid=which_prf_grid)
    # Get dims of each feature type
    dims = np.array(_feature_extractor.feature_type_dims_all)
    is_ll = _feature_extractor.feature_is_ll
    # going to treat lower-level and higher-level separately here, doing pca within each set.
    feature_type_dims_ll = dims[is_ll]
    feature_type_dims_hl = dims[~is_ll]
    n_ll_feats = np.sum(feature_type_dims_ll)
    n_hl_feats = np.sum(feature_type_dims_hl)
    
    # Define groups of columns to zscore within.
    # Treating every pixel statistic as a different group because of different scales.
    # Keeping the groups of mean magnitude features together across orients and scales - rather
    # than z-scoring each column, to preserve informative difference across these channels.
    zgroup_sizes_ll = [1,1,1,1,1,1] + list(feature_type_dims_ll[1:])
    zgroup_sizes_hl = list(feature_type_dims_hl)
    zgroup_labels_ll = np.concatenate([np.ones(shape=(1, zgroup_sizes_ll[ff]))*ff \
                                           for ff in range(len(zgroup_sizes_ll))], axis=1)
    # For the marginal stats of lowpass recons, separating skew/kurtosis here
    zgroup_labels_ll[zgroup_labels_ll==9] = 10
    zgroup_labels_ll[0,np.where(zgroup_labels_ll==8)[1][np.arange(1,10,2)]] = 9

    # for higher level groups, just retaining original grouping scheme 
    zgroup_labels_hl = np.concatenate([np.ones(shape=(1, zgroup_sizes_hl[ff]))*ff \
                                           for ff in range(len(zgroup_sizes_hl))], axis=1)

    # training / validation data always split the same way - shared 1000 inds are validation.
    subject_df = nsd_utils.get_subj_df(subject)
    valinds = np.array(subject_df['shared1000'])
    trninds = np.array(subject_df['shared1000']==False)
    
    prf_inds_loaded = []
    scores_ll_each_prf = []
    wts_ll_each_prf = []
    ev_ll_each_prf = []
    pre_mean_ll_each_prf = []
    scores_hl_each_prf = []
    wts_hl_each_prf = []
    ev_hl_each_prf = []
    pre_mean_hl_each_prf = []
    
    for prf_model_index in range(n_prfs):

        if debug and prf_model_index>1:
            continue

        print('Processing pRF %d of %d'%(prf_model_index, n_prfs))
        if prf_model_index not in prf_inds_loaded:

            batch_to_use = np.where([prf_model_index in prf_batch_inds[bb] for \
                                     bb in range(len(prf_batch_inds))])[0][0]
            assert(prf_model_index in prf_batch_inds[batch_to_use])

            print('Loading pre-computed features for prf models [%d - %d] from %s'%(prf_batch_inds[batch_to_use][0], \
                                                                              prf_batch_inds[batch_to_use][-1], features_file))
            features_each_prf_batch = None

            t = time.time()
            with h5py.File(features_file, 'r') as data_set:
                values = np.copy(data_set['/features'][:,:,prf_batch_inds[batch_to_use]])
                data_set.close() 
            elapsed = time.time() - t
            print('Took %.5f seconds to load file'%elapsed)

            prf_inds_loaded = prf_batch_inds[batch_to_use]
            features_each_prf_batch = values.astype(np.float32)
            values=None

        index_into_batch = np.where(prf_model_index==prf_inds_loaded)[0][0]
        print('Index into batch for prf %d: %d'%(prf_model_index, index_into_batch))
        features_in_prf = features_each_prf_batch[:,:,index_into_batch]
        values=None
        print('Size of features array for this image set and prf is:')
        print(features_in_prf.shape)
        
        features_ll = features_in_prf[:,0:n_ll_feats]
        features_hl = features_in_prf[:,n_ll_feats:]
        assert(n_hl_feats==features_hl.shape[1])
        
        # z-score each group of columns.
        features_ll_z = np.zeros_like(features_ll)
        features_ll_z[trninds,:] = numpy_utils.zscore_in_groups(features_ll[trninds,:], zgroup_labels_ll)
        features_ll_z[~trninds,:] = numpy_utils.zscore_in_groups(features_ll[~trninds,:], zgroup_labels_ll)
        features_hl_z = np.zeros_like(features_hl)
        features_hl_z[trninds,:] = numpy_utils.zscore_in_groups(features_hl[trninds,:], zgroup_labels_hl)
        features_hl_z[~trninds,:] = numpy_utils.zscore_in_groups(features_hl[~trninds,:], zgroup_labels_hl)
        
        _, wts_ll, pre_mean_ll, ev_ll = do_pca(features_ll_z[trninds,:], max_pc_to_retain=max_pc_to_retain,\
                                                      zscore_first=False)
        feat_submean_ll = features_ll_z - np.tile(pre_mean_ll[np.newaxis,:], [features_ll_z.shape[0],1])
        scores_ll = feat_submean_ll @ wts_ll.T
        scores_ll_each_prf.append(scores_ll)
        wts_ll_each_prf.append(wts_ll)
        ev_ll_each_prf.append(ev_ll)
        pre_mean_ll_each_prf.append(pre_mean_ll)
        
        _, wts_hl, pre_mean_hl, ev_hl = do_pca(features_hl_z[trninds,:], max_pc_to_retain=max_pc_to_retain,\
                                                      zscore_first=False)
        feat_submean_hl = features_hl_z - np.tile(pre_mean_hl[np.newaxis,:], [features_hl_z.shape[0],1])
        scores_hl = feat_submean_hl @ wts_hl.T
        scores_hl_each_prf.append(scores_hl)
        wts_hl_each_prf.append(wts_hl)
        ev_hl_each_prf.append(ev_hl)
        pre_mean_hl_each_prf.append(pre_mean_hl)
        
        print('size of wts_hl:')
        print(wts_hl.shape)    
        
        print('size of scores_hl:')
        print(scores_hl.shape) 
        print(scores_hl.dtype)
        print('scores_hl_each_prf list is approx %.2f GiB'%numpy_utils.get_list_size_gib(scores_hl_each_prf))
        
        sys.stdout.flush()

    fn2save = os.path.join(path_to_save, 'S%d_%dori_%dsf_PCA_lower-level_only_grid%d.npy'%(subject, n_ori, n_sf, which_prf_grid))

    print('saving to %s'%fn2save)
    np.save(fn2save, {'scores': scores_ll_each_prf,'wts': wts_ll_each_prf, \
                      'ev': ev_ll_each_prf, 'pre_mean': pre_mean_ll_each_prf})

    fn2save = os.path.join(path_to_save, 'S%d_%dori_%dsf_PCA_higher-level_only_grid%d.npy'%(subject, n_ori, n_sf, which_prf_grid))

    print('saving to %s'%fn2save)
    np.save(fn2save, {'scores': scores_hl_each_prf, 'wts': wts_hl_each_prf, \
                      'ev': ev_hl_each_prf,  'pre_mean': pre_mean_hl_each_prf})

    
def run_pca_sketch_tokens(subject, max_pc_to_retain=None, debug=False, zscore_first=False, which_prf_grid=1):

    path_to_load = default_paths.sketch_token_feat_path

    features_file = os.path.join(path_to_load, 'S%d_features_each_prf_grid%d.h5py'%(subject, which_prf_grid))
    if not os.path.exists(features_file):
        raise RuntimeError('Looking at %s for precomputed features, not found.'%features_file)   
    path_to_save = os.path.join(path_to_load, 'PCA')
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    # Params for the spatial aspect of the model (possible pRFs)
    aperture_rf_range = 1.1
    aperture, models = initialize_fitting.get_prf_models(aperture_rf_range=aperture_rf_range)    
    n_prfs = models.shape[0]
    
    print('Loading pre-computed features from %s'%features_file)
    t = time.time()
    with h5py.File(features_file, 'r') as data_set:
        values = np.copy(data_set['/features'])
        data_set.close() 
    elapsed = time.time() - t
    print('Took %.5f seconds to load file'%elapsed)
    features_each_prf = values

    zgroup_labels = np.concatenate([np.zeros(shape=(1,150)), np.ones(shape=(1,1))], axis=1)

    # training / validation data always split the same way - shared 1000 inds are validation.
    subject_df = nsd_utils.get_subj_df(subject)
    valinds = np.array(subject_df['shared1000'])
    trninds = np.array(subject_df['shared1000']==False)
    
    
    print('Size of features array for this image set is:')
    print(features_each_prf.shape)

    scores_each_prf = []
    wts_each_prf = []
    ev_each_prf = []
    pre_mean_each_prf = []
   
    for prf_model_index in range(n_prfs):

        if debug and prf_model_index>1:
            continue

        print('Processing pRF %d of %d'%(prf_model_index, n_prfs))
        
        features_in_prf = features_each_prf[:,:,prf_model_index]
        
        features_in_prf_z = np.zeros_like(features_in_prf)
        features_in_prf_z[trninds,:] = numpy_utils.zscore_in_groups(features_in_prf[trninds,:], zgroup_labels)
        features_in_prf_z[~trninds,:] = numpy_utils.zscore_in_groups(features_in_prf[~trninds,:], zgroup_labels)
           
        print('Size of features array for this image set and prf is:')
        print(features_in_prf_z.shape)

        # finding pca solution for just training data
        _, wts, pre_mean, ev = do_pca(features_in_prf_z[trninds,:], max_pc_to_retain=max_pc_to_retain,\
                                                      zscore_first=False)

        # now projecting all the data incl. val into same subspace
        feat_submean = features_in_prf_z - np.tile(pre_mean[np.newaxis,:], [features_in_prf_z.shape[0],1])
        scores = feat_submean @ wts.T
        
        scores_each_prf.append(scores)
        wts_each_prf.append(wts)
        ev_each_prf.append(ev)
        pre_mean_each_prf.append(pre_mean)

    fn2save = os.path.join(path_to_save, 'S%d_PCA_grid%d.npy'%(subject, which_prf_grid))
    print('saving to %s'%fn2save)
    np.save(fn2save, {'scores': scores_each_prf,'wts': wts_each_prf, \
                      'ev': ev_each_prf, 'pre_mean': pre_mean_each_prf})

    
# def run_pca_alexnet(subject, layer_name, max_pc_to_retain=None, debug=False, zscore_first=False, which_prf_grid=1):

#     path_to_load = default_paths.alexnet_feat_path

#     if which_prf_grid==1:
#         features_file = os.path.join(path_to_load, 'S%d_%s_ReLU_reflect_features_each_prf.h5py'%(subject, layer_name))
#     else:
#         features_file = os.path.join(path_to_load, 'S%d_%s_ReLU_reflect_features_each_prf_grid%d.h5py'%(subject, layer_name, which_prf_grid))
#     if not os.path.exists(features_file):
#         raise RuntimeError('Looking at %s for precomputed features, not found.'%features_file)   
#     path_to_save = os.path.join(path_to_load, 'PCA')
#     if not os.path.exists(path_to_save):
#         os.mkdir(path_to_save)

#     # Params for the spatial aspect of the model (possible pRFs)
#     aperture_rf_range = 1.1
#     aperture, models = initialize_fitting.get_prf_models(aperture_rf_range=aperture_rf_range)    
#     n_prfs = models.shape[0]
    
#     print('Loading pre-computed features from %s'%features_file)
#     t = time.time()
#     with h5py.File(features_file, 'r') as data_set:
#         values = np.copy(data_set['/features'])
#         data_set.close() 
#     elapsed = time.time() - t
#     print('Took %.5f seconds to load file'%elapsed)
#     features_each_prf = values

#     zgroup_labels = np.concatenate([np.zeros(shape=(1,150)), np.ones(shape=(1,1))], axis=1)

#     # training / validation data always split the same way - shared 1000 inds are validation.
#     subject_df = nsd_utils.get_subj_df(subject)
#     valinds = np.array(subject_df['shared1000'])
#     trninds = np.array(subject_df['shared1000']==False)
    
    
#     print('Size of features array for this image set is:')
#     print(features_each_prf.shape)

#     scores_each_prf = []
#     wts_each_prf = []
#     ev_each_prf = []
#     pre_mean_each_prf = []
   
#     for prf_model_index in range(n_prfs):

#         if debug and prf_model_index>1:
#             continue

#         print('Processing pRF %d of %d'%(prf_model_index, n_prfs))
        
#         features_in_prf = features_each_prf[:,:,prf_model_index]
        
#         features_in_prf_z = np.zeros_like(features_in_prf)
#         features_in_prf_z[trninds,:] = numpy_utils.zscore_in_groups(features_in_prf[trninds,:], zgroup_labels)
#         features_in_prf_z[~trninds,:] = numpy_utils.zscore_in_groups(features_in_prf[~trninds,:], zgroup_labels)
           
#         print('Size of features array for this image set and prf is:')
#         print(features_in_prf_z.shape)

#         # finding pca solution for just training data
#         _, wts, pre_mean, ev = do_pca(features_in_prf_z[trninds,:], max_pc_to_retain=max_pc_to_retain,\
#                                                       zscore_first=False)

#         # now projecting all the data incl. val into same subspace
#         feat_submean = features_in_prf_z - np.tile(pre_mean[np.newaxis,:], [features_in_prf_z.shape[0],1])
#         scores = feat_submean @ wts.T
        
#         scores_each_prf.append(scores)
#         wts_each_prf.append(wts)
#         ev_each_prf.append(ev)
#         pre_mean_each_prf.append(pre_mean)

#     if which_prf_grid==1:
#         fn2save = os.path.join(path_to_save, 'S%d_PCA.npy'%(subject))
#     else:
#         fn2save = os.path.join(path_to_save, 'S%d_PCA_grid%d.npy'%(subject, which_prf_grid))
#     print('saving to %s'%fn2save)
#     np.save(fn2save, {'scores': scores_each_prf,'wts': wts_each_prf, \
#                       'ev': ev_each_prf, 'pre_mean': pre_mean_each_prf})

    
def do_pca(values, max_pc_to_retain=None, zscore_first=False):
    """
    Apply PCA to the data, return reduced dim data as well as weights, var explained.
    """
    n_features_actual = values.shape[1]
    n_trials = values.shape[0]
    
    if max_pc_to_retain is not None:        
        n_comp = np.min([np.min([max_pc_to_retain, n_features_actual]), n_trials])
    else:
        n_comp = np.min([n_features_actual, n_trials])
        
    if zscore_first:
        # zscore each column (optional)
        vals_m = np.mean(values, axis=0, keepdims=True) 
        vals_s = np.std(values, axis=0, keepdims=True)         
        values -= vals_m
        values /= vals_s 
        
    print('Running PCA: original size of array is [%d x %d]'%(n_trials, n_features_actual))
    t = time.time()
    pca = decomposition.PCA(n_components = n_comp, copy=False)
    scores = pca.fit_transform(values)           
    elapsed = time.time() - t
    print('Time elapsed: %.5f'%elapsed)
    values = None            
    wts = pca.components_
    ev = pca.explained_variance_
    ev = ev/np.sum(ev)*100
    pre_mean = pca.mean_
    
    print('First element of ev: %.2f'%ev[0])
    # print this out as a check...if it is always 1, this can mean something is wrong w data.
    if np.size(np.where(np.cumsum(ev)>=95))>0:
        n_comp_needed = np.where(np.cumsum(ev)>=95)[0][0]
        print('Requires %d components to explain 95 pct var'%n_comp_needed)    
    else:
        n_comp_needed = n_comp
        print('Requires more than %d components to explain 95 pct var'%n_comp_needed)
    
    return scores, wts, pre_mean, ev


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=1,
                    help="number of the subject, 1-8")
    parser.add_argument("--type", type=str,default='sketch_tokens',
                    help="what kind of features are we using?")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    parser.add_argument("--zscore", type=int,default=0,
                    help="want to zscore individual columns before pca? 1 for yes, 0 for no")
    parser.add_argument("--max_pc_to_retain", type=int,default=0,
                    help="max pc to retain? enter 0 for None")
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which pRF grid to use?")
    
    args = parser.parse_args()
    
    print('\nusing prf grid %d\n'%(args.which_prf_grid))
    
    if args.max_pc_to_retain==0:
        args.max_pc_to_retain = None
    
    if args.type=='sketch_tokens':
        run_pca_sketch_tokens(subject=args.subject, max_pc_to_retain=args.max_pc_to_retain, debug=args.debug==1, zscore_first=args.zscore==1, which_prf_grid=args.which_prf_grid)
    elif args.type=='texture_pyramid':
        n_ori=4
        n_sf=4
        run_pca_texture_pyramid(subject=args.subject, n_ori=n_ori, n_sf=n_sf, max_pc_to_retain=args.max_pc_to_retain, debug=args.debug==1, zscore_first=args.zscore==1, which_prf_grid=args.which_prf_grid)
    else:
        raise ValueError('--type %s is not recognized'%args.type)