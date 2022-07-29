import sys, os
import numpy as np
import time, h5py

from utils import default_paths, numpy_utils, nsd_utils
from model_fitting import initialize_fitting 
from sklearn import decomposition
import argparse
import pandas as pd

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

"""
Code to perform PCA on features within a given feature space (texture or contour etc).
PCA is done separately within each pRF position, and the results for all pRFs are saved in a single file.
"""

def run_pca_each_prf(raw_filename, pca_filename, \
            fit_inds=None, prf_batch_size=50, \
              zscore_before_pca = False, \
              zgroup_labels = None,\
              min_pct_var=95, max_pc_to_retain=100,\
              save_dtype=np.float32, compress=True, \
              debug=False):

    if isinstance(raw_filename, list):  
        n_prf_batches = len(raw_filename) 
        batches_in_separate_files=True        
    else:
        raw_filename = [raw_filename]
        
        batches_in_separate_files=False
        
    
    n_prfs=0;
    for fi, fn in enumerate(raw_filename):
        if not os.path.exists(fn):
            raise RuntimeError('Looking at %s for precomputed features, not found.'%fn)   
        with h5py.File(fn, 'r') as data_set:
            dsize = data_set['/features'].shape
            data_set.close() 

        n_trials, n_feat_orig, n_prfs_tmp = dsize
        if fi==0 and batches_in_separate_files:
            assert(n_prfs_tmp==prf_batch_size)
        n_prfs += n_prfs_tmp
    
    if not batches_in_separate_files:
        n_prf_batches = int(np.ceil(n_prfs/prf_batch_size))  
    
    if max_pc_to_retain is None:
        max_pc_to_retain = n_feat_orig
        
    if fit_inds is not None:
        assert(len(fit_inds)==n_trials)
    else:
        fit_inds = np.ones((n_trials,),dtype=bool)
        
    # batching prfs for loading    
    prf_batch_inds = [np.arange(prf_batch_size*bb, np.min([prf_batch_size*(bb+1), n_prfs])) \
                          for bb in range(n_prf_batches)]
    prf_inds_loaded = []
    
    if zscore_before_pca and zgroup_labels is None:
        zgroup_labels = np.ones(shape=(1,n_feat_orig))
        
    scores_each_prf = np.zeros((n_trials, max_pc_to_retain, n_prfs), dtype=save_dtype)
    actual_max_ncomp=0
       
    for prf_model_index in range(n_prfs):

        if debug and prf_model_index>1:
            continue

        print('Processing pRF %d of %d'%(prf_model_index, n_prfs))

        if prf_model_index not in prf_inds_loaded:

            batch_to_use = np.where([prf_model_index in prf_batch_inds[bb] for \
                                     bb in range(len(prf_batch_inds))])[0][0]
            features_each_prf_batch = None
            prf_inds_loaded = prf_batch_inds[batch_to_use]
            
            t = time.time()
            
            if batches_in_separate_files:
                
                # each batch of prfs is in a separate file here
                filename_load = raw_filename[batch_to_use]
                print('Loading pre-computed features for prf models [%d - %d] from %s'%\
                  (prf_batch_inds[batch_to_use][0],prf_batch_inds[batch_to_use][-1], filename_load))
                with h5py.File(filename_load, 'r') as data_set:
                    values = np.copy(data_set['/features'][:,:,:])
                    data_set.close() 
                
            else:
                
                # all prfs are in same file, can just load part of it
                filename_load = raw_filename[0]
                print('Loading pre-computed features for prf models [%d - %d] from %s'%\
                    (prf_batch_inds[batch_to_use][0],prf_batch_inds[batch_to_use][-1], filename_load))
                with h5py.File(filename_load, 'r') as data_set:
                    values = np.copy(data_set['/features'][:,:,prf_batch_inds[batch_to_use]])
                    data_set.close() 
                
            features_each_prf_batch = values

        index_into_batch = np.where(prf_model_index==prf_inds_loaded)[0][0]
        print('Index into batch for prf %d: %d'%(prf_model_index, index_into_batch))
        features_in_prf = features_each_prf_batch[:,:,index_into_batch]

        elapsed = time.time() - t
        print('Took %.5f seconds to load file'%elapsed)
        
        if zscore_before_pca:
            features_in_prf_z = np.zeros_like(features_in_prf)
            features_in_prf_z[fit_inds,:] = numpy_utils.zscore_in_groups(features_in_prf[fit_inds,:], \
                                                                         zgroup_labels)
            zero_var = np.var(features_in_prf[fit_inds,:], axis=0)==0
            if np.sum(~fit_inds)>0:
                features_in_prf_z[~fit_inds,:] = numpy_utils.zscore_in_groups(features_in_prf[~fit_inds,:], \
                                                                              zgroup_labels)
                # if any feature channels had no variance, fix them now
                zero_var = zero_var | (np.var(features_in_prf[~fit_inds,:], axis=0)==0)
            print('there are %d columns with zero variance'%np.sum(zero_var))
            features_in_prf_z[:,zero_var] = features_in_prf_z[0,zero_var]
       
            features_in_prf = features_in_prf_z
        
        print('Size of features array for this image set and prf is:')
        print(features_in_prf.shape)
        print('any nans in array: %s'%np.any(np.isnan(features_in_prf)))
        
        # finding pca solution for just training data (specified in fit_inds)
        _, wts, pre_mean, ev = do_pca(features_in_prf[fit_inds,:], max_pc_to_retain=max_pc_to_retain)

        # now projecting all the data incl. val into same subspace
        feat_submean = features_in_prf - np.tile(pre_mean[np.newaxis,:], [features_in_prf.shape[0],1])
        scores = feat_submean @ wts.T
        
        n_comp_needed = np.where(np.cumsum(ev)>min_pct_var)
        if np.size(n_comp_needed)>0:
            n_comp_needed = n_comp_needed[0][0]+1
        else:
            n_comp_needed = scores.shape[1]
        print('Retaining %d components to explain %d pct var'%(n_comp_needed, min_pct_var))
        actual_max_ncomp = np.max([n_comp_needed, actual_max_ncomp])
        
        scores_each_prf[:,0:n_comp_needed,prf_model_index] = scores[:,0:n_comp_needed]
        scores_each_prf[:,n_comp_needed:,prf_model_index] = np.nan

    # To save space, get rid of portion of array that ended up all nans
    if debug:
        actual_max_ncomp=np.max([2,actual_max_ncomp])
        assert(np.all((scores_each_prf[:,actual_max_ncomp:,:]==0) | np.isnan(scores_each_prf[:,actual_max_ncomp:,:])))
    else:
        assert(np.all(np.isnan(scores_each_prf[:,actual_max_ncomp:,:])))
    scores_each_prf = scores_each_prf[:,0:actual_max_ncomp,:]
    
    print('final size of array to save:')
    print(scores_each_prf.shape)    
    print('saving to %s'%pca_filename)
    
    t = time.time()
    
    with h5py.File(pca_filename, 'w') as data_set:
        if compress==True:
            dset = data_set.create_dataset("features", np.shape(scores_each_prf), dtype=save_dtype, compression='gzip')
        else:
            dset = data_set.create_dataset("features", np.shape(scores_each_prf), dtype=save_dtype)
        data_set['/features'][:,:,:] = scores_each_prf
        data_set.close() 
    elapsed = time.time() - t

    print('Took %.5f sec to write file'%elapsed)
      
  
        
def run_pca(subject=None, \
            image_set=None, \
            feature_type=None, \
            layer_name = None, \
            which_prf_grid=5, min_pct_var=95,\
            max_pc_to_retain=None, debug=False):
       
    if subject is not None:
        image_set = 'S%d'%subject
        if subject==999:
            # 999 is a code for the set of images that are independent of NSD images, 
            # not shown to any participant.
            fit_inds = np.ones((10000,),dtype=bool)
        else:            
            # training / validation data always split the same way - shared 1000 inds are validation.
            subject_df = nsd_utils.get_subj_df(subject)
            fit_inds = np.array(subject_df['shared1000']==False)
    else:
        assert(image_set is not None)
        subject=0
        if image_set=='floc':
            labels_file = os.path.join(default_paths.floc_image_root,'floc_image_labels.csv')
            labels = pd.read_csv(labels_file)
            fit_inds = np.ones((labels.shape[0],),dtype=bool)
        else:
            raise ValueError('image_set %s not recognized'%image_set)
   
    if feature_type=='gabor':
        
        path_to_load = default_paths.gabor_texture_feat_path
        if debug:
            path_to_save = os.path.join(default_paths.gabor_texture_feat_path, 'DEBUG','PCA')
        else:
            path_to_save = os.path.join(default_paths.gabor_texture_feat_path, 'PCA')
        
        n_ori=12; n_sf = 8;
        raw_filename = os.path.join(path_to_load, \
                     '%s_features_each_prf_%dori_%dsf_gabor_solo_nonlin_grid%d.h5py'%\
                     (image_set, n_ori, n_sf, which_prf_grid))
        pca_filename = os.path.join(path_to_save, '%s_%dori_%dsf_nonlin_PCA_grid%d.h5py'%\
                     (image_set, n_ori, n_sf, which_prf_grid))

        zscore_before_pca = False; zgroup_labels = None;
        prf_batch_size=1500
        
    elif feature_type=='sketch_tokens':
        
        path_to_load = default_paths.sketch_token_feat_path
        if debug:
            path_to_save = os.path.join(default_paths.sketch_token_feat_path, 'DEBUG','PCA')
        else:
            path_to_save = os.path.join(default_paths.sketch_token_feat_path, 'PCA')
        
        raw_filename = os.path.join(path_to_load, '%s_features_each_prf_grid%d.h5py'%(image_set, which_prf_grid))
        pca_filename = os.path.join(path_to_save, '%s_PCA_grid%d.h5py'%(image_set, which_prf_grid))
        
        zscore_before_pca = True;
        zgroup_labels = np.concatenate([np.zeros(shape=(1,150)), np.ones(shape=(1,1))], axis=1)
        prf_batch_size=1500
        
    elif feature_type=='alexnet':
        
        path_to_load = default_paths.alexnet_feat_path
        if debug:
            path_to_save = os.path.join(default_paths.alexnet_feat_path, 'DEBUG','PCA')
        else:
            path_to_save = os.path.join(default_paths.alexnet_feat_path, 'PCA')
        
                   
        raw_filename = os.path.join(path_to_load, '%s_%s_ReLU_reflect_features_each_prf_grid%d.h5py'%\
                                 (image_set, layer_name, which_prf_grid))
        pca_filename = os.path.join(path_to_save, '%s_%s_ReLU_reflect_PCA_grid%d.h5py'%\
                                    (image_set, layer_name, which_prf_grid))
         
        zscore_before_pca = True;
        zgroup_labels = None
        prf_batch_size=100
        
    elif feature_type=='clip':

        path_to_load = default_paths.clip_feat_path
        if debug:
            path_to_save = os.path.join(default_paths.clip_feat_path, 'DEBUG','PCA')
        else:
            path_to_save = os.path.join(default_paths.clip_feat_path, 'PCA')
        
        model_architecture = 'RN50'
        n_prf_batches = 15
        raw_filename = [os.path.join(path_to_load, '%s_%s_%s_features_each_prf_grid%d_prfbatch%d.h5py'%\
                                 (image_set, model_architecture,layer_name, which_prf_grid, bb)) \
                                    for bb in range(n_prf_batches)]
        pca_filename = os.path.join(path_to_save, '%s_%s_%s_PCA_grid%d.h5py'%\
                                    (image_set, model_architecture, layer_name, which_prf_grid))
        zscore_before_pca = True;
        zgroup_labels = None
        prf_batch_size=100
        
    else:
        raise ValueError('feature type %s not recognized'%feature_type)
   
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
        
    run_pca_each_prf(raw_filename, pca_filename, \
            fit_inds = fit_inds, \
            prf_batch_size=prf_batch_size, 
            zscore_before_pca = zscore_before_pca, \
            zgroup_labels = zgroup_labels,\
            min_pct_var=min_pct_var,\
            max_pc_to_retain=max_pc_to_retain,\
            save_dtype=np.float32, compress=True, \
            debug=debug)
    

def do_pca(values, max_pc_to_retain=None):
    """
    Apply PCA to the data, return reduced dim data as well as weights, var explained.
    """
    n_features_actual = values.shape[1]
    n_trials = values.shape[0]
    
    if max_pc_to_retain is not None:        
        n_comp = np.min([np.min([max_pc_to_retain, n_features_actual]), n_trials])
    else:
        n_comp = np.min([n_features_actual, n_trials])
         
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
  
    return scores, wts, pre_mean, ev



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=0,
                    help="number of the subject, 1-8")
    parser.add_argument("--image_set", type=str,default='none',
                    help="name of the image set to use (if not an NSD subject)")
    parser.add_argument("--type", type=str,default='sketch_tokens',
                    help="what kind of features are we using?")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")

    parser.add_argument("--max_pc_to_retain", type=int,default=0,
                    help="max pc to retain? enter 0 for None")
    parser.add_argument("--min_pct_var", type=int,default=95,
                    help="min pct var to explain? default 95")
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which pRF grid to use?")
    
    
    args = parser.parse_args()
    
    if args.subject==0:
        args.subject=None
    if args.image_set=='none':
        args.image_set=None
    
    if args.max_pc_to_retain==0:
        args.max_pc_to_retain = None
    
    if args.type=='alexnet':
        layers = ['Conv%d'%(ll+1) for ll in range(5)]
        for layer in layers:
            run_pca(subject=args.subject, 
                    image_set = args.image_set,\
                            feature_type=args.type, \
                            layer_name = layer,
                            which_prf_grid=args.which_prf_grid,
                            min_pct_var=args.min_pct_var, 
                            max_pc_to_retain=args.max_pc_to_retain, 
                            debug=args.debug==1)
    elif args.type=='clip':
        layers = ['block%d'%(ll) for ll in range(16)]
        for layer in layers:
            run_pca(subject=args.subject, 
                    image_set = args.image_set,\
                            feature_type=args.type, \
                            layer_name = layer,
                            which_prf_grid=args.which_prf_grid,
                            min_pct_var=args.min_pct_var, 
                            max_pc_to_retain=args.max_pc_to_retain, 
                            debug=args.debug==1)
    else:
        run_pca(subject=args.subject, 
                image_set = args.image_set,\
                            feature_type=args.type, \
                            which_prf_grid=args.which_prf_grid,
                            min_pct_var=args.min_pct_var, 
                            max_pc_to_retain=args.max_pc_to_retain, 
                            debug=args.debug==1)
    