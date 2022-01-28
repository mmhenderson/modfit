import sys, os
import numpy as np
import time, h5py
import scipy.stats
codepath = '/user_data/mmhender/imStat/code'
sys.path.append(codepath)
from utils import default_paths, nsd_utils, coco_utils, stats_utils
from model_fitting import initialize_fitting 
import argparse
import pandas as pd
   
def get_feature_discrim(subject, feature_type, discrim_type='animacy', which_prf_grid=1, debug=False):

    print('\nusing prf grid %d\n'%(which_prf_grid))
    # Params for the spatial aspect of the model (possible pRFs)
    models = initialize_fitting.get_prf_models(which_grid = which_prf_grid)    

    # training / validation data always split the same way - shared 1000 inds are validation.
    subject_df = nsd_utils.get_subj_df(subject)
    valinds = np.array(subject_df['shared1000'])
    trninds = np.array(subject_df['shared1000']==False)
    # working only with training data here.
    labels_all, discrim_type_list, unique_labels_each = coco_utils.load_labels_each_prf(subject, which_prf_grid, \
                                                     image_inds=np.where(trninds)[0], models=models,verbose=False)
    n_sem_axes = labels_all.shape[1]
        
    if feature_type=='sketch_tokens':

        path_to_load = default_paths.sketch_token_feat_path
        features_file = os.path.join(path_to_load, 'S%d_features_each_prf_grid%d.h5py'%(subject, \
                                                                                        which_prf_grid)) 
    elif feature_type=='pyramid_texture':
        
        path_to_load = default_paths.pyramid_texture_feat_path      
        n_ori = 4; n_sf = 4;
        features_file = os.path.join(path_to_load, 'S%d_features_each_prf_%dori_%dsf_grid%d.h5py'%\
                                     (subject,n_ori, n_sf, which_prf_grid))    
    elif feature_type=='gabor_solo':
        
        path_to_load = default_paths.gabor_texture_feat_path   
        n_ori = 12; n_sf = 8;
        features_file = os.path.join(path_to_load, \
                                 'S%d_features_each_prf_%dori_%dsf_gabor_solo_nonlin_grid%d.h5py'%\
                                 (subject,n_ori, n_sf, which_prf_grid))
    else:
        raise RuntimeError('feature type %s not recognized'%feature_type)
      
    if not os.path.exists(features_file):
        raise RuntimeError('Looking at %s for precomputed features, not found.'%features_file)   
  
                               
    prf_batch_size = 50 # batching prfs for loading, because it is a bit faster
    n_prfs = models.shape[0]
    n_prf_batches = int(np.ceil(n_prfs/prf_batch_size))          
    prf_batch_inds = [np.arange(prf_batch_size*bb, np.min([prf_batch_size*(bb+1), n_prfs])) for bb in range(n_prf_batches)]
    prf_inds_loaded = []
    
    path_to_save = os.path.join(path_to_load, 'feature_stats')
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    fn2save_corrs = os.path.join(path_to_save, 'S%d_semantic_corrs_grid%d.npy'%(subject, which_prf_grid))
    fn2save_discrim = os.path.join(path_to_save, 'S%d_semantic_discrim_fstat_grid%d.npy'%(subject, which_prf_grid))
    fn2save_mean = os.path.join(path_to_save, 'S%d_mean_grid%d.npy'%(subject, which_prf_grid))
    fn2save_var = os.path.join(path_to_save, 'S%d_var_grid%d.npy'%(subject, which_prf_grid))
    fn2save_covar = os.path.join(path_to_save, 'S%d_covar_grid%d.npy'%(subject, which_prf_grid)) 
    
    with h5py.File(features_file, 'r') as data_set:
        dims = data_set['/features'].shape
    n_trials, n_features, n_prfs = dims
    all_corrs = np.zeros((n_features, n_prfs, n_sem_axes), dtype=np.float32)
    all_discrim =  np.zeros((n_features, n_prfs, n_sem_axes), dtype=np.float32)
    all_mean = np.zeros((n_features, n_prfs), dtype=np.float32)
    all_var =  np.zeros((n_features, n_prfs), dtype=np.float32)
    all_covar =  np.zeros((n_features, n_features, n_prfs), dtype=np.float32)
    
    for prf_model_index in range(n_prfs):

        if debug and prf_model_index>1:
            continue

        print('Processing pRF %d of %d'%(prf_model_index, n_prfs))
        if prf_model_index not in prf_inds_loaded:

            batch_to_use = np.where([prf_model_index in prf_batch_inds[bb] for \
                                     bb in range(len(prf_batch_inds))])[0][0]
            assert(prf_model_index in prf_batch_inds[batch_to_use])

            print('Loading pre-computed features for prf models [%d - %d] from %s'%\
                  (prf_batch_inds[batch_to_use][0],prf_batch_inds[batch_to_use][-1], features_file))
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
        features_in_prf_trn = features_in_prf[trninds,:]
        print('Size of features array for this image set and prf is:')
        print(features_in_prf_trn.shape)
        
        # computing some basic stats for the features in this pRF
        all_mean[:,prf_model_index] = np.mean(features_in_prf_trn, axis=0);
        all_var[:,prf_model_index] = np.var(features_in_prf_trn, axis=0);
        all_covar[:,:,prf_model_index] = np.cov(features_in_prf_trn.T)
        
        sys.stdout.flush()
                                 
        for aa in range(n_sem_axes):

            labels = labels_all[:,aa,prf_model_index]
            inds2use = ~np.isnan(labels)          
            unique_labels_actual = np.unique(labels[inds2use])
            
            if prf_model_index==0:
                print('processing axis: %s'%discrim_type_list[aa])
                print('labels: ')
                print(unique_labels_each[aa])
               
            if np.any(~np.isin(unique_labels_each[aa], unique_labels_actual)):
                print('missing some labels for axis %d'%aa)
                print('expected labels')
                print(unique_labels_each[aa])
                print('actual labels')
                print(unique_labels_actual)
                
            if len(unique_labels_actual)>1:
                
                group_inds = [((labels==ll) & inds2use) for ll in unique_labels_actual]
                
                for ff in range(n_features):
                    
                    groups = [features_in_prf_trn[gi,ff] for gi in group_inds]
                    fstat = stats_utils.anova_oneway_warn(groups).statistic
                    all_discrim[ff, prf_model_index, aa] = fstat
                
            else:

                print('nans for model %d, axis %d - need >1 levels'\
                          %(prf_model_index, aa))
                all_discrim[:, prf_model_index, aa] = np.nan
             
            # for the binary categories, also getting a correlation coefficient (sign matters here)
            if (len(unique_labels_each[aa])==2) and (len(unique_labels_actual)==2):
                for ff in range(n_features):
                    all_corrs[ff,prf_model_index,aa] = stats_utils.numpy_corrcoef_warn(\
                                                        features_in_prf_trn[inds2use,ff],labels[inds2use])[0,1]
            else:
                all_corrs[:,prf_model_index,aa] = np.nan
                
    print('saving to %s\n'%fn2save_corrs)
    np.save(fn2save_corrs, all_corrs)                     
    print('saving to %s\n'%fn2save_discrim)
    np.save(fn2save_discrim, all_discrim)     
    print('saving to %s\n'%fn2save_mean)
    np.save(fn2save_mean, all_mean)                     
    print('saving to %s\n'%fn2save_var)
    np.save(fn2save_var, all_var)    
    print('saving to %s\n'%fn2save_covar)
    np.save(fn2save_covar, all_covar)    

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=1,
                    help="number of the subject, 1-8")
    parser.add_argument("--feature_type", type=str,default='sketch_tokens',
                    help="what kind of features are we using?")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which prf grid to use")
   
    args = parser.parse_args()

    if args.debug:
        print('DEBUG MODE\n')

    get_feature_discrim(subject=args.subject, feature_type=args.feature_type, debug=args.debug==1, which_prf_grid=args.which_prf_grid)
   