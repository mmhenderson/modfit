import sys, os
import numpy as np
import time, h5py
codepath = '/user_data/mmhender/imStat/code'
sys.path.append(codepath)
from utils import default_paths, nsd_utils
from model_fitting import initialize_fitting 
import argparse
import pandas as pd
   
def get_discrim(subject, feature_type, discrim_type='animacy', which_prf_grid=1, debug=False):

    print('\nusing prf grid %d\n'%(which_prf_grid))
    # Params for the spatial aspect of the model (possible pRFs)
    models = initialize_fitting.get_prf_models(which_grid = which_prf_grid)    

    labels_folder = os.path.join(default_paths.stim_labels_root, 'S%d_within_prf_grid%d'%(subject, \
                                                                                        which_prf_grid))
    
    # training / validation data always split the same way - shared 1000 inds are validation.
    subject_df = nsd_utils.get_subj_df(subject)
    valinds = np.array(subject_df['shared1000'])
    trninds = np.array(subject_df['shared1000']==False)
    
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
  
    path_to_save = os.path.join(path_to_load, 'semantic_discrim')
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    fn2save_corrs = os.path.join(path_to_save, 'S%d_corrs_%s_grid%d.npy'%(subject, discrim_type, which_prf_grid))
    fn2save_dprime = os.path.join(path_to_save, 'S%d_dprime_%s_grid%d.npy'%(subject, discrim_type, which_prf_grid))
    fn2save_mean = os.path.join(path_to_load, 'S%d_mean_grid%d.npy'%(subject, which_prf_grid))
    fn2save_var = os.path.join(path_to_load, 'S%d_var_grid%d.npy'%(subject, which_prf_grid))
    fn2save_covar = os.path.join(path_to_load, 'S%d_covar_grid%d.npy'%(subject, which_prf_grid))                            
    prf_batch_size = 50 # batching prfs for loading, because it is a bit faster
    n_prfs = models.shape[0]
    n_prf_batches = int(np.ceil(n_prfs/prf_batch_size))          
    prf_batch_inds = [np.arange(prf_batch_size*bb, np.min([prf_batch_size*(bb+1), n_prfs])) for bb in range(n_prf_batches)]
    prf_inds_loaded = []
    
    if discrim_type=='indoor_outdoor':
        # this property is defined across whole images, so loading labels outside the pRF loop.
        coco_labels_fn = os.path.join(default_paths.stim_labels_root, 'S%d_indoor_outdoor.csv'%subject)
        print('Reading labels from %s...'%coco_labels_fn)
        coco_df = pd.read_csv(coco_labels_fn, index_col=0)
        ims_to_use = np.sum(np.array(coco_df)==1, axis=1)==1
        labels = np.array(coco_df['has_indoor']).astype(np.float32)
        labels[~ims_to_use] = np.nan
        neach = [np.sum(labels==ll) for ll in np.unique(labels[~np.isnan(labels)])] + \
                [np.sum(np.isnan(labels))]
        print('n outdoor/n indoor/n ambiguous:')
        print(neach)
        unvals = np.unique(labels[ims_to_use])                               
        print('unique values:')
        print(unvals)
            
    with h5py.File(features_file, 'r') as data_set:
        dims = data_set['/features'].shape
    n_trials, n_features, n_prfs = dims
    all_corrs = np.zeros((n_features, n_prfs), dtype=np.float32)
    all_dprime =  np.zeros((n_features, n_prfs), dtype=np.float32)
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
                                 
        if discrim_type=='indoor_outdoor':
            labels = labels
            ims_to_use = ims_to_use
        elif discrim_type=='animacy' or discrim_type=='person' or discrim_type=='food' \
                                 or discrim_type=='vehicle' or discrim_type=='animal':
            coco_labels_fn = os.path.join(labels_folder, \
                                          'S%d_cocolabs_binary_prf%d.csv'%(subject, prf_model_index))
            print('Reading labels from %s...'%coco_labels_fn)
            coco_df = pd.read_csv(coco_labels_fn, index_col=0)
            print('Using %s as label'%discrim_type)
            if discrim_type=='animacy':
                 labels = np.array(coco_df['has_animate']).astype('int')
            else:
                labels = np.array(coco_df[discrim_type]).astype('int')
            unvals = np.unique(labels)
            ims_to_use = np.ones(np.shape(labels))==1
            print('Unique labels:')
            print(unvals)
            print('Proportion with %s:'%discrim_type)
            print(np.mean(labels==1))            
        else:
            raise ValueError('discrimination type %s not recognized.'%discrim_type)
            
        labels_trn = labels[trninds]
        inds2use = ~np.isnan(labels_trn) 
        inds1 = (labels_trn==0) & inds2use
        inds2 = (labels_trn==1) & inds2use

        # now computing relationship between each feature and the semantic labels
        if np.any(inds1) and np.any(inds2):
            
            for ff in range(n_features):
                if np.std(features_in_prf_trn[inds2use,ff])==0:
                    print('std==0')
                all_corrs[ff,prf_model_index] = np.corrcoef(features_in_prf_trn[inds2use,ff], \
                                                            labels_trn[inds2use])[0,1]
                # (mu1-mu2) / std
                all_dprime[ff,prf_model_index] = (np.mean(features_in_prf_trn[inds1,ff]) -\
                                      np.mean(features_in_prf_trn[inds2,ff]))/np.std(features_in_prf_trn[inds2use,ff])          
        else:
            print('model %d - at least one label category is missing'%(prf_model_index))
            all_corrs[ff,prf_model_index] = np.nan
            all_dprime[ff,prf_model_index] = np.nan
            
    print('saving to %s\n'%fn2save_corrs)
    np.save(fn2save_corrs, all_corrs)                     
    print('saving to %s\n'%fn2save_dprime)
    np.save(fn2save_dprime, all_dprime)     
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
    parser.add_argument("--discrim_type", type=str,default='animacy',
                    help="what semantic labels are we using to classify?")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which prf grid to use")
   
    args = parser.parse_args()

    if args.debug:
        print('DEBUG MODE\n')

    get_discrim(subject=args.subject, feature_type=args.feature_type, debug=args.debug==1, discrim_type=args.discrim_type, which_prf_grid=args.which_prf_grid)
   