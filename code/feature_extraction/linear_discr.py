import sys, os
import numpy as np
import time, h5py
codepath = '/user_data/mmhender/imStat/code'
sys.path.append(codepath)
from utils import default_paths, nsd_utils, numpy_utils, stats_utils
from model_fitting import initialize_fitting 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import argparse
import pandas as pd   
    
def find_lda_axes(subject, feature_type, discrim_type='animacy', which_prf_grid=1, debug=False, zscore_each=True, zscore_groups=False, balance_downsample=True):

    if zscore_each:
        zscore_groups=False
    if not zscore_each:
        assert(balance_downsample)
        
    print('zscore_each=%s, zscore_groups=%s, balance_downsample=%s'\
              %(zscore_each, zscore_groups, balance_downsample))
        
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
         # how to do z-scoring? can set up groups of columns here.
        zgroup_labels = np.concatenate([np.zeros(shape=(150,)), np.ones(shape=(1,))], axis=0)
        features_to_use = np.ones(np.shape(zgroup_labels))==1
        n_features = 151
        
    elif (feature_type=='pyramid_texture_ll') or (feature_type=='pyramid_texture_hl'):
        
        path_to_load = default_paths.pyramid_texture_feat_path      
        n_ori = 4; n_sf = 4;
        features_file = os.path.join(path_to_load, 'S%d_features_each_prf_%dori_%dsf_grid%d.h5py'%\
                                     (subject,n_ori, n_sf, which_prf_grid))  
        # Get dims of each feature type
        feature_type_dims_ll = [ 6, 16, 16, 10,  1];
        n_ll_feats = np.sum(feature_type_dims_ll)
        feature_type_dims_hl = [272,  73,  25,  24,  24,  48,  96,  10,  20];  
        n_hl_feats = np.sum(feature_type_dims_hl)
        if feature_type=='pyramid_texture_ll':   
            # Define groups of columns to zscore within.
            # Treating every pixel statistic as a different group because of different scales.
            # Keeping the groups of mean magnitude features together across orients and scales - rather
            # than z-scoring each column, to preserve informative difference across these channels.
            zgroup_sizes_ll = [1,1,1,1,1,1] + list(feature_type_dims_ll[1:])
            zgroup_labels_ll = np.concatenate([np.ones(shape=(1, zgroup_sizes_ll[ff]))*ff \
                                               for ff in range(len(zgroup_sizes_ll))], axis=1)
            # For the marginal stats of lowpass recons, separating skew/kurtosis here
            zgroup_labels_ll[zgroup_labels_ll==9] = 10
            zgroup_labels_ll[0,np.where(zgroup_labels_ll==8)[1][np.arange(1,10,2)]] = 9
            zgroup_labels = zgroup_labels_ll
            features_to_use = np.arange(n_ll_feats+n_hl_feats)<n_ll_feats
            n_features = n_ll_feats
        else:
            zgroup_sizes_hl = list(feature_type_dims_hl)
            # for higher level groups, just retaining original grouping scheme 
            zgroup_labels_hl = np.concatenate([np.ones(shape=(1, zgroup_sizes_hl[ff]))*ff \
                                                   for ff in range(len(zgroup_sizes_hl))], axis=1)
            zgroup_labels = zgroup_labels_hl
            features_to_use = np.arange(n_ll_feats+n_hl_feats)>=n_ll_feats
            n_features = n_hl_feats
            
    elif feature_type=='gabor_solo':
        
        path_to_load = default_paths.gabor_texture_feat_path   
        n_ori = 12; n_sf = 8;
        features_file = os.path.join(path_to_load, \
                                 'S%d_features_each_prf_%dori_%dsf_gabor_solo_nonlin_grid%d.h5py'%\
                                 (subject,n_ori, n_sf, which_prf_grid))
        zgroup_labels = np.ones((n_ori*n_sf,))
        features_to_use = np.ones(np.shape(zgroup_labels))==1
        n_features = n_ori*n_sf
        
    else:
        raise RuntimeError('feature type %s not recognized'%feature_type)
      
    if not os.path.exists(features_file):
        raise RuntimeError('Looking at %s for precomputed features, not found.'%features_file)   

    # Choose where to save results
    path_to_save = os.path.join(path_to_load, 'LDA')
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    if zscore_groups==True:        
        fn2save = os.path.join(path_to_save, 'S%d_%s_LDA_zscoregroups_%s_grid%d.npy'%(subject, feature_type, discrim_type, which_prf_grid))
    elif zscore_each==False:
        fn2save = os.path.join(path_to_save, 'S%d_%s_LDA_nzscore_%s_grid%d.npy'%(subject, feature_type, discrim_type, which_prf_grid))
    elif balance_downsample==False:
        fn2save = os.path.join(path_to_save, 'S%d_%s_LDA_nobalance_%s_grid%d.npy'%(subject, feature_type, discrim_type, which_prf_grid))
    else:
        fn2save = os.path.join(path_to_save, 'S%d_%s_LDA_%s_grid%d.npy'%(subject, feature_type, discrim_type, which_prf_grid))
                              
    prf_batch_size = 50 # batching prfs for loading, because it is a bit faster
    n_prfs = models.shape[0]
    n_prf_batches = int(np.ceil(n_prfs/prf_batch_size))          
    prf_batch_inds = [np.arange(prf_batch_size*bb, np.min([prf_batch_size*(bb+1), n_prfs])) for bb in range(n_prf_batches)]
    prf_inds_loaded = []
 
    scores_each_prf = []
    wts_each_prf = []
    pre_mean_each_prf = []
    trn_acc_each_prf = []
    trn_dprime_each_prf = []
    val_acc_each_prf = []
    val_dprime_each_prf = []
    labels_pred_each_prf = []
    labels_actual_each_prf = []
    
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
   
    for prf_model_index in range(n_prfs):

        if debug and prf_model_index>1:
            continue
            
        print('\nProcessing pRF %d of %d'%(prf_model_index, n_prfs))
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
            features_each_prf_batch = values[:,features_to_use,:].astype(np.float32)
            values=None
                                 
        index_into_batch = np.where(prf_model_index==prf_inds_loaded)[0][0]
        print('Index into batch for prf %d: %d'%(prf_model_index, index_into_batch))
        features_in_prf = features_each_prf_batch[:,:,index_into_batch]
        assert(features_in_prf.shape[1]==n_features)
        values=None
        print('Size of features array for this image set and prf is:')
        print(features_in_prf.shape)

            
        # Gather semantic labels for the images, specific to this pRF position.
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
                labels = np.array(coco_df['has_animate']).astype(np.float32)
            else:
                labels = np.array(coco_df[discrim_type]).astype(np.float32)
            ims_to_use = np.any(np.array(coco_df)[:,0:12]==1, axis=1)
            labels[~ims_to_use] = np.nan
            neach = [np.sum(labels==ll) for ll in np.unique(labels[~np.isnan(labels)])] + \
                    [np.sum(np.isnan(labels))]
            print('yes %s/no %s/no labels:'%(discrim_type, discrim_type))
            print(neach)
            unvals = np.unique(labels[ims_to_use])                               
            print('unique values:')
            print(unvals)       
        elif discrim_type=='all_supcat':
            coco_labels_fn = os.path.join(labels_folder, \
                                          'S%d_cocolabs_binary_prf%d.csv'%(subject, prf_model_index))
            print('Reading labels from %s...'%coco_labels_fn)
            coco_df = pd.read_csv(coco_labels_fn, index_col=0)            
            supcat_labels = np.array(coco_df)[:,0:12]            
            labels = [np.where(supcat_labels[ii,:])[0] for ii in range(supcat_labels.shape[0])]
            labels = np.array([ll[0] if len(ll)>0 else -1 for ll in labels]).astype(np.float32)
            ims_to_use = np.sum(supcat_labels, axis=1)==1
            labels[~ims_to_use] = np.nan
            print('Proportion of images that have exactly one super-cat label:')
            print(np.mean(ims_to_use))           
            print('Unique labels:')
            unvals, counts = np.unique(labels[ims_to_use], return_counts=True)
            print(np.unique(labels[ims_to_use]))
            print(counts)
        else:
            raise ValueError('discrimination type %s not recognized.'%discrim_type)
        
       
        assert(np.all(ims_to_use==~np.isnan(labels)))
          
        # z-score in advance of computing lda 
        if zscore_each:
            trn_mean = np.mean(features_in_prf[trninds,:], axis=0, keepdims=True)
            trn_std = np.std(features_in_prf[trninds,:], axis=0, keepdims=True)
            features_in_prf_z = (features_in_prf - np.tile(trn_mean, [features_in_prf.shape[0],1]))/ \
                    np.tile(trn_std, [features_in_prf.shape[0],1])
        elif zscore_groups:
            features_in_prf_z = np.zeros_like(features_in_prf)
            trnz, tstz = numpy_utils.zscore_in_groups_trntest(features_in_prf[trninds,:],\
                                                              features_in_prf[~trninds,:], zgroup_labels)
            features_in_prf_z[trninds,:] = trnz
            features_in_prf_z[~trninds,:] = tstz
        else:
            features_in_prf_z = features_in_prf
            
        # Identify the LDA subspace, based on training data only.
        _, scalings, trn_acc, trn_dprime, clf = do_lda(features_in_prf_z[trninds & ims_to_use,:], \
                                                       labels[trninds & ims_to_use], \
                                                       verbose=True, balance_downsample=balance_downsample)
        
        # applying the transformation to all images. Including both train and validation here.
        scores = clf.transform(features_in_prf_z)
        print(scores.shape)
        labels_pred = clf.predict(features_in_prf_z)
        pre_mean = clf.xbar_
        
        val_acc = np.mean(labels_pred[valinds & ims_to_use]==labels[valinds & ims_to_use])
        val_dprime = stats_utils.get_dprime(labels_pred[valinds & ims_to_use], \
                                            labels[valinds & ims_to_use], un=unvals)
        print('Validation data prediction accuracy = %.2f pct'%(val_acc*100))
        print('Validation data dprime = %.2f'%(val_dprime))
        print('Proportion predictions each category:')        
        n_each_cat_pred = [np.sum(labels_pred==cc) for cc in np.unique(labels_pred)]
        print(np.round(n_each_cat_pred/np.sum(n_each_cat_pred), 2))
        
        trn_acc_each_prf.append(trn_acc)
        trn_dprime_each_prf.append(trn_dprime)
        val_acc_each_prf.append(val_acc)
        val_dprime_each_prf.append(val_dprime)
        
        scores_each_prf.append(scores)
        wts_each_prf.append(scalings)       
        pre_mean_each_prf.append(pre_mean)
        
        labels_pred_each_prf.append(labels_pred)
        labels_actual_each_prf.append(labels)
        
        sys.stdout.flush()

    
    print('saving to %s'%fn2save)
    np.save(fn2save, {'scores': scores_each_prf,'wts': wts_each_prf, \
                      'pre_mean': pre_mean_each_prf, \
                      'trn_acc': trn_acc_each_prf, \
                      'trn_dprime': trn_dprime_each_prf, 'val_acc': val_acc_each_prf, \
                      'val_dprime': val_dprime_each_prf, \
                     'labels_actual': labels_actual_each_prf, 'labels_pred': labels_pred_each_prf})


def do_lda(values, categ_labels, verbose=False, balance_downsample=True, rndseed=None):
    """
    Apply linear discriminant analysis to the data - find axes that best separate the given category labels.
    """
    n_features_actual = values.shape[1]
    n_trials = values.shape[0]
    n_each_cat = [np.sum(categ_labels==cc) for cc in np.unique(categ_labels)]
    if verbose:        
        print('Running linear discriminant analysis: original size of array is [%d x %d]'%(n_trials, \
                                                                                           n_features_actual))
    
    # Balance class labels    
    min_samples = np.min(n_each_cat)
    if balance_downsample and not np.all(n_each_cat==min_samples):
        
        if verbose:
            print('\nBefore downsampling - proportion samples each category:')               
            print(np.round(n_each_cat/np.sum(n_each_cat), 2)) 

        un = np.unique(categ_labels)
        if rndseed is None:
            if verbose:
                print('Computing a new random seed')
            shuff_rnd_seed = int(time.strftime('%S%M%H', time.localtime()))
        if verbose:
            print('Seeding random number generator: seed is %d'%shuff_rnd_seed)
        np.random.seed(shuff_rnd_seed)
        samples_to_use = []
        for ii in range(len(un)):
            samples_to_use += [np.random.choice(np.where(categ_labels==un[ii])[0], min_samples, replace=False)]
        samples_to_use = np.array(samples_to_use).ravel()
        
        values = values[samples_to_use,:]
        categ_labels = categ_labels[samples_to_use]
        n_each_cat = [np.sum(categ_labels==cc) for cc in np.unique(categ_labels)]
    
        if verbose:
            print('\nAfter downsampling - proportion samples each category:')               
            print(np.round(n_each_cat/np.sum(n_each_cat), 2)) 
            print('Number of samples each category:')
            print(n_each_cat) 
        
    elif verbose:       
        print('\nProportion samples each category:')        
        print(np.round(n_each_cat/np.sum(n_each_cat), 2)) 
        
    t = time.time()
    
    X = values; y = np.squeeze(categ_labels)
    clf = LinearDiscriminantAnalysis(solver='svd')
    clf.fit(X, y)

    elapsed = time.time() - t
    if verbose:
        print('Time elapsed: %.5f'%elapsed)
    
    ypred = clf.predict(X)
    trn_acc = np.mean(y==ypred)
    trn_dprime = stats_utils.get_dprime(ypred, y)
    if verbose:
        print('Training data prediction accuracy = %.2f pct'%(trn_acc*100))
        print('Training data dprime = %.2f'%(trn_dprime))
        print('Proportion predictions each category:')        
        n_each_cat_pred = [np.sum(ypred==cc) for cc in np.unique(categ_labels)]
        print(np.round(n_each_cat_pred/np.sum(n_each_cat_pred), 2))

    # Transform method is doing : scores = (X - clf.xbar_) @ clf.scalings_
    # So the "scalings" define a mapping from mean-centered feature data to the LDA subspace.
    scores = clf.transform(X)    
    scalings = clf.scalings_
   
    return scores, scalings, trn_acc, trn_dprime, clf


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
    parser.add_argument("--zscore_each", type=int,default=0,
                    help="want to zscore each column before decoding? 1 for yes, 0 for no")
    parser.add_argument("--zscore_groups", type=int,default=0,
                    help="want to zscore in groups of columns before decoding? 1 for yes, 0 for no")
    parser.add_argument("--balance_downsample", type=int,default=0,
                    help="want to re-sample if classes are unbalanced? 1 for yes, 0 for no")
    args = parser.parse_args()

    if args.debug:
        print('DEBUG MODE\n')
     
    find_lda_axes(subject=args.subject, feature_type=args.feature_type, debug=args.debug==1, \
                  discrim_type=args.discrim_type, which_prf_grid=args.which_prf_grid, \
                  zscore_each = args.zscore_each==1, zscore_groups = args.zscore_groups==1, \
                  balance_downsample = args.balance_downsample==1)
    