import sys, os
import numpy as np
import time, h5py
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import argparse
import pandas as pd   

codepath = '/user_data/mmhender/imStat/code'
sys.path.append(codepath)
from utils import default_paths, nsd_utils, numpy_utils, stats_utils, coco_utils
from model_fitting import initialize_fitting 
from feature_extraction import texture_statistics_gabor, sketch_token_features, \
                texture_statistics_pyramid

def find_lda_axes(subject, feature_type, which_prf_grid=1, debug=False, \
                  zscore_each=False, balance_downsample=False, device=None):
  

    if device is None:
        device = 'cpu:0'

    print('\nusing prf grid %d\n'%(which_prf_grid))
    # Params for the spatial aspect of the model (possible pRFs)
    models = initialize_fitting.get_prf_models(which_grid = which_prf_grid)    
    n_prfs = len(models)

    if subject=='all':       
        subjects = np.arange(1,9)
    else:
        subjects = [int(subject)]
    print('Using images/labels for subjects:')
    print(subjects)
    
    # load and concatenate coco labels for the subs of interest
    for si, ss in enumerate(subjects):
        # training / validation data always split the same way - shared 1000 inds are validation.
        subject_df = nsd_utils.get_subj_df(ss)
        valinds_ss = np.array(subject_df['shared1000'])
        trninds_ss = np.array(subject_df['shared1000']==False)
        image_inds_ss = np.arange(len(trninds_ss))
        labels_all_ss, discrim_type_list_ss, unique_labels_each_ss = coco_utils.load_labels_each_prf(ss, \
                             which_prf_grid, image_inds=image_inds_ss, models=models,verbose=False)
        if si==0:
            labels_all = labels_all_ss
            discrim_type_list = discrim_type_list_ss
            unique_labels_each = unique_labels_each_ss
            trninds_full = trninds_ss
            valinds_full = valinds_ss
        else:
            labels_all = np.concatenate([labels_all, labels_all_ss], axis=0)
            trninds_full = np.concatenate([trninds_full, trninds_ss],axis=0)
            # validation set images (1000) are identical for all subs, so only going to grab the first 1000.
            valinds_full = np.concatenate([valinds_full, np.zeros(np.shape(valinds_ss), dtype=bool)], axis=0)
            unique_labels_each = [np.unique(np.concatenate([unique_labels_each[ii],unique_labels_each_ss[ii]], axis=0))\
                                 for ii in range(len(unique_labels_each))]
            # check that columns are same for all subs
            assert(np.all(np.array(discrim_type_list)==np.array(discrim_type_list_ss)))
   
    print('Number of images using: %d'%labels_all.shape[0])
     
    n_trials = labels_all.shape[0]
    n_sem_axes = labels_all.shape[1] 

    # create feature extractor (just a module that will load the pre-computed features easily)
    if feature_type=='sketch_tokens':

        path_to_load = default_paths.sketch_token_feat_path
        _feature_extractors = [sketch_token_features.sketch_token_feature_extractor(subject=ss, device=device,\
                         which_prf_grid=which_prf_grid, \
                         use_pca_feats = False) for ss in subjects];

    elif 'pyramid_texture' in feature_type:

        path_to_load = default_paths.pyramid_texture_feat_path
        # Set up the pyramid loader       
        if feature_type=='pyramid_texture_ll': 
            include_ll=True
            include_hl=False
            use_pca_feats_hl = False
        elif feature_type=='pyramid_texture_hl':
            include_ll=False
            include_hl=True
            use_pca_feats_hl = False
        elif feature_type=='pyramid_texture_hl_pca':
            include_ll=False
            include_hl=True
            use_pca_feats_hl = True

        compute_features = False; do_varpart=False; group_all_hl_feats=True;
        n_ori = 4; n_sf = 4;
        _fmaps_fn = texture_statistics_pyramid.steerable_pyramid_extractor(pyr_height = n_sf, n_ori = n_ori)
        _feature_extractors = [texture_statistics_pyramid.texture_feature_extractor(_fmaps_fn,\
                  subject=ss, include_ll=include_ll, include_hl=include_hl, \
                  use_pca_feats_hl = use_pca_feats_hl,\
                  which_prf_grid=which_prf_grid, \
                  do_varpart = do_varpart, group_all_hl_feats = group_all_hl_feats, \
                  compute_features = compute_features, device=device) for ss in subjects]

    elif feature_type=='gabor_solo':

        path_to_load = default_paths.gabor_texture_feat_path   
        feature_types_exclude = ['pixel', 'simple_feature_means', 'autocorrs', 'crosscorrs']
        gabor_nonlin_fn=True
        n_ori=12; n_sf=8; autocorr_output_pix=5
        _gabor_ext_complex, _gabor_ext_simple, _fmaps_fn_complex, _fmaps_fn_simple = \
                    initialize_fitting.get_gabor_feature_map_fn(n_ori, n_sf,device=device,\
                    nonlin_fn=gabor_nonlin_fn);    
        compute_features = False; do_varpart=False; group_all_hl_feats_gabor = False;
        _feature_extractors = [texture_statistics_gabor.texture_feature_extractor(\
                    _fmaps_fn_complex,_fmaps_fn_simple,\
                    subject=ss, which_prf_grid=which_prf_grid,\
                    autocorr_output_pix=autocorr_output_pix, \
                    feature_types_exclude=feature_types_exclude, do_varpart=do_varpart, \
                    group_all_hl_feats=group_all_hl_feats_gabor, nonlin_fn=gabor_nonlin_fn, \
                    compute_features = compute_features, device=device) for ss in subjects]
    else:
        raise RuntimeError('feature type %s not recognized'%feature_type)

    _feature_extractor.init_for_fitting(image_size=None, models=models, dtype=np.float32)

    # Choose where to save results
    path_to_save = os.path.join(path_to_load, 'LDA')
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    if subject=='all':
        fn2save = os.path.join(path_to_save, 'All_%s_LDA_all_grid%d.npy'%(feature_type, which_prf_grid))
    else:
        fn2save = os.path.join(path_to_save, 'S%s_%s_LDA_all_grid%d.npy'%(subject, feature_type, which_prf_grid))

       
    trn_acc_each_prf = np.zeros((n_prfs, n_sem_axes), dtype=np.float32)
    val_acc_each_prf = np.zeros((n_prfs, n_sem_axes), dtype=np.float32)
    trn_dprime_each_prf = np.zeros((n_prfs, n_sem_axes), dtype=np.float32)
    val_dprime_each_prf = np.zeros((n_prfs, n_sem_axes), dtype=np.float32)

    for prf_model_index in range(n_prfs):

        if debug and prf_model_index>1:
            continue

        print('\nProcessing pRF %d of %d'%(prf_model_index, n_prfs))

        for si, _feature_extractor in enumerate(_feature_extractors):
            features_in_prf_ss, feature_inds_defined = _feature_extractor(image_inds_ss, models[prf_model_index,:], \
                                                            prf_model_index, fitting_mode=False)
            features_in_prf_ss = features_in_prf_ss.numpy()
            if si==0:
                features_in_prf = features_in_prf_ss
            else:
                features_in_prf = np.concatenate([features_in_prf,features_in_prf_ss], axis=0)

        print('Size of features array for this image set and prf is:')
        print(features_in_prf.shape)
        assert(not np.any(np.isnan(features_in_prf)))
        assert(not np.any(np.sum(features_in_prf, axis=0)==0))

        for aa in range(n_sem_axes):

            labels = labels_all[:,aa,prf_model_index]
            inds2use = ~np.isnan(labels)     
            any_in_val = np.any(inds2use & valinds_full)
            unique_labels_actual = np.unique(labels[inds2use & trninds_full])

            if prf_model_index==0:
                print('processing axis: %s'%discrim_type_list[aa])
                print('labels: ')
                print(unique_labels_each[aa])  

            if np.all(np.isin(unique_labels_each[aa], unique_labels_actual)) and any_in_val:
           
                if zscore_each:
                    # z-score in advance of computing lda 
                    trn_mean = np.mean(features_in_prf[trninds_full,:], axis=0, keepdims=True)
                    trn_std = np.std(features_in_prf[trninds_full,:], axis=0, keepdims=True)
                    features_in_prf_z = (features_in_prf - np.tile(trn_mean, [features_in_prf.shape[0],1]))/ \
                            np.tile(trn_std, [features_in_prf.shape[0],1])
                else:
                    features_in_prf_z = features_in_prf

                # Identify the LDA subspace, based on training data only.
                _, scalings, trn_acc, trn_dprime, clf = do_lda(features_in_prf_z[trninds_full & inds2use,:], \
                                                               labels[trninds_full & inds2use], \
                                                               verbose=True, balance_downsample=balance_downsample)
                if clf is not None:

                    # applying the transformation to all images. Including both train and validation here.
                    # scores = clf.transform(features_in_prf_z)
                    #  print(scores.shape)
                    labels_pred = clf.predict(features_in_prf_z)
                    #  pre_mean = clf.xbar_

                    val_acc = np.mean(labels_pred[valinds_full & inds2use]==labels[valinds_full & inds2use])
                    if np.all(np.isin(unique_labels_actual, np.unique(labels[valinds_full & inds2use]))):
                        val_dprime = stats_utils.get_dprime(labels_pred[valinds_full & inds2use], \
                                                        labels[valinds_full & inds2use], un=unique_labels_actual)
                    else:
                        print('Missing at least one category in validation set for model %d, axis %d'%(prf_model_index, aa))
                        val_dprime = np.nan
                    print('Validation data prediction accuracy = %.2f pct'%(val_acc*100))
                    print('Validation data dprime = %.2f'%(val_dprime))
                    print('Proportion predictions each category:')        
                    n_each_cat_pred = [np.sum(labels_pred==cc) for cc in np.unique(labels_pred)]
                    print(np.round(n_each_cat_pred/np.sum(n_each_cat_pred), 2))

                    trn_acc_each_prf[prf_model_index,aa] = trn_acc
                    trn_dprime_each_prf[prf_model_index,aa] = trn_dprime
                    val_acc_each_prf[prf_model_index,aa] = val_acc
                    val_dprime_each_prf[prf_model_index,aa] = val_dprime

                else:
                    print('Problem with LDA fit, returning nans for model %d, axis %d'%(prf_model_index, aa))
                    trn_acc_each_prf[prf_model_index,aa] = np.nan
                    trn_dprime_each_prf[prf_model_index,aa] = np.nan
                    val_acc_each_prf[prf_model_index,aa] = np.nan
                    val_dprime_each_prf[prf_model_index,aa] = np.nan

            else:
                if any_in_val:
                    print('missing some labels for axis %d, model %d'%(aa, prf_model_index))
                    print('expected labels')
                    print(unique_labels_each[aa])
                    print('actual labels')
                    print(unique_labels_actual)
                else:
                    print('nans for model %d, axis %d - no labeled trials in validation set'\
                          %(prf_model_index, aa))
                trn_acc_each_prf[prf_model_index,aa] = np.nan
                trn_dprime_each_prf[prf_model_index,aa] = np.nan
                val_acc_each_prf[prf_model_index,aa] = np.nan
                val_dprime_each_prf[prf_model_index,aa] = np.nan

            sys.stdout.flush()


    print('saving to %s'%fn2save)
    np.save(fn2save, {'trn_acc': trn_acc_each_prf, \
                      'trn_dprime': trn_dprime_each_prf, 'val_acc': val_acc_each_prf, \
                      'val_dprime': val_dprime_each_prf})
            
    

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
    try:
        clf.fit(X, y)
    except ValueError as e:
        print(e)
        print('error with LDA fit, returning nans')
        return None, None, None, None, None
    
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
    
    parser.add_argument("--subject", type=str,default=1,
                    help="number of the subject, 1-8 or all")
    parser.add_argument("--feature_type", type=str,default='sketch_tokens',
                    help="what kind of features are we using?")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which prf grid to use")
    parser.add_argument("--zscore_each", type=int,default=0,
                    help="want to zscore each column before decoding? 1 for yes, 0 for no")
    parser.add_argument("--balance_downsample", type=int,default=0,
                    help="want to re-sample if classes are unbalanced? 1 for yes, 0 for no")
    args = parser.parse_args()

    if args.debug:
        print('DEBUG MODE\n')
     
    find_lda_axes(subject=args.subject, feature_type=args.feature_type, debug=args.debug==1, \
                  which_prf_grid=args.which_prf_grid, \
                  zscore_each = args.zscore_each==1, \
                  balance_downsample = args.balance_downsample==1)
    