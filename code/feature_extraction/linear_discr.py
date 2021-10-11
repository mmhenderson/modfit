import sys, os
import numpy as np
import time, h5py
codepath = '/user_data/mmhender/imStat/code'
sys.path.append(codepath)
from utils import default_paths, coco_utils, nsd_utils, numpy_utils, stats_utils
from model_fitting import initialize_fitting 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import argparse
import pandas as pd   
    
def find_lda_axes(subject, discrim_type='animacy', debug=False):

    # load full-dimension features, already computed 
    path_to_load = default_paths.sketch_token_feat_path
    features_file = os.path.join(path_to_load, 'S%d_features_each_prf.h5py'%(subject))
    if not os.path.exists(features_file):
        raise RuntimeError('Looking at %s for precomputed features, not found.'%features_file)   
    print('Loading pre-computed features from %s'%features_file)
    t = time.time()
    with h5py.File(features_file, 'r') as data_set:
        values = np.copy(data_set['/features'])
        data_set.close() 
    elapsed = time.time() - t
    print('Took %.5f seconds to load file'%elapsed)
    features_each_prf = values

    # Choose where to save results
    path_to_save = os.path.join(path_to_load, 'LDA')
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    # how to do z-scoring? can set up groups of columns here.
    zgroup_labels = np.concatenate([np.zeros(shape=(1,150)), np.ones(shape=(1,1))], axis=1)

    # training / validation data always split the same way - shared 1000 inds are validation.
    subject_df = nsd_utils.get_subj_df(subject)
    valinds = np.array(subject_df['shared1000'])
    trninds = np.array(subject_df['shared1000']==False)
    
    print('Size of features array for this image set is:')
    print(features_each_prf.shape)

    # Params for the spatial aspect of the model (possible pRFs)
    aperture_rf_range = 1.1
    aperture, models = initialize_fitting.get_prf_models(aperture_rf_range=aperture_rf_range)    
    n_prfs = models.shape[0]
    
    scores_each_prf = []
    wts_each_prf = []
    trn_acc_each_prf = []
    trn_dprime_each_prf = []
    val_acc_each_prf = []
    val_dprime_each_prf = []
    labels_pred_each_prf = []

    for prf_model_index in range(n_prfs):

        if debug and prf_model_index>1:
            continue
            
        print('\nProcessing pRF %d of %d'%(prf_model_index, n_prfs))

        # Gather semantic labels for the images, specific to this pRF position. 
        if discrim_type=='animacy':
            coco_labels_fn = os.path.join(default_paths.stim_labels_root, 'S%d_within_prf'%subject, \
                                          'S%d_cocolabs_binary_prf%d.csv'%(subject, prf_model_index))
            print('Reading labels from %s...'%coco_labels_fn)
            coco_df = pd.read_csv(coco_labels_fn, index_col=0)
            labels = np.array(coco_df['has_animate'])
            unvals = np.unique(labels)
            print('Overall proportion animate this pRF:')
            print(np.mean(labels==1))
            print('Proportion w any annotation this pRF:')
            mat = np.array(coco_df)
            ims_to_use = np.any(mat, axis=1)
            print(np.mean(ims_to_use))
            ims_to_use = np.ones(np.shape(labels))==1
            
        elif discrim_type=='all_supcat':
            coco_labels_fn = os.path.join(default_paths.stim_labels_root, 'S%d_within_prf'%subject, \
                                          'S%d_cocolabs_binary_prf%d.csv'%(subject, prf_model_index))
            print('Reading labels from %s...'%coco_labels_fn)
            coco_df = pd.read_csv(coco_labels_fn, index_col=0)
            
            supcat_labels = np.array(coco_df)[:,0:12]
            
            labels = [np.where(supcat_labels[ii,:])[0] for ii in range(supcat_labels.shape[0])]
            labels = np.array([ll[0] if len(ll)>0 else -1 for ll in labels])

            ims_to_use = np.sum(supcat_labels, axis=1)==1
            print('Proportion of images that have exactly one super-cat label:')
            print(np.mean(ims_to_use))
            
            print('Unique labels:')
            unvals = np.unique(labels[ims_to_use])
            print(np.unique(labels[ims_to_use]))

        else:
            raise ValueError('discrimination type %s not recognized.'%discrim_type)
        
        features_in_prf = features_each_prf[:,:,prf_model_index]
        print('Size of features array for this image set and prf is:')
        print(features_in_prf.shape)
        
        # z-score in advance of computing lda 
        features_in_prf_z = numpy_utils.zscore_in_groups(features_in_prf, zgroup_labels)

        # Identify the LDA subspace, based on training data only.
#         _, scalings, trn_acc, trn_dprime, clf = do_lda(features_in_prf_z[trninds,:], labels[trninds], \
#                                                        verbose=True, balance_downsample=False)
        
        _, scalings, trn_acc, trn_dprime, clf = do_lda(features_in_prf_z[trninds & ims_to_use,:], labels[trninds & ims_to_use], \
                                                       verbose=True, balance_downsample=False)
        
        # applying the transformation to all images. Including both train and validation here.
        scores = clf.transform(features_in_prf_z)
        print(scores.shape)
        labels_pred = clf.predict(features_in_prf_z)
        
        val_acc = np.mean(labels_pred[valinds]==labels[valinds])
        val_dprime = stats_utils.get_dprime(labels_pred[valinds], labels[valinds], un=unvals)
        print('Validation data prediction accuracy = %.2f pct'%(val_acc*100))
        print('Validation data dprime = %.2f'%(val_dprime))
        print('Proportion predictions each category:')        
        n_each_cat_pred = [np.sum(labels_pred==cc) for cc in np.unique(labels_pred)]
        print(np.round(n_each_cat_pred/np.sum(n_each_cat_pred), 2))
        
        trn_acc_each_prf.append(trn_acc)
        trn_acc_each_prf.append(trn_dprime)
        
        scores_each_prf.append(scores)
        wts_each_prf.append(scalings)       
        labels_pred_each_prf.append(labels_pred)
        
        sys.stdout.flush()

    fn2save = os.path.join(path_to_save, 'S%d_LDA_%s.npy'%(subject, discrim_type))
    print('saving to %s'%fn2save)
    np.save(fn2save, {'scores': scores_each_prf,'wts': wts_each_prf, 'trn_acc': trn_acc_each_prf, \
                      'trn_dprime': trn_dprime_each_prf, 'val_acc': val_acc_each_prf, \
                      'val_dprime': val_dprime_each_prf, \
                     'labels': labels, 'labels_pred': labels_pred_each_prf})

def run_lda_crossval(subject, debug=False, n_cv=10):
    
    path_to_load = default_paths.sketch_token_feat_path

    features_file = os.path.join(path_to_load, 'S%d_features_each_prf.h5py'%(subject))
    if not os.path.exists(features_file):
        raise RuntimeError('Looking at %s for precomputed features, not found.'%features_file)   
    path_to_save = os.path.join(path_to_load, 'LDA')
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    # Params for the spatial aspect of the model (possible pRFs)
    aperture_rf_range = 1.1
    aperture, models = initialize_fitting.get_prf_models(aperture_rf_range=aperture_rf_range)    
    n_prfs = models.shape[0]

    print('Loading pre-computed features from %s'%features_file)
    t = time.time()
    with h5py.File(features_file, 'r') as data_set:
        values = np.copy(data_set['/features'][:,:,0:2])
        data_set.close() 
    elapsed = time.time() - t
    print('Took %.5f seconds to load file'%elapsed)
    features_each_prf = values

    zgroup_labels = np.concatenate([np.zeros(shape=(1,150)), np.ones(shape=(1,1))], axis=1)
    
    print('Size of features array for this image set is:')
    print(features_each_prf.shape)

    # Gather basic info about the images included in the features matrix (coco ids)
    subject_df = nsd_utils.get_subj_df(subject);
    all_coco_ids = np.array(subject_df['cocoId'])

    # Gather labels - here these are the 1000 hand labels generated as part of food project.
    # Only have these labels for validation set images.
    labels_file = os.path.join(default_paths.stim_root, 'shared_1000_all_labels_matrix.csv')
    labels_df = pd.read_csv(labels_file)
    # sort this into order to match my subject's sequence of stimuli
    df_inds = [np.where(np.array(labels_df['cocoId'])==all_coco_ids[ii])[0][0] for ii in range(1000)]
    labels_df_sorted = labels_df.loc[df_inds]
    labels_df_sorted = labels_df_sorted.set_index(np.arange(1000))
    assert(np.all(np.array(labels_df_sorted['cocoId'])==all_coco_ids[0:1000]))
    
    # binary labels for the 16 category attributes
    categ_binary = np.array(labels_df_sorted)[:,9:]
    categ_names = list(labels_df_sorted.keys())[9:]
    print(categ_names)
    print(categ_binary.shape)
    
    # which classification schemes to do, looping over these
    class_names = ['indoor_outdoor','scale','all_objects','food_face','food_object','food_vs_nofood']
    categ_use_list = [[0,1],[12,13,14],[3,4,5,6,7,8,9,10,11,15], [4,8], [8, 15], [8,-1]]

    for ci, categ_use in enumerate(categ_use_list):

        print('\nClassifying based on %s'%class_names[ci])
        print(categ_use)
        if categ_use[1]==-1:
            ims_to_use = np.where(np.ones((categ_binary.shape[0],))==1)[0]
            categ_labels = categ_binary[:,categ_use[0]]==1
            print('Labels to classify between (n-way):')
            print(['no '+categ_names[categ_use[0]]] + [categ_names[categ_use[0]]])

        else:    
            ims_to_use = np.where(np.sum(categ_binary[:,categ_use],axis=1)==1)[0]
            categ_labels = np.array([np.where(categ_binary[cc,categ_use])[0][0] for cc in ims_to_use])
            print('Labels to classify between (n-way):')
            print([categ_names[cc] for cc in categ_use])

        print('unique labels:')
        print(np.unique(categ_labels))

        print('Across all %d images: proportion samples each category:'%len(ims_to_use))        
        n_each_cat = [np.sum(categ_labels==cc) for cc in np.unique(categ_labels)]
        print(np.round(n_each_cat/np.sum(n_each_cat), 2))


        n_per_cv = int(np.ceil(len(categ_labels)/n_cv))
        cv_labels = np.repeat(np.arange(n_cv), n_per_cv)
        cv_labels = cv_labels[0:len(categ_labels)]
        [np.sum(cv_labels==cv) for cv in range(n_cv)]

        class_each_prf = []

        for prf_model_index in range(n_prfs):

            if debug and prf_model_index>1:
                continue

            print('\nProcessing pRF %d of %d'%(prf_model_index, n_prfs))

            features_in_prf = features_each_prf[ims_to_use,:,prf_model_index]
            features_in_prf_z = numpy_utils.zscore_in_groups(features_in_prf, zgroup_labels)

            print('Size of features array for this image set and prf is:')
            print(features_in_prf.shape)

            pred_categ_labels = np.zeros(np.shape(categ_labels))

            for cv in range(n_cv):

                print('    cv %d of %d'%(cv, n_cv))
                trninds = cv_labels!=cv
                tstinds = cv_labels==cv

                trn_features = features_in_prf[trninds,:]
                trn_features_z = numpy_utils.zscore_in_groups(trn_features, zgroup_labels)
                tst_features = features_in_prf[tstinds,:]
                tst_features_z = numpy_utils.zscore_in_groups(tst_features, zgroup_labels)

                scores, scalings, trn_acc, trn_dprime, clf = do_lda(trn_features_z, categ_labels[trninds], \
                                                                    verbose=True, balance_downsample=True)

                pred = clf.predict(tst_features_z)
                pred_categ_labels[tstinds] = pred

            print('\nFull dataset: proportion predictions each category:')        
            n_each_cat_pred = [np.sum(pred_categ_labels==cc) for cc in np.unique(categ_labels)]
            print(np.round(n_each_cat_pred/np.sum(n_each_cat_pred), 2))

            acc = np.mean(categ_labels==pred_categ_labels)
            dprime = stats_utils.get_dprime(pred_categ_labels, categ_labels)
            print('Held-out data acc = %.2f'%(acc*100))
            print('Held-out data d-prime = %.2f'%(dprime))

            class_each_prf.append({'acc': acc, 'dprime': dprime, 'actual_categ_labels': categ_labels, \
                                   'pred_categ_labels': pred_categ_labels})

        fn2save = os.path.join(path_to_save, 'S%d_LDA_preds_crossval_%s.npy'%(subject, class_names[ci]))
        print('saving to %s'%fn2save)
        np.save(fn2save, class_each_prf)


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
    parser.add_argument("--do_class_cv", type=int, default=0, 
                    help="want to perform cross-validated decoding w LDA and save performance?")
    parser.add_argument("--do_features", type=int, default=0, 
                    help="want to find category diagnostic features and save them?")
    
    args = parser.parse_args()

    if args.debug:
        print('DEBUG MODE\n')
        
    if args.feature_type=='sketch_tokens':
        if args.do_features:
            find_lda_axes(subject=args.subject, debug=args.debug==1, discrim_type=args.discrim_type)
        if args.do_class_cv:
            run_lda_crossval(subject=args.subject, debug=args.debug==1)
            
    else:
        raise ValueError('--feature_type %s is not recognized'%args.feature_type)