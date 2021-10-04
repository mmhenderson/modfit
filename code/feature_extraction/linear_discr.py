import sys, os
import numpy as np
import time, h5py
codepath = '/user_data/mmhender/imStat/code'
sys.path.append(codepath)
from utils import default_paths, coco_utils, nsd_utils, numpy_utils
from model_fitting import initialize_fitting 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import argparse
    
def run_lda_sketch_tokens(subject, debug=False):

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
        values = np.copy(data_set['/features'])
        data_set.close() 
    elapsed = time.time() - t
    print('Took %.5f seconds to load file'%elapsed)
    features_each_prf = values

    zgroup_labels = np.concatenate([np.zeros(shape=(1,150)), np.ones(shape=(1,1))], axis=1)

    print('Size of features array for this image set is:')
    print(features_each_prf.shape)

    scores_each_prf = []
    wts_each_prf = []
    trn_acc_each_prf = []
    
    # Gather semantic labels for the images (COCO super-categories)
    coco_trn, coco_val = coco_utils.init_coco()
    cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat = coco_utils.get_coco_cat_info(coco_val)
    subject_df = nsd_utils.get_subj_df(subject);
    all_coco_ids = np.array(subject_df['cocoId'])
    ims_each_supcat = []
    for sc, scname in enumerate(supcat_names):
        ims_in_supcat = coco_utils.get_ims_in_supcat(coco_trn, coco_val, scname, all_coco_ids)
        ims_each_supcat.append(ims_in_supcat)
    ims_each_supcat = np.array(ims_each_supcat)
    supcats_each_image = [np.where(ims_each_supcat[:,ii])[0] for ii in range(ims_each_supcat.shape[1])]
    
    # For now, choosing just the images which have only one super-category present.
    ims_to_use = [sc for sc in range(len(supcats_each_image)) if len(supcats_each_image[sc])==1]
    supcat_labels = np.array([supcats_each_image[sc] for sc in ims_to_use])
    n_each_supcat = [np.sum(supcat_labels==sc) for sc in np.unique(supcat_labels)]
    print('\nProportion data labels each super-cat:')
    print(supcat_names)
    print(n_each_supcat)
    print(np.sum(n_each_supcat))

    for prf_model_index in range(n_prfs):

        if debug and prf_model_index>1:
            continue

        print('\nProcessing pRF %d of %d'%(prf_model_index, n_prfs))

        features_in_prf = features_each_prf[:,:,prf_model_index]
        features_in_prf_z = numpy_utils.zscore_in_groups(features_in_prf, zgroup_labels)

        print('Size of features array for this image set and prf is:')
        print(features_in_prf.shape)

        _, wts, trn_acc, clf = do_lda(features_in_prf_z[ims_to_use,:], supcat_labels)
        # applying the transformation to all images. 
        scores = clf.transform(features_in_prf_z)
        print(scores.shape)
        scores_each_prf.append(scores)
        wts_each_prf.append(wts)
        trn_acc_each_prf.append(trn_acc)

    fn2save = os.path.join(path_to_save, 'S%d_LDA.npy'%(subject))
    print('saving to %s'%fn2save)
    np.save(fn2save, {'scores': scores_each_prf,'wts': wts_each_prf, 'trn_acc': trn_acc_each_prf})

    
def do_lda(values, categ_labels):
    """
    Apply linear discriminant analysis to the data - find axes that best separate the given category labels.
    """
    n_features_actual = values.shape[1]
    n_trials = values.shape[0]
    
    print('Running linear discriminant analysis: original size of array is [%d x %d]'%(n_trials, n_features_actual))
    t = time.time()
    
    X = values; y = np.squeeze(categ_labels)
    clf = LinearDiscriminantAnalysis(solver='svd')
    clf.fit(X, y)

    elapsed = time.time() - t
    print('Time elapsed: %.5f'%elapsed)
    
    ypred = clf.predict(X)
    trn_acc = np.mean(y==ypred)
    print('Training data prediction accuracy = %.2f pct'%(trn_acc*100))

    print('\nProportion predictions each category:')
    n_each_cat_pred = [np.sum(ypred==cc) for cc in np.unique(categ_labels)]
    print(n_each_cat_pred)

    # Transform method is doing : scores = (X - clf.xbar_) @ clf.scalings_
    # So the "scalings" define a mapping from mean-centered feature data to the LDA subspace.
    scores = clf.transform(X)    
    scalings = clf.scalings_
   
    return scores, scalings, trn_acc, clf


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=1,
                    help="number of the subject, 1-8")
    parser.add_argument("--type", type=str,default='sketch_tokens',
                    help="what kind of features are we using?")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    
    args = parser.parse_args()

    if args.type=='sketch_tokens':
        run_lda_sketch_tokens(subject=args.subject, debug=args.debug==1)
    else:
        raise ValueError('--type %s is not recognized'%args.type)