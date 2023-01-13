import sys, os
import numpy as np
import time, h5py
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import argparse
import pandas as pd   

from utils import default_paths, nsd_utils, stats_utils, prf_utils
from model_fitting import initialize_fitting 
from feature_extraction import default_feature_loaders

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def run_decoding(subject=999, \
                 feature_type='gabor_solo', \
                 which_prf_grid=1, debug=False, \
                 zscore_each=False, match_prf_trialcounts=True, \
                 layer_name=None):
  
    print('\nusing prf grid %d\n'%(which_prf_grid))
    # Params for the spatial aspect of the model (possible pRFs)
    models = prf_utils.get_prf_models(which_grid=which_prf_grid)
    n_prfs = models.shape[0]

    if match_prf_trialcounts:
        prfs_use = prf_utils.get_prfs_use_decoding(which_prf_grid=which_prf_grid)
    else:
        prfs_use = np.ones((n_prfs,),dtype=bool)
    
    print('Using images/labels for subject:')
    print(subject)    
    print('debug=%s'%debug)
    sys.stdout.flush()

    if subject==999:
        # 999 is a code for the independent coco image set, using all images
        # (cross-validate within this set)
        image_inds = np.arange(10000)
    elif subject==998:
        image_inds = np.arange(50000)
    else:
        raise ValueError('subject must be 999 or 998')
        
    labels_all, discrim_type_list, unique_labels_each = initialize_fitting.load_highlevel_labels_each_prf(subject, \
                         which_prf_grid, image_inds=image_inds, models=models,verbose=False, debug=debug)
   
    print('Number of images using: %d'%labels_all.shape[0])
    sys.stdout.flush()
   
    print('doing decoding for:')
    print(discrim_type_list)
    
    n_trials = labels_all.shape[0]
    n_sem_axes = labels_all.shape[1] 
    
    # find the sub-sampled indices to use (computed offline)
    trial_masks = np.zeros((n_trials, n_prfs, n_sem_axes))       
    for aa in range(n_sem_axes):
        if match_prf_trialcounts:
            fn2load = os.path.join(default_paths.stim_labels_root, 'resampled_trial_orders', \
                       'S%d_balance_%s_for_decoding_matchprfs.npy'\
                           %(subject, discrim_type_list[aa]))
        else:
            fn2load = os.path.join(default_paths.stim_labels_root, 'resampled_trial_orders', \
                       'S%d_balance_%s_for_decoding.npy'\
                           %(subject, discrim_type_list[aa]))
        print('loading from %s'%fn2load)
        resamp_order = np.load(fn2load, allow_pickle=True).item()
        assert(np.all(resamp_order['image_order']==image_inds))
        trial_masks[:,:,aa] = resamp_order['trial_inds_balanced'][:,0,:]
        
    n_each_first = (np.sum(trial_masks[:,np.where(prfs_use)[0][0],:], axis=0)/2).astype(int)
    print(n_each_first)
    
    # create feature loaders
    feat_loaders, path_to_load = \
        default_feature_loaders.get_feature_loaders([subject], feature_type, which_prf_grid)
    feat_loader = feat_loaders[0]
    
    # Choose where to save results
    if debug:
        path_to_save = os.path.join(path_to_load, 'feature_decoding_DEBUG')
    else:
        path_to_save = os.path.join(path_to_load, 'feature_decoding')
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    if match_prf_trialcounts:
        fn2save = os.path.join(path_to_save, \
               'S%s_%s_LDA_all_grid%d_matchprfs.npy'%(subject, feature_type, which_prf_grid))
    else:
        fn2save = os.path.join(path_to_save, \
               'S%s_%s_LDA_all_grid%d.npy'%(subject, feature_type, which_prf_grid))
       
    print(fn2save)
    sys.stdout.flush()
    
    acc_each_prf = np.zeros((n_prfs, n_sem_axes), dtype=np.float32)
    dprime_each_prf = np.zeros((n_prfs, n_sem_axes), dtype=np.float32)
    
    n_trials_eachlabel = np.zeros((n_prfs, n_sem_axes), dtype=int)
    
    # first looping over pRFs
    for prf_model_index in range(n_prfs):

        if not prfs_use[prf_model_index]:
            continue
            
        if debug and prf_model_index>1:
            continue

        print('\nProcessing pRF %d of %d'%(prf_model_index, n_prfs))

        features_in_prf, _ = feat_loader.load(image_inds, prf_model_index)
       
        print('Size of features array for this image set and prf is:')
        print(features_in_prf.shape)
        assert(not np.any(np.isnan(features_in_prf)))
        # assert(not np.any(np.sum(features_in_prf, axis=0)==0))

        # then looping over axes to decode
        for aa in range(n_sem_axes):

            if prf_model_index==0:
                print('processing axis: %s'%discrim_type_list[aa])
                print('labels: ')
                print(unique_labels_each[aa])  
            
            labels = labels_all[:,aa,prf_model_index]
            
            # which trials to use here? evenly balanced for each label
            inds2use = trial_masks[:,prf_model_index, aa]==1
            
            X = features_in_prf[inds2use,:]
            y = labels[inds2use]
            
            # check the labels
            assert(not np.any(np.isnan(y)))
            un,n_each = np.unique(y, return_counts=True)
            print('label counts:')
            print(un, n_each)
            assert(n_each[0]==n_each[1])
            if match_prf_trialcounts:
                print(n_each, n_each_first)
                assert(np.all(n_each==n_each_first[aa]))
            n_trials_eachlabel[prf_model_index, aa] = n_each[0]
            
            tst_acc, tst_dprime = stats_utils.decode_lda(X, y, n_crossval_folds=10, debug=debug)
            
            acc_each_prf[prf_model_index,aa] = tst_acc
            dprime_each_prf[prf_model_index, aa] = tst_dprime
            
            # print to see how we are doing so far
            print('decode %s, prf %d: acc=%.2f, dprime=%.2f'%(discrim_type_list[aa], prf_model_index, tst_acc, tst_dprime))
            sys.stdout.flush()


    print('saving to %s'%fn2save)
    d = {'acc': acc_each_prf, \
          'dprime': dprime_each_prf, \
          'discrim_type_list': discrim_type_list, \
          'n_trials_eachlabel': n_trials_eachlabel}
    print(d.keys())
    print(d['n_trials_eachlabel'])
    np.save(fn2save, d, allow_pickle=True)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=1,
                    help="number of the subject, 1-8 or all")
    parser.add_argument("--feature_type", type=str,default='sketch_tokens',
                    help="what kind of features are we using?")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which prf grid to use")
    parser.add_argument("--match_prf_trialcounts", type=int,default=1,
                    help="use same number of trials for each pRF?")

    args = parser.parse_args()

    if args.debug:
        print('DEBUG MODE\n')
        
    sys.stdout.flush()
     
    run_decoding(subject=args.subject, feature_type=args.feature_type, debug=args.debug==1, \
                  which_prf_grid=args.which_prf_grid, match_prf_trialcounts=args.match_prf_trialcounts==1)
    