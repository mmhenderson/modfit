import sys, os
import numpy as np
import time, h5py
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import argparse
import pandas as pd   

from utils import default_paths, stats_utils, floc_utils
from model_fitting import initialize_fitting 
from feature_extraction import default_feature_loaders_visvssem

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def run_decoding(image_set='floc', 
                 feature_type='gabor_solo', \
                 which_prf_grid=1, debug=False, \
                 layer_name=None):
      
    models = initialize_fitting.get_prf_models(which_grid = which_prf_grid)    
    n_prfs = len(models)
    print('feature_type=%s, which_prf_grid=%d'%(feature_type, which_prf_grid))
    print('debug=%s'%debug)
    sys.stdout.flush()

    image_inds_use = floc_utils.load_balanced_floc_set()
    
    labels_file = os.path.join(default_paths.floc_image_root,'floc_image_labels.csv')
    labels = pd.read_csv(labels_file)

    domain_labels = np.array([labels['domain']==domain for domain in floc_utils.domains]).T.astype(int)
    domain_labels = domain_labels[image_inds_use,:]
    n_trials = domain_labels.shape[0]
    n_sem_axes = domain_labels.shape[1] 
    
    # to make decoding fair, create a set of labels that downsamples larger class
    np.random.seed(239843)
    min_each = np.sum(domain_labels, axis=0)[0]
    downsample_inds = np.zeros(np.shape(domain_labels),dtype=bool)
    for dd in range(n_sem_axes):
        for uu in np.unique(domain_labels[:,dd]):
            inds = np.where(domain_labels[:,dd]==uu)[0]
            if len(inds)>min_each:
                inds_ds = np.random.choice(inds, min_each, replace=False)
            else:
                inds_ds = inds
            downsample_inds[inds_ds,dd] = True
 
    discrim_type_list = ['%s > other domains'%domain for domain in floc_utils.domains]
    domains = floc_utils.domains
    
    print('Number of images using: %d'%domain_labels.shape[0])
    print('doing decoding for:')
    print(discrim_type_list)
    sys.stdout.flush()

    
    # create feature loaders
    feat_loader, path_to_load = \
        default_feature_loaders_visvssem.get_feature_loaders(image_set, feature_type, which_prf_grid)
    
    # Choose where to save results
    if debug:
        path_to_save = os.path.join(path_to_load, 'feature_decoding_floc_DEBUG')
    else:
        path_to_save = os.path.join(path_to_load, 'feature_decoding_floc')
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    fn2save = os.path.join(path_to_save, \
               '%s_%s_LDA_all_grid%d.npy'%(image_set, feature_type, which_prf_grid))
       
    print(fn2save)
    sys.stdout.flush()
    
    acc_each_prf = np.zeros((n_prfs, n_sem_axes), dtype=np.float32)
    dprime_each_prf = np.zeros((n_prfs, n_sem_axes), dtype=np.float32)
    
    pairwise_acc_each_prf = np.zeros((n_prfs, n_sem_axes, n_sem_axes), dtype=np.float32)
    pairwise_dprime_each_prf = np.zeros((n_prfs, n_sem_axes, n_sem_axes), dtype=np.float32)
    
    # first looping over pRFs
    for prf_model_index in range(n_prfs):

        if debug and prf_model_index>1:
            continue

        print('\nProcessing pRF %d of %d'%(prf_model_index, n_prfs))

        image_inds = np.where(image_inds_use)[0]
        print(len(image_inds))
        features_in_prf, _ = feat_loader.load(image_inds, prf_model_index)
       
        print('Size of features array for this image set and prf is:')
        print(features_in_prf.shape)
        assert(not np.any(np.isnan(features_in_prf)))
        # assert(not np.any(np.sum(features_in_prf, axis=0)==0))

        # then looping over axes to decode
        for aa in range(n_sem_axes):

            X = features_in_prf[downsample_inds[:,aa],:]
            y = domain_labels[downsample_inds[:,aa],aa]
            un, counts = np.unique(y, return_counts=True)
            assert(counts[0]==counts[1]) # double check balancing
            
            if prf_model_index==0:
                print('processing axis: %s'%discrim_type_list[aa])
                print('labels: ')
                print(np.unique(y))
                print('size of X, y: %s, %s'%(X.shape, y.shape))
            
            tst_acc, tst_dprime = stats_utils.decode_lda(X, y, n_crossval_folds=10, debug=debug)
            
            acc_each_prf[prf_model_index,aa] = tst_acc
            dprime_each_prf[prf_model_index, aa] = tst_dprime
            
            # print to see how we are doing so far
            print('decode %s, prf %d: acc=%.2f, dprime=%.2f'%(discrim_type_list[aa], 
                                                              prf_model_index, tst_acc, tst_dprime))
            sys.stdout.flush()
            
            for aa2 in np.arange(aa+1, n_sem_axes):
                
                inds_use_pairwise = domain_labels[:,aa] | domain_labels[:,aa2]
                X = features_in_prf[inds_use_pairwise==1,:]
                y = domain_labels[inds_use_pairwise==1,aa]
                y2 = domain_labels[inds_use_pairwise==1,aa2]
                assert(np.all(y==(1-y2))) # should be opposites
                un, counts = np.unique(y, return_counts=True)
                assert(counts[0]==counts[1]) # double check balancing
               
                if prf_model_index==0:
                    print('processing axis: %s vs %s'%(domains[aa], domains[aa2]))
                    print('labels: ')
                    print(np.unique(y))
                    print('size of X, y: %s, %s'%(X.shape, y.shape))

                tst_acc, tst_dprime = stats_utils.decode_lda(X, y, n_crossval_folds=10, debug=debug)

                pairwise_acc_each_prf[prf_model_index, aa,aa2] = tst_acc
                pairwise_dprime_each_prf[prf_model_index, aa,aa2] = tst_dprime
                pairwise_acc_each_prf[prf_model_index, aa2,aa] = tst_acc
                pairwise_dprime_each_prf[prf_model_index, aa2,aa] = tst_dprime

                # print to see how we are doing so far
                print('decode %s vs %s, prf %d: acc=%.2f, dprime=%.2f'%(domains[aa], domains[aa2], 
                                                                        prf_model_index, tst_acc, tst_dprime))
                sys.stdout.flush()


    print('saving to %s'%fn2save)
    np.save(fn2save, {'acc': acc_each_prf, \
                      'dprime': dprime_each_prf, 
                      'pairwise_acc': pairwise_acc_each_prf, \
                      'pairwise_dprime': pairwise_dprime_each_prf, 
                      'discrim_type_list': discrim_type_list})



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--image_set", type=str,default='floc',
                    help="which image set?")
    parser.add_argument("--feature_type", type=str,default='sketch_tokens',
                    help="what kind of features are we using?")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which prf grid to use")
    
    args = parser.parse_args()

    if args.debug:
        print('DEBUG MODE\n')
        
    sys.stdout.flush()
     
    run_decoding(image_set=args.image_set, feature_type=args.feature_type, debug=args.debug==1, \
                  which_prf_grid=args.which_prf_grid)
    