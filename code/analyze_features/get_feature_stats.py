import sys, os
import numpy as np
import argparse

from utils import default_paths, nsd_utils
from feature_extraction import default_feature_loaders
from model_fitting import initialize_fitting 

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def get_feature_stats(subject, feature_type, which_prf_grid=5, debug=False, layer_name=None):

    print('\nusing prf grid %d\n'%(which_prf_grid))
    # Params for the spatial aspect of the model (possible pRFs)
    models = initialize_fitting.get_prf_models(which_grid = which_prf_grid)    

    if subject=='all':       
        subjects = np.arange(1,9)
    else:
        subjects = [int(subject)]
    print('Using images/labels for subjects:')
    print(subjects)
    
    # get training set indices
    trninds_list = []
    for si, ss in enumerate(subjects):
        if ss==999:
            # 999 is a code for the set of images that are independent of NSD images, 
            # not shown to any participant.
            trninds = np.ones((10000,),dtype=bool)
        else:            
            # training / validation data always split the same way - shared 1000 inds are validation.
            subject_df = nsd_utils.get_subj_df(ss)
            trninds = np.array(subject_df['shared1000']==False)
        trninds_list.append(trninds)

    # create feature loaders
    feat_loaders, path_to_load = \
        default_feature_loaders.get_feature_loaders(subjects, feature_type, which_prf_grid)
   
    if debug:
        path_to_save = os.path.join(path_to_load, 'feature_stats_DEBUG')
    else:
        path_to_save = os.path.join(path_to_load, 'feature_stats')
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
        
    if subject=='all':
        substr = 'All_trn'
    else:
        substr = 'S%s'%subject

    fn2save_mean = os.path.join(path_to_save, '%s_%s_mean_grid%d.npy'\
                                 %(substr, feature_type, which_prf_grid))
    fn2save_var = os.path.join(path_to_save, '%s_%s_var_grid%d.npy'\
                                 %(substr, feature_type, which_prf_grid))
    fn2save_covar = os.path.join(path_to_save, '%s_%s_covar_grid%d.npy'\
                                 %(substr, feature_type, which_prf_grid)) 

    n_features = feat_loaders[0].max_features
    n_prfs = models.shape[0]
    all_mean = np.zeros((n_features, n_prfs), dtype=np.float32)
    all_var =  np.zeros((n_features, n_prfs), dtype=np.float32)
    all_covar =  np.zeros((n_features, n_features, n_prfs), dtype=np.float32)
   
    for prf_model_index in range(n_prfs):

        if debug and prf_model_index>1:
            continue

        print('Processing pRF %d of %d'%(prf_model_index, n_prfs))
        
        for si, feat_loader in enumerate(feat_loaders):

            # take training set trials only
            features_ss, def_ss = feat_loader.load(np.where(trninds_list[si])[0],prf_model_index);
 
            if si==0:
                features_in_prf_trn = features_ss
                feature_inds_defined = def_ss
                
            else:
                features_in_prf_trn = np.concatenate([features_in_prf_trn,features_ss], axis=0)
                assert(np.all(def_ss==feature_inds_defined))
  
        assert(len(feature_inds_defined)==n_features)
       
        print('Size of features array for this image set and prf is:')
        print(features_in_prf_trn.shape)
        
        # computing some basic stats for the features in this pRF
        all_mean[feature_inds_defined,prf_model_index] = np.mean(features_in_prf_trn, axis=0);
        all_var[feature_inds_defined,prf_model_index] = np.var(features_in_prf_trn, axis=0);
        cov_subset = all_covar[feature_inds_defined,:,prf_model_index]
        cov_subset[:,feature_inds_defined] = np.cov(features_in_prf_trn.T)
        all_covar[feature_inds_defined,:,prf_model_index] = cov_subset
        
        sys.stdout.flush()
               
    print('saving to %s\n'%fn2save_mean)
    np.save(fn2save_mean, all_mean)                     
    print('saving to %s\n'%fn2save_var)
    np.save(fn2save_var, all_var)    
    print('saving to %s\n'%fn2save_covar)
    np.save(fn2save_covar, all_covar)    
   
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=str, default='all',
                    help="number of the subject, 1-8, or all")
    parser.add_argument("--feature_type", type=str,default='sketch_tokens',
                    help="what kind of features are we using?")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which prf grid to use")
    parser.add_argument("--layer_name", type=str,default='',
                    help="which DNN layer to use (if clip or alexnet)")
   
    args = parser.parse_args()

    if args.debug:
        print('DEBUG MODE\n')

    get_feature_stats(subject=args.subject, feature_type=args.feature_type, debug=args.debug==1, which_prf_grid=args.which_prf_grid, layer_name=args.layer_name)
   