import sys, os
import numpy as np
import time, h5py
import scipy.stats
from utils import default_paths, nsd_utils, stats_utils
from feature_extraction import fwrf_features
from model_fitting import initialize_fitting 
import argparse
import pandas as pd
   
def get_feature_discrim(subject, feature_type, which_prf_grid=1, debug=False, layer_name=None):

    print('\nusing prf grid %d\n'%(which_prf_grid))
    # Params for the spatial aspect of the model (possible pRFs)
    models = initialize_fitting.get_prf_models(which_grid = which_prf_grid)    

    if subject=='all':       
        subjects = np.arange(1,9)
    else:
        subjects = [int(subject)]
    print('Using images/labels for subjects:')
    print(subjects)
    
    # First gather all semantic labels
    trninds_list = []
    for si, ss in enumerate(subjects):
        # training / validation data always split the same way - shared 1000 inds are validation.
        subject_df = nsd_utils.get_subj_df(ss)
        valinds = np.array(subject_df['shared1000'])
        trninds = np.array(subject_df['shared1000']==False)
        trninds_list.append(trninds)
        # working only with training data here.
        labels_all_ss, discrim_type_list_ss, unique_labels_each_ss = initialize_fitting.load_labels_each_prf(ss, \
                             which_prf_grid, image_inds=np.where(trninds)[0], models=models,verbose=False, debug=debug)
        if si==0:
            labels_all = labels_all_ss
            discrim_type_list = discrim_type_list_ss
            unique_labels_each = unique_labels_each_ss
        else:
            labels_all = np.concatenate([labels_all, labels_all_ss], axis=0)
            # check that columns are same for all subs
            assert(np.all(np.array(discrim_type_list)==np.array(discrim_type_list_ss)))
            assert(np.all([np.all(unique_labels_each[ii]==unique_labels_each_ss[ii]) \
                           for ii in range(len(unique_labels_each))]))
            
    # all categories must be binary.
    assert(np.all([len(un)==2 for un in unique_labels_each]))
    
    print('Number of images using: %d'%labels_all.shape[0])
    n_sem_axes = labels_all.shape[1]
        
    # make feature loaders to get visual features
    if feature_type=='gabor_solo':
        path_to_load = default_paths.gabor_texture_feat_path
        feat_loaders = [fwrf_features.fwrf_feature_loader(subject=ss,\
                                                        which_prf_grid=which_prf_grid,\
                                                        feature_type='gabor_solo', \
                                                        n_ori=12, n_sf=8, nonlin=True) for ss in subjects]
    elif 'pyramid_texture' in feature_type:
        path_to_load = default_paths.pyramid_texture_feat_path
        if feature_type=='pyramid_texture_ll': 
            include_ll=True
            include_hl=False
            use_pca_feats_hl = False
        elif feature_type=='pyramid_texture_hl':
            include_ll=False
            include_hl=True
            use_pca_feats_hl = False
        elif feature_type=='pyramid_texture_hl_pca':
            assert(len(subjects)==1) # since these features are pca-ed within subject, can't concatenate.
            include_ll=False
            include_hl=True
            use_pca_feats_hl = True
        feat_loaders = [fwrf_features.fwrf_feature_loader(subject=ss,\
                                                        which_prf_grid=which_prf_grid, \
                                                        feature_type='pyramid_texture',\
                                                        n_ori=4, n_sf=4,\
                                                        include_ll=include_ll, include_hl=include_hl,\
                                                        use_pca_feats_hl = use_pca_feats_hl) for ss in subjects]       
 
    elif feature_type=='sketch_tokens':
        path_to_load = default_paths.sketch_token_feat_path
        feat_loaders = [fwrf_features.fwrf_feature_loader(subject=ss,\
                                                        which_prf_grid=which_prf_grid, \
                                                        feature_type='sketch_tokens',\
                                                        use_pca_feats = False) for ss in subjects]

    elif feature_type=='alexnet':
        assert(len(subjects)==1) # since these features are pca-ed within subject, can't concatenate.
        path_to_load = default_paths.alexnet_feat_path
        if layer_name is None or layer_name=='':
            layer_name='Conv5_ReLU'
        feat_loaders = [fwrf_features.fwrf_feature_loader(subject=ss,\
                                                        which_prf_grid=which_prf_grid, \
                                                        feature_type='alexnet',layer_name=layer_name,\
                                                        use_pca_feats = True, padding_mode = 'reflect') \
                                                        for ss in subjects]

    elif feature_type=='clip':
        assert(len(subjects)==1) # since these features are pca-ed within subject, can't concatenate.
        path_to_load = default_paths.clip_feat_path
        if layer_name is None or layer_name=='':
            layer_name='block15'
        feat_loaders = [fwrf_features.fwrf_feature_loader(subject=ss,\
                                                        which_prf_grid=which_prf_grid, \
                                                        feature_type='clip',layer_name=layer_name,\
                                                        model_architecture='RN50',use_pca_feats=True) \
                                                        for ss in subjects]

    else:
        raise RuntimeError('feature type %s not recognized'%feature_type)
    
    if debug:
        path_to_save = os.path.join(path_to_load, 'feature_stats_DEBUG')
    else:
        path_to_save = os.path.join(path_to_load, 'feature_stats')
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    if subject=='all':
        fn2save_corrs = os.path.join(path_to_save, 'All_trn_%s_semantic_corrs_grid%d.npy'\
                                     %(feature_type, which_prf_grid))
        fn2save_discrim = os.path.join(path_to_save, 'All_trn_%s_semantic_discrim_tstat_grid%d.npy'\
                                     %(feature_type, which_prf_grid))
        fn2save_mean = os.path.join(path_to_save, 'All_trn_%s_mean_grid%d.npy'\
                                     %(feature_type, which_prf_grid))
        fn2save_var = os.path.join(path_to_save, 'All_trn_%s_var_grid%d.npy'\
                                     %(feature_type, which_prf_grid))
        fn2save_covar = os.path.join(path_to_save, 'All_trn_%s_covar_grid%d.npy'\
                                     %(feature_type, which_prf_grid)) 
        fn2save_nsamp = os.path.join(path_to_save, 'All_trn_%s_nsamp_grid%d.npy'\
                                     %(feature_type, which_prf_grid))
    else:        
        fn2save_corrs = os.path.join(path_to_save, 'S%s_%s_semantic_corrs_grid%d.npy'\
                                     %(subject, feature_type, which_prf_grid))
        fn2save_discrim = os.path.join(path_to_save, 'S%s_%s_semantic_discrim_tstat_grid%d.npy'\
                                     %(subject, feature_type, which_prf_grid))
        fn2save_mean = os.path.join(path_to_save, 'S%s_%s_mean_grid%d.npy'\
                                     %(subject,feature_type, which_prf_grid))
        fn2save_var = os.path.join(path_to_save, 'S%s_%s_var_grid%d.npy'\
                                     %(subject, feature_type, which_prf_grid))
        fn2save_covar = os.path.join(path_to_save, 'S%s_%s_covar_grid%d.npy'\
                                     %(subject, feature_type, which_prf_grid)) 
        fn2save_nsamp = os.path.join(path_to_save, 'S%s_%s_nsamp_grid%d.npy'\
                                     %(subject, feature_type, which_prf_grid))
    
    n_features = feat_loaders[0].max_features
    n_prfs = models.shape[0]
    all_corrs = np.zeros((n_features, n_prfs, n_sem_axes), dtype=np.float32)
    all_discrim =  np.zeros((n_features, n_prfs, n_sem_axes), dtype=np.float32)
    all_mean = np.zeros((n_features, n_prfs), dtype=np.float32)
    all_var =  np.zeros((n_features, n_prfs), dtype=np.float32)
    all_covar =  np.zeros((n_features, n_features, n_prfs), dtype=np.float32)
    n_samp_each_axis = np.zeros((n_prfs, n_sem_axes, 2), dtype=np.float32)
    
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
                feat_defined = np.where(def_ss)[0]
                
            else:
                features_in_prf_trn = np.concatenate([features_in_prf_trn,features_ss], axis=0)
                assert(np.all(def_ss==feature_inds_defined))
        
        assert(features_in_prf_trn.shape[0]==labels_all.shape[0])
        assert(features_in_prf_trn.shape[1]==len(feat_defined))
        assert(len(feature_inds_defined)==n_features)
       
        print('Size of features array for this image set and prf is:')
        print(features_in_prf_trn.shape)
        
        # computing some basic stats for the features in this pRF
        all_mean[feature_inds_defined,prf_model_index] = np.mean(features_in_prf_trn, axis=0);
        all_var[feature_inds_defined,prf_model_index] = np.var(features_in_prf_trn, axis=0);
        all_covar[feature_inds_defined,:,prf_model_index][:,feature_inds_defined] = np.cov(features_in_prf_trn.T)
        
        sys.stdout.flush()
                                 
        for aa in range(n_sem_axes):

            labels = labels_all[:,aa,prf_model_index]
            inds2use = ~np.isnan(labels)          
            unique_labels_actual = np.unique(labels[inds2use])
            
            if prf_model_index==0:
                print('processing axis: %s'%discrim_type_list[aa])
                print('labels: ')
                print(unique_labels_each[aa])
               
            if np.all(np.isin(unique_labels_each[aa], unique_labels_actual)):
                
                group_inds = [((labels==ll) & inds2use) for ll in unique_labels_actual]  
                
                for fi, ff in enumerate(feat_defined):
                  
                    groups = [features_in_prf_trn[gi,fi] for gi in group_inds]
                    # use t-statistic as a measure of discriminability
                    # larger value means resp[label==1] > resp[label==0]
                    all_discrim[ff, prf_model_index, aa] = stats_utils.ttest_warn(groups[1], groups[0]).statistic
                    # also computing a correlation coefficient between semantic label/voxel response
                    # sign is consistent with t-statistic
                    all_corrs[ff,prf_model_index,aa] = stats_utils.numpy_corrcoef_warn(\
                                                        features_in_prf_trn[inds2use,fi],labels[inds2use])[0,1]
                n_samp_each_axis[prf_model_index,aa,0] = np.sum(group_inds[0])
                n_samp_each_axis[prf_model_index,aa,1] = np.sum(group_inds[1])
                
            else:
                # if any labels are missing, skip this axis for this pRF
                print('missing some labels for axis %d'%aa)
                print('expected labels')
                print(unique_labels_each[aa])
                print('actual labels')
                print(unique_labels_actual)
                print('nans for model %d, axis %d, because some labels were missing'\
                          %(prf_model_index, aa))
                all_discrim[:, prf_model_index, aa] = np.nan
                all_corrs[:,prf_model_index,aa] = np.nan
                n_samp_each_axis[prf_model_index,aa,:] = np.nan
                
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
    print('saving to %s\n'%fn2save_nsamp)
    np.save(fn2save_nsamp, n_samp_each_axis)    

    
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

    get_feature_discrim(subject=args.subject, feature_type=args.feature_type, debug=args.debug==1, which_prf_grid=args.which_prf_grid, layer_name=args.layer_name)
   