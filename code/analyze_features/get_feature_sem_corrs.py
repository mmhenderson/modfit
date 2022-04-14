import sys, os
import numpy as np
import argparse

from utils import default_paths, nsd_utils, stats_utils
from feature_extraction import default_feature_loaders
from model_fitting import initialize_fitting 
  
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def get_feature_corrs(subject, feature_type, which_prf_grid=1, debug=False, layer_name=None):

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
        if ss==999:
            # 999 is a code for the set of images that are independent of NSD images, 
            # not shown to any participant.
            trninds = np.ones((10000,),dtype=bool)
        else:  
            # training / validation data always split the same way - shared 1000 inds are validation.
            subject_df = nsd_utils.get_subj_df(ss)
            trninds = np.array(subject_df['shared1000']==False)
        trninds_list.append(trninds)
        # working only with training data here.
        labels_all_ss, discrim_type_list_ss, unique_labels_each_ss = \
                            initialize_fitting.load_labels_each_prf(ss, \
                                    which_prf_grid, image_inds=np.where(trninds)[0], \
                                    models=models,verbose=False, debug=debug)
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
    
    fn2save_corrs = os.path.join(path_to_save, '%s_%s_semantic_corrs_grid%d.npy'\
                                 %(substr, feature_type, which_prf_grid))
    fn2save_discrim = os.path.join(path_to_save, '%s_%s_semantic_discrim_tstat_grid%d.npy'\
                                 %(substr, feature_type, which_prf_grid))
    fn2save_nsamp = os.path.join(path_to_save, '%s_%s_nsamp_grid%d.npy'\
                                 %(substr, feature_type, which_prf_grid))
    fn2save_partial_corrs = os.path.join(path_to_save, '%s_%s_semantic_partial_corrs_grid%d.npy'\
                                 %(substr, feature_type, which_prf_grid))
    fn2save_nsamp_partial = os.path.join(path_to_save, '%s_%s_nsamp_partial_corrs_grid%d.npy'\
                                 %(substr, feature_type, which_prf_grid))
    
    
    n_total_ims = labels_all.shape[0]
    print('Number of images using: %d'%n_total_ims)
    max_categ = np.max([len(un) for un in unique_labels_each])
    n_sem_axes = labels_all.shape[1]    
    n_features = feat_loaders[0].max_features
    n_prfs = models.shape[0]
    
    all_corrs = np.zeros((n_features, n_prfs, n_sem_axes), dtype=np.float32)
    all_discrim =  np.zeros((n_features, n_prfs, n_sem_axes), dtype=np.float32)
    n_samp_each_axis = np.zeros((n_prfs, n_sem_axes, max_categ), dtype=np.float32)
    
    axes_to_do_partial = [0,2,3]
    all_partial_corrs = np.zeros((n_features, n_prfs, len(axes_to_do_partial)), dtype=np.float32)
    n_samp_each_axis_partial = np.zeros((n_prfs, len(axes_to_do_partial), max_categ), dtype=np.float32)
    
    
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
        
        assert(features_in_prf_trn.shape[0]==n_total_ims)
        assert(len(feature_inds_defined)==n_features)
        n_features_defined = np.sum(feature_inds_defined)
        assert(features_in_prf_trn.shape[1]==n_features_defined)
        
        
        print('Size of features array for this image set and prf is:')
        print(features_in_prf_trn.shape)
        
        sys.stdout.flush()
         
        # first looking at each axis individually
        for aa in range(n_sem_axes):

            labels = labels_all[:,aa,prf_model_index]
            inds2use = ~np.isnan(labels)          
            unique_labels_actual = np.unique(labels[inds2use])
            
            if prf_model_index==0:
                print('processing axis: %s'%discrim_type_list[aa])
                print('labels: ')
                print(unique_labels_each[aa])
               
            if np.all(np.isin(unique_labels_each[aa], unique_labels_actual)):
                
                # groups is a list of arrays, one for each level of this semantic axis.
                # each array is [n_trials x n_features]
                groups = [features_in_prf_trn[(labels==ll) & inds2use,:] for ll in unique_labels_actual]

                if len(unique_labels_actual)==2:
                    # use t-statistic as a measure of discriminability
                    # larger pos value means feature[label==1] > feature[label==0]
                    # automatically goes down the 0th axis, so we get n_features tstats returned.
                    d = stats_utils.ttest_warn(groups[1],groups[0]).statistic
                    assert(len(d)==n_features_defined)
                    all_discrim[feature_inds_defined, prf_model_index, aa] = d;
                else:
                    # if more than 2 classes, computing an F statistic 
                    f = stats_utils.anova_oneway_warn(groups).statistic
                    assert(len(f)==n_features_defined)
                    all_discrim[feature_inds_defined, prf_model_index, aa] = f;
                    
                # also computing a correlation coefficient between each feature and the label
                # sign is consistent with t-statistic
                c_vals = stats_utils.numpy_corrcoef_warn(features_in_prf_trn[inds2use,:].T,labels[inds2use])
                c = c_vals[0:n_features_defined, n_features_defined]
                assert(len(c)==n_features_defined)
                all_corrs[feature_inds_defined, prf_model_index, aa] = c;

                for gi, gg in enumerate(groups):
                    n_samp_each_axis[prf_model_index, aa, gi] = len(gg)
                
            else:
                # if any labels are missing, skip this axis for this pRF
                all_discrim[:,prf_model_index,aa] = np.nan
                all_corrs[:,prf_model_index,aa] = np.nan
                n_samp_each_axis[prf_model_index,aa,:] = np.nan
                
        # Next doing the partial correlations, for some sub-set of the semantic axes.
        inds2use = np.sum(np.isnan(labels_all[:,axes_to_do_partial,prf_model_index]), axis=1)==0
        for ai, aa in enumerate(axes_to_do_partial):

            other_axes = np.array(axes_to_do_partial)[~np.isin(np.array(axes_to_do_partial), aa)]

            # going to compute information about the current axis of interest, while
            # partialling out the other axes. 
            labels_main_axis = labels_all[:,aa,prf_model_index]
            labels_other_axes = labels_all[:,other_axes,prf_model_index]
            
            unique_labels_actual = np.unique(labels_main_axis[inds2use])
            
            if np.all(np.isin(unique_labels_each[aa], unique_labels_actual)):
            
                for fi, ff in enumerate(np.where(feature_inds_defined)[0]):
                
                    partial_corr = stats_utils.compute_partial_corr(x=labels_main_axis[inds2use], \
                                                                y=features_in_prf_trn[inds2use,fi], \
                                                                c=labels_other_axes[inds2use,:])
                    all_partial_corrs[ff,prf_model_index,ai] = partial_corr
                
                for ui, uu in enumerate(unique_labels_actual):
                    n_samp_each_axis_partial[prf_model_index,ai,ui] = np.sum(labels_main_axis[inds2use]==uu)
                   
            else:
                # at least one category is missing for this voxel's pRF and this semantic axis.
                # skip it and put nans in the arrays.               
                all_partial_corrs[:,prf_model_index,ai] = np.nan
                n_samp_each_axis_partial[prf_model_index,ai,:] = np.nan
                
    print('saving to %s\n'%fn2save_corrs)
    np.save(fn2save_corrs, all_corrs)                     
    print('saving to %s\n'%fn2save_discrim)
    np.save(fn2save_discrim, all_discrim)     
    print('saving to %s\n'%fn2save_nsamp)
    np.save(fn2save_nsamp, n_samp_each_axis)
    
    print('saving to %s\n'%fn2save_partial_corrs)
    np.save(fn2save_partial_corrs, all_partial_corrs)     
    print('saving to %s\n'%fn2save_nsamp_partial)
    np.save(fn2save_nsamp_partial, n_samp_each_axis_partial)   

    
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

    get_feature_corrs(subject=args.subject, feature_type=args.feature_type, debug=args.debug==1, which_prf_grid=args.which_prf_grid, layer_name=args.layer_name)
   