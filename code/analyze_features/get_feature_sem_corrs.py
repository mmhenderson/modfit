import sys, os
import numpy as np
import argparse

from utils import default_paths, stats_utils, label_utils, prf_utils
from feature_extraction import default_feature_loaders
  
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def get_feature_corrs(subject, feature_type, which_prf_grid=1, layer_name=None):

    print('\nusing prf grid %d\n'%(which_prf_grid))
    # Params for the spatial aspect of the model (possible pRFs)
    models = prf_utils.get_prf_models(which_grid = which_prf_grid)    

    print(subject)
    subject = int(subject)
    print(subject)
    if subject==999:
        # 999 is a code for the independent coco image set, using all images
        # (cross-validate within this set)
        image_inds = np.arange(10000)
    elif subject==998:
        image_inds = np.arange(50000)
    else:
        raise ValueError('subject must be 999 or 998')
        
    substr = 'S%s'%subject
    print('Using images/labels for subject %d, %s, %d images'%(subject, substr, len(image_inds))) 
    
    labels_binary, axis_names, unique_labels_each = label_utils.load_highlevel_labels_each_prf(subject, \
                         which_prf_grid, image_inds=image_inds, models=models)
   
    axes_use = [1,2,3,4,5]
    labels_all = labels_binary[:,axes_use,:]
    axis_names = np.array(axis_names)[axes_use]
    print('using axes:')
    print(axis_names)
    unique_labels_each = np.array(unique_labels_each)[axes_use]
    
    # create feature loaders
    feat_loaders, path_to_load = \
        default_feature_loaders.get_feature_loaders([subject], feature_type, which_prf_grid)
    feat_loader = feat_loaders[0]
    
    # decide where to save results
    path_to_save = os.path.join(path_to_load, 'feature_stats')
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    
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
    max_categ = 2
    n_sem_axes = labels_all.shape[1]    
    n_features = feat_loaders[0].max_features
    n_prfs = models.shape[0]
    
    all_corrs = np.zeros((n_features, n_prfs, n_sem_axes), dtype=np.float32)
    all_discrim =  np.zeros((n_features, n_prfs, n_sem_axes), dtype=np.float32)
    n_samp_each_axis = np.zeros((n_prfs, n_sem_axes, max_categ), dtype=np.float32)
    
    all_partial_corrs = np.zeros((n_features, n_prfs, n_sem_axes), dtype=np.float32)
    n_samp_each_axis_partial = np.zeros((n_prfs, n_sem_axes, max_categ), dtype=np.float32)
    
    
    for prf_model_index in range(n_prfs):

        print('Processing pRF %d of %d'%(prf_model_index, n_prfs))
        features_in_prf, _ = feat_loader.load(image_inds, prf_model_index)
       
        print('Size of features array for this image set and prf is:')
        print(features_in_prf.shape)
        assert(not np.any(np.isnan(features_in_prf)))

        sys.stdout.flush()
         
        # first looking at each axis individually
        for aa in range(n_sem_axes):

            labels = labels_all[:,aa,prf_model_index]
            inds2use = ~np.isnan(labels)          
            unique_labels_actual = np.unique(labels[inds2use])
            
            if prf_model_index==0:
                print('processing axis: %s'%axis_names[aa])
                print('labels: ')
                print(unique_labels_each[aa])
               
            if np.all(np.isin(unique_labels_each[aa], unique_labels_actual)):
                
                # groups is a list of arrays, one for each level of this semantic axis.
                # each array is [n_trials x n_features]
                groups = [features_in_prf[(labels==ll) & inds2use,:] for ll in unique_labels_actual]

                # use t-statistic as a measure of discriminability
                # larger pos value means feature[label==1] > feature[label==0]
                # automatically goes down the 0th axis, so we get n_features tstats returned.
                d = stats_utils.ttest_warn(groups[1],groups[0]).statistic
                
                all_discrim[:, prf_model_index, aa] = d;
                  
                # also computing a correlation coefficient between each feature and the label
                # sign is consistent with t-statistic
                c_vals = stats_utils.numpy_corrcoef_warn(features_in_prf[inds2use,:].T,labels[inds2use])
                c = c_vals[0:n_features, n_features]
                all_corrs[:, prf_model_index, aa] = c;

                for gi, gg in enumerate(groups):
                    n_samp_each_axis[prf_model_index, aa, gi] = len(gg)
                
            else:
                print('missing label, entering a nan for corr')
                # if any labels are missing, skip this axis for this pRF
                all_discrim[:,prf_model_index,aa] = np.nan
                all_corrs[:,prf_model_index,aa] = np.nan
                n_samp_each_axis[prf_model_index,aa,:] = np.nan
                
        # Next doing the partial correlations
        # only use the images where every single label is defined
        inds2use = np.sum(np.isnan(labels_all[:,:,prf_model_index]), axis=1)==0
        print('for partial correlations, there are %d trials available'%np.sum(inds2use))
        
        for aa in range(n_sem_axes):
            
            other_axes = np.arange(n_sem_axes)[~np.isin(np.arange(n_sem_axes), aa)]

            # going to compute information about the current axis of interest, while
            # partialling out the other axes. 
            labels_main_axis = labels_all[:,aa,prf_model_index]
            labels_other_axes = labels_all[:,other_axes,prf_model_index]
            
            unique_labels_actual = np.unique(labels_main_axis[inds2use])
            
            if np.all(np.isin(unique_labels_each[aa], unique_labels_actual)):
            
                for ff in range(n_features):
                
                    partial_corr = stats_utils.compute_partial_corr(x=labels_main_axis[inds2use], \
                                                                y=features_in_prf[inds2use,ff], \
                                                                c=labels_other_axes[inds2use,:])
                    all_partial_corrs[ff,prf_model_index,aa] = partial_corr
                
                for ui, uu in enumerate(unique_labels_actual):
                    n_samp_each_axis_partial[prf_model_index,aa,ui] = np.sum(labels_main_axis[inds2use]==uu)
                   
            else:
                # at least one category is missing for this voxel's pRF and this semantic axis.
                # skip it and put nans in the arrays.    
                print('missing label, entering a nan for partial corr')
                all_partial_corrs[:,prf_model_index,aa] = np.nan
                n_samp_each_axis_partial[prf_model_index,aa,:] = np.nan
                
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
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which prf grid to use")
    parser.add_argument("--layer_name", type=str,default='',
                    help="which DNN layer to use (if clip or alexnet)")
   
    args = parser.parse_args()

    get_feature_corrs(subject=args.subject, feature_type=args.feature_type, which_prf_grid=args.which_prf_grid, layer_name=args.layer_name)
   