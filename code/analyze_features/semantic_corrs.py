import sys, os
import numpy as np
from utils import default_paths, nsd_utils, stats_utils
from model_fitting import initialize_fitting 
import argparse
 
def get_sem_corrs(which_prf_grid=1, debug=False):

    print('\nusing prf grid %d\n'%(which_prf_grid))
    # Params for the spatial aspect of the model (possible pRFs)
    models = initialize_fitting.get_prf_models(which_grid = which_prf_grid)    
     
    subjects = np.arange(1,9)
    
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
    n_sem_axes_total = labels_all.shape[1]
    
    path_to_save = default_paths.stim_labels_root

    fn2save_corrs = os.path.join(path_to_save, 'Semantic_to_semantic_corrs_grid%d.npy'%(which_prf_grid))
    fn2save_nsamp = os.path.join(path_to_save, 'Semantic_to_semantic_nsamp_grid%d.npy'%(which_prf_grid))
    
    n_prfs = models.shape[0]
    all_corrs = np.zeros((n_sem_axes_total, n_sem_axes_total, n_prfs), dtype=np.float32)
    all_nsamp = np.zeros((n_sem_axes_total, n_sem_axes_total, n_prfs, 4), dtype=np.float32)
    
    for prf_model_index in range(n_prfs):

        if debug and prf_model_index>1:
            continue

        print('Processing pRF %d of %d'%(prf_model_index, n_prfs))
        
        sys.stdout.flush()
                                 
        for aa1 in range(n_sem_axes_total):
            
            labels1 = labels_all[:,aa1,prf_model_index]
            
            for aa2 in range(n_sem_axes_total):
                
                if aa2==aa1:
                    continue
                    
                labels2 = labels_all[:,aa2,prf_model_index]

                inds2use = ~np.isnan(labels1) & ~np.isnan(labels2)  
                
                n1 = np.sum((labels1[inds2use]==0) & (labels2[inds2use]==0))
                n2 = np.sum((labels1[inds2use]==0) & (labels2[inds2use]==1))
                n3 = np.sum((labels1[inds2use]==1) & (labels2[inds2use]==0))
                n4 = np.sum((labels1[inds2use]==1) & (labels2[inds2use]==1))
                nsamp = np.array([n1,n2,n3,n4])
                all_nsamp[aa1, aa2, prf_model_index, :] = nsamp

                if prf_model_index==0:
                    print('processing %s vs. %s'%(discrim_type_list[aa1],discrim_type_list[aa2]))

                if (len(np.unique(labels1[inds2use]))==2) and (len(np.unique(labels2[inds2use]))==2):

                    all_corrs[aa1, aa2, prf_model_index ] = stats_utils.numpy_corrcoef_warn(\
                                                            labels1[inds2use],labels2[inds2use])[0,1]
                else:
                    all_corrs[aa1, aa2, prf_model_index ] =np.nan

    print('saving to %s\n'%fn2save_corrs)
    np.save(fn2save_corrs, all_corrs)                     
    print('saving to %s\n'%fn2save_nsamp)
    np.save(fn2save_nsamp, all_nsamp)    

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which prf grid to use")

    args = parser.parse_args()

    if args.debug:
        print('DEBUG MODE\n')

    get_sem_corrs(debug=args.debug==1, which_prf_grid=args.which_prf_grid)
    