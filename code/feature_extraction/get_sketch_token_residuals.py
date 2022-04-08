import argparse
import numpy as np
import time
import os
import sys
import h5py
import pandas as pd

from utils import nsd_utils, default_paths
from model_fitting import initialize_fitting
from feature_extraction import fwrf_features

def get_sketch_token_gabor_residuals(subject, debug=False, which_prf_grid=5):
   
    # Params for the spatial aspect of the model (possible pRFs)
    models = initialize_fitting.get_prf_models(which_grid=which_prf_grid)  
    n_prfs = models.shape[0]
    
    feat_loader_st = fwrf_features.fwrf_feature_loader(subject=subject,\
                                which_prf_grid=which_prf_grid, \
                                feature_type='sketch_tokens',\
                                use_pca_feats = False)
    
   
    feat_loader_gabor = fwrf_features.fwrf_feature_loader(subject=subject,\
                                which_prf_grid=which_prf_grid, \
                                feature_type='gabor_solo',\
                                n_ori=12, n_sf=8, nonlin=True)
        
    subject_df = nsd_utils.get_subj_df(subject)
    trninds = np.array(subject_df['shared1000']==False)
    n_trials = len(trninds)
    inds_load = np.arange(n_trials)
    
    n_st_feats = feat_loader_st.max_features
    
    resid_feats = np.zeros((n_trials,n_st_feats,n_prfs),dtype=np.float32)
    r2_vals = np.zeros((n_st_feats,n_prfs),dtype=np.float32)

    for mm in range(n_prfs):
                
        if debug and mm>1:
            continue

        feat_st, _ = feat_loader_st.load(inds_load, prf_model_index=mm)
        feat_gabor, _ = feat_loader_gabor.load(inds_load, prf_model_index=mm)

        print('pRF %d of %d - size of features are'%(mm, n_prfs))
        print(feat_st.shape)
        print(feat_gabor.shape)
        print('number of training trials: %d'%(np.sum(trninds)))
        sys.stdout.flush()
        
        for st in range(n_st_feats):

            # for this sketch tokens feature, try to predict it as a sum of
            # the gabor feature activations on same trials.
            y = feat_st[:,st]

            X = np.concatenate([feat_gabor, np.ones((n_trials,1))], axis=1)
            # fitting regression wts on just training set
            linefit =  np.linalg.pinv(X[trninds,:]) @ y[trninds]          
            # make predictions for whole data set
            yhat = X @ linefit

            ssr = np.sum((y-yhat)**2)
            sst = np.sum((y-np.mean(y))**2)
            r2 = 1-ssr/sst

            r2_vals[st,mm] = r2

            resid_feats[:,st,mm] = y - yhat
            
    # save the residual features as a new file
    sketch_token_feat_path = default_paths.sketch_token_feat_path
    r2_df = pd.DataFrame(r2_vals)
    fn2save = os.path.join(sketch_token_feat_path, \
                       'S%d_gabor_regression_r2_grid%d.csv'%(subject, which_prf_grid))
    print('Writing r2 for sketch tokens-gabor regression to %s\n'%fn2save)    
    r2_df.to_csv(fn2save);
    
    fn2save = os.path.join(sketch_token_feat_path, \
                       'S%d_gabor_residuals_grid%d.h5py'%(subject, which_prf_grid))
    print('Writing residual sketch token features to %s\n'%fn2save)   
    t = time.time()
    with h5py.File(fn2save, 'w') as data_set:
        dset = data_set.create_dataset("features", np.shape(resid_feats), dtype=np.float32)
        data_set['/features'][:,:,:] = resid_feats
        data_set.close()  
    elapsed = time.time() - t    
    print('time elapsed to save: %.5f'%elapsed)
    
    
            
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=1,
                    help="number of the subject, 1-8")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which version of prf grid to use")
    
    args = parser.parse_args()
    
    get_sketch_token_gabor_residuals(subject = args.subject, debug = args.debug==1, \
                               which_prf_grid=args.which_prf_grid)
