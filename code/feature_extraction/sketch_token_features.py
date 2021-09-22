import numpy as np
import sys, os
import torch
import time
import h5py
import torch.nn as nn
from sklearn import decomposition

from utils import prf_utils, torch_utils, texture_utils, default_paths
sketch_token_feat_path = default_paths.sketch_token_feat_path

class sketch_token_feature_extractor(nn.Module):
    
    def __init__(self, subject, device, map_resolution=227, aperture = 1.0, n_prf_sd_out = 2, \
                 batch_size=100, mult_patch_by_prf=True, do_avg_pool = True, \
                 do_pca = True, min_pct_var = 99, max_pc_to_retain = 100):
        
        super(sketch_token_feature_extractor, self).__init__()
        
        self.subject = subject
        
        self.features_file = os.path.join(sketch_token_feat_path, 'S%d_features_each_prf.h5py'%(subject))
        if not os.path.exists(self.features_file):
            raise RuntimeError('Looking at %s for precomputed features, not found.'%self.features_file)
        with h5py.File(self.features_file, 'r') as data_set:
            ds_size = np.shape(data_set['/features'])
        self.n_features = ds_size[1]
        self.device = device
          
        self.map_resolution = map_resolution
        self.aperture = aperture
        self.n_prf_sd_out = n_prf_sd_out
        self.batch_size = batch_size
        self.mult_patch_by_prf = mult_patch_by_prf
        self.do_avg_pool = do_avg_pool # else max pool
        
        self.do_pca = do_pca
        if self.do_pca:
            self.min_pct_var = min_pct_var
            self.max_pc_to_retain = np.min([self.n_features, max_pc_to_retain])
        else:
            self.min_pct_var = None
            self.max_pc_to_retain = None  
            
        self.do_varpart=False # only one set of features in this model for now, not doing variance partition
        self.features_each_prf = None
        
    def init_for_fitting(self, image_size, models, dtype):

        """
        Additional initialization operations which can only be done once we know image size and
        desired set of candidate prfs.
        """
        
        print('Initializing for fitting')
        n_prfs = len(models)
        n_feat_each_prf = self.n_features * np.ones(shape=(n_prfs,),dtype=int)      
        self.n_feat_each_prf = n_feat_each_prf
        
        if self.do_pca:
            
            print('Initializing arrays for PCA params')
            # will need to save pca parameters to reproduce it during validation stage
            # max pc to retain is just to save space, otherwise the "pca_wts" variable becomes huge  
            self.max_features = self.max_pc_to_retain
            self.pca_wts = [np.zeros(shape=(self.max_pc_to_retain, n_feat_each_prf[mm]), dtype=dtype) for mm in range(n_prfs)] 
            self.pca_pre_z_mean = [np.zeros(shape=(n_feat_each_prf[mm],), dtype=dtype) for mm in range(n_prfs)]
            self.pca_pre_z_std = [np.zeros(shape=(n_feat_each_prf[mm],), dtype=dtype) for mm in range(n_prfs)]
            self.pca_pre_mean = [np.zeros(shape=(n_feat_each_prf[mm],), dtype=dtype) for mm in range(n_prfs)]
            self.pct_var_expl = np.zeros(shape=(self.max_pc_to_retain, n_prfs), dtype=dtype)
            self.n_comp_needed = np.full(shape=(n_prfs), fill_value=-1, dtype=int)

        else:
            self.max_features = np.max(n_feat_each_prf)
       
        self.clear_big_features()
        
    def get_partial_versions(self):

        if not hasattr(self, 'max_features'):
            raise RuntimeError('need to run init_for_fitting first')
           
        partial_version_names = ['full_model']
        masks = np.ones([1,self.max_features])

        return masks, partial_version_names

    def load_precomputed_features(self, image_inds):
        
        print('Loading pre-computed features from %s'%self.features_file)
        t = time.time()
        with h5py.File(self.features_file, 'r') as data_set:
            values = np.copy(data_set['/features'])
            data_set.close() 
        elapsed = time.time() - t
        print('Took %.5f seconds to load file'%elapsed)
        
        self.features_each_prf = values[image_inds,:,:]
        
        print('Size of features array for this image set is:')
        print(self.features_each_prf.shape)
        
    def clear_big_features(self):
        
        print('Clearing features from memory')
        self.features_each_prf = None 
    
    def forward(self, image_inds, prf_params, prf_model_index, fitting_mode = True):
        
        if self.features_each_prf is None:
            self.load_precomputed_features(image_inds)
        else:
            assert(self.features_each_prf.shape[0]==len(image_inds))
            
        # Taking the features for the desired prf model
        features = self.features_each_prf[:,:,prf_model_index]
        
        if self.do_pca:    
            features = self.reduce_pca(features, prf_model_index, fitting_mode)

        print('Final size of feature matrix is:')
        print(features.shape)
        
        features = torch_utils._to_torch(features, self.device)
        
        feature_inds_defined = np.zeros((self.max_features,), dtype=bool)
        feature_inds_defined[0:features.shape[1]] = 1
            
        return features, feature_inds_defined
     
        
    def reduce_pca(self, features, prf_model_index, fitting_mode=True):
        
        if torch.is_tensor(features):
            features = features.detach().cpu().numpy()
            was_tensor=True
        else:
            was_tensor=False
            
        n_trials = features.shape[0]
        n_features_actual = features.shape[1]
        assert(n_features_actual == self.n_feat_each_prf[prf_model_index])
        print('Preparing for PCA: original dims of features:')
        print(features.shape)
        
        if fitting_mode:
            
            # Going to perform pca on the raw features
            # First make sure it hasn't been done yet!
            assert(self.n_comp_needed[prf_model_index]==-1) 
            print('Running PCA...')
            pca = decomposition.PCA(n_components = np.min([np.min([self.max_pc_to_retain, n_features_actual]), n_trials]), copy=False)
            # for this model, need to normalize the columns otherwise the last one dominates...
            features_m = np.mean(features, axis=0, keepdims=True) #[:trn_size]
            features_s = np.std(features, axis=0, keepdims=True) + 1e-6          
            features -= features_m
            features /= features_s 
            self.pca_pre_z_mean[prf_model_index][0:n_features_actual] = features_m
            self.pca_pre_z_std[prf_model_index][0:n_features_actual] = features_s
            
            # Perform PCA to decorrelate feats and reduce dimensionality
            scores = pca.fit_transform(features)           
            features = None            
            wts = pca.components_
            ev = pca.explained_variance_
            ev = ev/np.sum(ev)*100
            # wts/components goes [ncomponents x nfeatures]. 
            # nfeatures is always actual number of raw features
            # ncomponents is min(ntrials, nfeatures)
            # to save space, only going to save up to some max number of components.
            n_components_actual = np.min([wts.shape[0], self.max_pc_to_retain])
            # save a record of the transformation to be used for validating model
            self.pca_wts[prf_model_index][0:n_components_actual,0:n_features_actual] = wts[0:n_components_actual,:] 
            # mean of each feature, nfeatures long - needed to reproduce transformation
            self.pca_pre_mean[prf_model_index][0:n_features_actual] = pca.mean_ 
            # max len of ev is the number of components
            self.pct_var_expl[0:n_components_actual,prf_model_index] = ev[0:n_components_actual]  
            n_components_reduced = int(np.where(np.cumsum(ev)>self.min_pct_var)[0][0] if np.any(np.cumsum(ev)>self.min_pct_var) else len(ev))
            n_components_reduced = np.max([n_components_reduced, 1])
            self.n_comp_needed[prf_model_index] = n_components_reduced
            print('Retaining %d components to expl %d pct var'%(n_components_reduced, self.min_pct_var))
            assert(n_components_reduced<=self.max_pc_to_retain)            
            features_reduced = scores[:,0:n_components_reduced]
           
        else:
            
            # This is a validation pass, going to use the pca pars that were computed on training set
            # Make sure it has been done already!
            assert(self.n_comp_needed[prf_model_index]!=-1)
            print('Applying pre-computed PCA matrix...')
            # Apply the PCA transformation, just as it was done during training
            features -= np.tile(np.expand_dims(self.pca_pre_z_mean[prf_model_index][0:n_features_actual], axis=0), [n_trials, 1])
            features /= np.tile(np.expand_dims(self.pca_pre_z_std[prf_model_index][0:n_features_actual], axis=0), [n_trials, 1])
            
            features_submean = features - np.tile(np.expand_dims(self.pca_pre_mean[prf_model_index][0:n_features_actual], axis=0), [n_trials, 1])
            features_reduced = features_submean @ np.transpose(self.pca_wts[prf_model_index][0:self.n_comp_needed[prf_model_index],0:n_features_actual])               
                       
        features = None
        
        if was_tensor:
            features_reduced = torch.tensor(features_reduced).to(self.device)
        
        return features_reduced
    

def get_features_each_prf(features_file, models, mult_patch_by_prf=True, do_avg_pool=True, \
                          batch_size=100, aperture=1.0, debug=False, device=None):
    """
    Extract the portion of the feature maps corresponding to each prf in 'models'
    Start with loading the feature maps h5py file (generated by get_st_features.m)
    Save smaller features as an h5py file [n_images x n_features x n_prfs]
    """
    if device is None:
        device = 'cpu:0'
        
    with h5py.File(features_file, 'r') as data_set:
        ds_size = data_set['/features'].shape
    n_images = ds_size[3]
    n_features = ds_size[0]
    map_resolution = ds_size[1]
    n_prfs = models.shape[0]
    features_each_prf = np.zeros((n_images, n_features, n_prfs))
    n_batches = int(np.ceil(n_images/batch_size))

    for bb in range(n_batches):

        if debug and bb>1:
            continue

        batch_inds = np.arange(batch_size * bb, np.min([batch_size * (bb+1), n_images]))

        print('Loading features for images [%d - %d]'%(batch_inds[0], batch_inds[-1]))
        st = time.time()
        with h5py.File(features_file, 'r') as data_set:
            # Note this order is reversed from how it was saved in matlab originally.
            # The dimensions go [features x h x w x images]
            # Luckily h and w are swapped matlab to python anyway, so can just switch the first and last.
            values = np.copy(data_set['/features'][:,:,:,batch_inds])
            data_set.close()  
        fmaps_batch = np.moveaxis(values, [0,1,2,3],[3,1,2,0])

        elapsed = time.time() - st
        print('Took %.5f sec to load feature maps'%elapsed)

        maps_full_field = torch_utils._to_torch(fmaps_batch, device=device)

        for mm in range(n_prfs):

            if debug and mm>1:
                continue

            prf_params = models[mm,:]
            x,y,sigma = prf_params
            print('Getting features for pRF [x,y,sigma]:')
            print([x,y,sigma])
            n_pix = map_resolution

             # Define the RF for this "model" version
            prf = torch_utils._to_torch(prf_utils.make_gaussian_mass(x, y, sigma, n_pix, size=aperture, \
                                      dtype=np.float32)[2], device=device)
            minval = torch.min(prf)
            maxval = torch.max(prf-minval)
            prf_scaled = (prf - minval)/maxval

            if mult_patch_by_prf:         
                # This effectively restricts the spatial location, so no need to crop
                maps = maps_full_field * prf_scaled.view([1,map_resolution,map_resolution,1])
            else:
                # This is a coarser way of choosing which spatial region to look at
                # Crop the patch +/- n SD away from center
                n_pf_sd_out = 2
                bbox = texture_utils.get_bbox_from_prf(prf_params, prf.shape, n_prf_sd_out, min_pix=None, verbose=False, force_square=False)
                print('bbox to crop is:')
                print(bbox)
                maps = maps_full_field[:,bbox[0]:bbox[1], bbox[2]:bbox[3],:]

            if do_avg_pool:
                features_batch = torch.mean(maps, dim=(1,2))
            else:
                features_batch = torch.max(maps, dim=(1,2))
                
            print('model %d, min/max of features in batch: [%s, %s]'%(mm, torch.min(features_batch), torch.max(features_batch))) 

            features_each_prf[batch_inds,:,mm] = torch_utils.get_value(features_batch)
                      
    return features_each_prf