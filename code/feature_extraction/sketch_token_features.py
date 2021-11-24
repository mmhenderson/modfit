import numpy as np
import sys, os
import torch
import time
import h5py
import pandas as pd
import torch.nn as nn
from sklearn import decomposition

from utils import prf_utils, torch_utils, texture_utils, default_paths
sketch_token_feat_path = default_paths.sketch_token_feat_path

class sketch_token_feature_extractor(nn.Module):
    
    def __init__(self, subject, device, which_prf_grid=1, \
                 use_pca_feats = False, min_pct_var = 99, max_pc_to_retain = 100, \
                 use_lda_feats = False, lda_discrim_type = None, zscore_in_groups = False):
        
        super(sketch_token_feature_extractor, self).__init__()
        
        self.subject = subject
        
        self.use_pca_feats = use_pca_feats
        self.use_lda_feats = use_lda_feats
        self.lda_discrim_type = lda_discrim_type        
        self.which_prf_grid = which_prf_grid
        
        if self.use_pca_feats:
            self.n_features = 151
            self.use_lda_feats = False # only allow one of these to be true
            self.features_file = os.path.join(sketch_token_feat_path, 'PCA', \
                                          'S%d_PCA_grid%d.npy'%(self.subject, self.which_prf_grid))     
            self.min_pct_var = min_pct_var
            self.max_pc_to_retain = np.min([self.n_features, max_pc_to_retain])
        elif self.use_lda_feats:
            self.use_pca_feats = False
            self.min_pct_var = None
            self.max_pc_to_retain = None  
            self.features_file = os.path.join(sketch_token_feat_path, 'LDA', \
                  'S%d_LDA_%s_grid%d.npy'%(self.subject, self.lda_discrim_type, self.which_prf_grid))
            if self.lda_discrim_type=='all_supcat':
                self.n_features = 11                      
            elif self.lda_discrim_type=='animacy' or self.lda_discrim_type=='indoor_outdoor' or \
                    self.lda_discrim_type=='animal' or self.lda_discrim_type=='vehicle' or \
                    self.lda_discrim_type=='food' or self.lda_discrim_type=='person':
                self.n_features = 1   
            else:
                print(lda_discrim_type)
                raise ValueError('--lda_discrim_type was not recognized')
        else:
            self.n_features = 150 # 151
            self.features_file = os.path.join(sketch_token_feat_path, \
                                          'S%d_features_each_prf_grid%d.h5py'%(subject, self.which_prf_grid))
            self.min_pct_var = None
            self.max_pc_to_retain = None  
            if zscore_in_groups:
                self.zgroup_labels = np.concatenate([np.zeros(shape=(1,150)), np.ones(shape=(1,1))], axis=1)
                self.zgroup_labels = self.zgroup_labels[0,0:self.n_features]
                print('groups for z-scoring: ')
                print(np.unique(self.zgroup_labels))
            else:
                self.zgroup_labels = None
            
        if not os.path.exists(self.features_file):
            raise RuntimeError('Looking at %s for precomputed features, not found.'%self.features_file)

        self.prf_batch_size=50
        self.device = device
        self.do_varpart=False 
        # only one set of features in this model for now, not doing variance partition
        self.features_each_prf_batch = None
        self.prf_inds_loaded = []
        
    def init_for_fitting(self, image_size, models, dtype):

        """
        Additional initialization operations.
        """
        
        print('Initializing for fitting')

        if self.use_pca_feats:
            self.max_features = self.max_pc_to_retain        
        else:
            self.max_features = self.n_features
       
        n_prfs = models.shape[0]
        n_prf_batches = int(np.ceil(n_prfs/self.prf_batch_size))          
        self.prf_batch_inds = [np.arange(self.prf_batch_size*bb, np.min([self.prf_batch_size*(bb+1), \
                                                         n_prfs])) for bb in range(n_prf_batches)]
            
        self.clear_big_features()
        
    def get_partial_versions(self):

        if not hasattr(self, 'max_features'):
            raise RuntimeError('need to run init_for_fitting first')
           
        partial_version_names = ['full_model']
        masks = np.ones([1,self.max_features])

        return masks, partial_version_names

    def load_precomputed_features(self, image_inds, prf_model_index):
        
        if prf_model_index not in self.prf_inds_loaded:
            
            print('Loading pre-computed features from %s'%self.features_file)
            t = time.time()
            batch_to_use = np.where([prf_model_index in self.prf_batch_inds[bb] for \
                                         bb in range(len(self.prf_batch_inds))])[0][0]
            assert(prf_model_index in self.prf_batch_inds[batch_to_use])
            self.prf_inds_loaded = self.prf_batch_inds[batch_to_use]
            print('Loading pre-computed features for models [%d - %d] from %s'%\
                   (self.prf_batch_inds[batch_to_use][0], self.prf_batch_inds[batch_to_use][-1], \
                    self.features_file))
            self.features_each_prf_batch = None
            
            if self.use_pca_feats:

                # loading pre-computed pca features, and deciding here how many features to 
                # include in model.
                pc_result = np.load(self.features_file, allow_pickle=True).item()
                scores_each_prf = [pc_result['scores'][mm] for mm in self.prf_batch_inds[batch_to_use]]
                ev_each_prf = [pc_result['ev'][mm] for mm in self.prf_batch_inds[batch_to_use]]
                pc_result = None
                n_pcs_avail = scores_each_prf[0].shape[1]
                n_feat_each_prf = [np.where(np.cumsum(ev)>self.min_pct_var)[0][0] \
                                   if np.size(np.where(np.cumsum(ev)>self.min_pct_var))>0 \
                                   else n_pcs_avail for ev in ev_each_prf]
                n_feat_each_prf = [np.min([nf, self.max_pc_to_retain]) for nf in n_feat_each_prf]
                self.features_each_prf_batch = [scores_each_prf[mm][image_inds,0:n_feat_each_prf[mm]] \
                                          for mm in range(len(scores_each_prf))]   
                print('Length of features list this batch: %d'%len(self.features_each_prf_batch))
                print('Size of features array for first prf model with this image set is:')
                print(self.features_each_prf_batch[0].shape)

            elif self.use_lda_feats:

                # loading pre-computed linear discriminant analysis features
                lda_result = np.load(self.features_file, allow_pickle=True).item()
                scores_each_prf = [lda_result['scores'][mm] for mm in self.prf_batch_inds[batch_to_use]]
                lda_result = None
                self.features_each_prf_batch = np.moveaxis(np.array([scores_each_prf[mm][image_inds,:] \
                              for mm in range(len(scores_each_prf))]), [0,1,2], [2,0,1])
                assert(self.features_each_prf_batch.shape[1]==self.max_features)
                print('Size of features array for this image set is:')
                print(self.features_each_prf_batch.shape)

            else:

                # Loading raw sketch tokens features.
                with h5py.File(self.features_file, 'r') as data_set:
                    values = np.copy(data_set['/features'][:,:,self.prf_batch_inds[batch_to_use]])
                    data_set.close() 
                elapsed = time.time() - t
                print('Took %.5f seconds to load file'%elapsed)

                self.features_each_prf_batch = values[image_inds,:,:]
                values = None
                # Taking out the very last column, which represents "no contour"
                self.features_each_prf_batch = self.features_each_prf_batch[:,0:150,:]

                print('Size of features array for this image set is:')
                print(self.features_each_prf_batch.shape)

         
        index_into_batch = np.where(prf_model_index==self.prf_inds_loaded)[0][0]
        print('Index into batch for prf %d: %d'%(prf_model_index, index_into_batch))
        if self.use_pca_feats:
            features_in_prf = self.features_each_prf_batch[index_into_batch]
        else:
            features_in_prf = self.features_each_prf_batch[:,:,index_into_batch]
        assert(features_in_prf.shape[0]==len(image_inds))
        print('Size of features array for this image set and prf is:')
        print(features_in_prf.shape)
        
        return features_in_prf
        
    def clear_big_features(self):
        
        print('Clearing features from memory')
        self.features_each_prf_batch = None 
        self.prf_inds_loaded = []
        
    def forward(self, image_inds, prf_params, prf_model_index, fitting_mode = True):
        
        features = self.load_precomputed_features(image_inds, prf_model_index)
       
        assert(features.shape[0]==len(image_inds))
        print('Final size of feature matrix is:')
        print(features.shape)
        
        features = torch_utils._to_torch(features, self.device)
        
        feature_inds_defined = np.zeros((self.max_features,), dtype=bool)
        feature_inds_defined[0:features.shape[1]] = 1
            
        return features, feature_inds_defined
     
    
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
            prf = torch_utils._to_torch(prf_utils.gauss_2d(center=[x,y], sd=sigma, \
                               patch_size=n_pix, aperture=aperture, dtype=np.float32), device=device)
            minval = torch.min(prf)
            maxval = torch.max(prf-minval)
            prf_scaled = (prf - minval)/maxval

            if sigma==10:
                # creating a "flat" pRF here which will average across entire feature map.
                prf_scaled = torch.ones(prf_scaled.shape)
                prf_scaled = prf_scaled/torch.sum(prf_scaled)
                prf_scaled = prf_scaled.to(device)

            if mult_patch_by_prf:         
                # This effectively restricts the spatial location, so no need to crop
                maps = maps_full_field * prf_scaled.view([1,map_resolution,map_resolution,1])
            else:
                # This is a coarser way of choosing which spatial region to look at
                # Crop the patch +/- n SD away from center
                n_prf_sd_out = 2
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