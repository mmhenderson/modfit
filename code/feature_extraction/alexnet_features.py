import numpy as np
import sys, os
import time
import h5py
import gc
import torch.nn as nn
import torch

from utils import default_paths, torch_utils
alexnet_feat_path = default_paths.alexnet_feat_path
from feature_extraction import extract_alexnet_features
alexnet_layer_names  = extract_alexnet_features.alexnet_layer_names
n_features_each_layer = extract_alexnet_features.n_features_each_layer

class alexnet_feature_extractor(nn.Module):
    
    def __init__(self, subject, layer_name, device, which_prf_grid=1, padding_mode=None, \
                use_pca_feats = False, min_pct_var = 95, max_pc_to_retain=100):
        
        super(alexnet_feature_extractor, self).__init__()
        
        self.subject = subject       
        self.layer_name = layer_name
        self.which_prf_grid = which_prf_grid
        self.padding_mode = padding_mode 
        if padding_mode is None:
            padding_str = ''
        else:
            padding_str = padding_mode+'_'
        self.use_pca_feats = use_pca_feats
        
        layer_ind = [ll for ll in range(len(alexnet_layer_names)) \
                         if alexnet_layer_names[ll]==self.layer_name]
        assert(len(layer_ind)==1)
        layer_ind = layer_ind[0]        
        self.n_features = n_features_each_layer[layer_ind]
           
        if self.use_pca_feats:
            self.features_file = os.path.join(alexnet_feat_path, 'PCA', \
              'S%d_%s_%sPCA_grid%d.npy'%(subject, self.layer_name, \
                                                         padding_str, self.which_prf_grid))         
            self.min_pct_var = min_pct_var
            self.max_pc_to_retain = np.min([self.n_features, max_pc_to_retain])
        else:
            self.features_file = os.path.join(alexnet_feat_path, \
              'S%d_%s_%sfeatures_each_prf_grid%d.h5py'%(subject, self.layer_name, \
                                                         padding_str, self.which_prf_grid))
            self.min_pct_var = None
            self.max_pc_to_retain = None  
            
        if not os.path.exists(self.features_file):
            raise RuntimeError('Looking at %s for precomputed features, not found.'%self.features_file)

        
        self.device = device
        self.do_varpart=False # only one set of features in this model for now, not doing variance partition

        self.prf_batch_size=100
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
        
        # Prepare for loading the pre-computed features: as a 
        # compromise between speed and ram usage, will load them in
        # batches of multiple prfs at a time. 
        n_prfs = models.shape[0]
        n_prf_batches = int(np.ceil(n_prfs/self.prf_batch_size))          
        self.prf_batch_inds = [np.arange(self.prf_batch_size*bb, np.min([self.prf_batch_size*(bb+1), n_prfs])) \
                               for bb in range(n_prf_batches)]
       
        self.clear_big_features()
        
    def get_partial_versions(self):

        if not hasattr(self, 'max_features'):
            raise RuntimeError('need to run init_for_fitting first')
           
        partial_version_names = ['full_model']
        masks = np.ones([1,self.max_features])

        return masks, partial_version_names

    def load_precomputed_features(self, image_inds, prf_model_index):
        
        if prf_model_index not in self.prf_inds_loaded:
            
            batch_to_use = np.where([prf_model_index in self.prf_batch_inds[bb] for \
                                         bb in range(len(self.prf_batch_inds))])[0][0]
            assert(prf_model_index in self.prf_batch_inds[batch_to_use])
            self.prf_inds_loaded = self.prf_batch_inds[batch_to_use]
            print('Loading pre-computed features for models [%d - %d] from %s'%(self.prf_batch_inds[batch_to_use][0], \
                                                                              self.prf_batch_inds[batch_to_use][-1], self.features_file))
            self.features_each_prf_batch = None
        
            gc.collect()
            torch.cuda.empty_cache()

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
                
            else:
            
                t = time.time()
                # Loading raw features.
                with h5py.File(self.features_file, 'r') as data_set:
                    values = np.copy(data_set['/features'][:,:,self.prf_batch_inds[batch_to_use]])
                    data_set.close() 
                elapsed = time.time() - t
                print('Took %.5f seconds to load file'%elapsed)

                self.features_each_prf_batch = values[image_inds,:,:]
                values=None
                assert(self.n_features==self.features_each_prf_batch.shape[1])

                print('Size of features array for this batch, is:')
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
        gc.collect()
        torch.cuda.empty_cache()
    
    def forward(self, image_inds, prf_params, prf_model_index, fitting_mode = True):
         
        features = self.load_precomputed_features(image_inds, prf_model_index)

        assert(features.shape[0]==len(image_inds))
        print('Final size of feature matrix is:')
        print(features.shape)
        
        features = torch_utils._to_torch(features, self.device)
        
        feature_inds_defined = np.zeros((self.max_features,), dtype=bool)
        feature_inds_defined[0:features.shape[1]] = 1
            
        return features, feature_inds_defined
 