import numpy as np
import sys, os
import torch
import pandas as pd
import torch.nn as nn

from utils import torch_utils, default_paths

class semantic_feature_extractor(nn.Module):
    
    def __init__(self, subject, discrim_type, device, which_prf_grid=1):
        
        super(semantic_feature_extractor, self).__init__()
        
        self.subject = subject
        self.discrim_type = discrim_type  
        self.which_prf_grid = which_prf_grid
        
        if discrim_type=='indoor_outdoor':
            self.features_file = os.path.join(default_paths.stim_labels_root, \
                                          'S%d_indoor_outdoor.csv'%self.subject)
            self.same_labels_all_prfs=True
            self.n_features = 2
        elif discrim_type=='natural_humanmade':
            self.labels_folder = os.path.join(default_paths.stim_labels_root, \
                                         'S%d_within_prf_grid%d'%(self.subject, self.which_prf_grid))
            self.features_file = os.path.join(self.labels_folder, \
                                  'S%d_natural_humanmade_prf0.csv'%(self.subject))
            self.same_labels_all_prfs=False
            self.n_features = 2
        elif discrim_type=='all_supcat':
            self.labels_folder = os.path.join(default_paths.stim_labels_root, \
                                         'S%d_within_prf_grid%d'%(self.subject, self.which_prf_grid))
            self.features_file = os.path.join(self.labels_folder, \
                                  'S%d_cocolabs_binary_prf0.csv'%(self.subject))
            self.same_labels_all_prfs=False
            self.n_features = 12
        else:
            self.labels_folder = os.path.join(default_paths.stim_labels_root, \
                                         'S%d_within_prf_grid%d'%(self.subject, self.which_prf_grid))
            self.features_file = os.path.join(self.labels_folder, \
                                  'S%d_cocolabs_binary_prf0.csv'%(self.subject))
            self.same_labels_all_prfs=False
#             self.n_features = 1
            self.n_features = 2

        if not os.path.exists(self.features_file):
            raise RuntimeError('Looking at %s for precomputed features, not found.'%self.features_file)

        self.device = device
        self.features_in_prf = None
        
    def init_for_fitting(self, image_size, models, dtype):

        """
        Additional initialization operations.
        """
        
        print('Initializing for fitting')
        self.max_features = self.n_features       
        self.clear_big_features()
        
    def get_partial_versions(self):

        if not hasattr(self, 'max_features'):
            raise RuntimeError('need to run init_for_fitting first')
           
        partial_version_names = ['full_model']
        masks = np.ones([1,self.max_features])

        return masks, partial_version_names

    def load_precomputed_features(self, image_inds, prf_model_index):

        if self.same_labels_all_prfs:
            print('Loading pre-computed features from %s'%self.features_file)        
            coco_df = pd.read_csv(self.features_file, index_col=0)
            if self.discrim_type=='indoor_outdoor':
                # some of these images are ambiguous, so leaving both columns here to allow
                # for images to have both/neither label
                labels = np.concatenate([np.array(coco_df['has_indoor'])[:,np.newaxis], \
                                         np.array(coco_df['has_outdoor'])[:,np.newaxis]], axis=1)
            else:
                raise ValueError('discrim type not implemented yet')
            
        else:
            if self.discrim_type=='natural_humanmade':
                self.features_file = os.path.join(self.labels_folder, \
                                  'S%d_natural_humanmade_prf%d.csv'%(self.subject, prf_model_index))
                print('Loading pre-computed features from %s'%self.features_file)
                nat_hum_df = pd.read_csv(self.features_file, index_col=0)                
                labels = np.array(nat_hum_df)
            else:
                self.features_file = os.path.join(self.labels_folder, \
                                  'S%d_cocolabs_binary_prf%d.csv'%(self.subject, prf_model_index))
                print('Loading pre-computed features from %s'%self.features_file)
                coco_df = pd.read_csv(self.features_file, index_col=0)
                has_label = np.any(np.array(coco_df)[:,0:12]==1, axis=1)
                if self.discrim_type=='animacy':                    
                    animate_supcats = [1,9]
                    inanimate_supcats = [ii for ii in range(len(supcat_names))\
                                         if ii not in animate_supcats]
                    has_animate = np.any(np.array([supcat_labels[:,ii]==1 \
                                                   for ii in animate_supcats]), axis=0)
                    has_inanimate = np.any(np.array([supcat_labels[:,ii]==1 \
                                                for ii in inanimate_supcats]), axis=0)
#                     labels = np.array(coco_df['has_animate'])[:,np.newaxis]
#                     label1 = np.array(coco_df['has_animate'])[:,np.newaxis]
                    # add another column to distinguish unlabeled from inanimate
#                     label2 = (label1==0) & (has_label[:,np.newaxis])
                    labels = np.concatenate([has_animate, has_inanimate], axis=1)
                elif self.discrim_type=='all_supcat':
                    # images can have more than one label or no labels here
                    labels = np.array(coco_df)[:,0:12]
                else:
#                     labels = np.array(coco_df[self.discrim_type])[:,np.newaxis]
                    label1 = np.array(coco_df[self.discrim_type])[:,np.newaxis]
                    label2 = (label1==0) & (has_label[:,np.newaxis])
                    labels = np.concatenate([label1, label2], axis=1)
            
        labels = labels[image_inds,:].astype(np.float32)           
        self.features_in_prf = labels;

        # print counts to verify things are working ok
        if self.n_features==2:
            print('num 1/1, 1/0, 0/1, 0/0:')
            print([np.sum((labels[:,0]==1) & (labels[:,1]==1)), \
                   np.sum((labels[:,0]==1) & (labels[:,1]==0)),\
                   np.sum((labels[:,0]==0) & (labels[:,1]==1)),\
                   np.sum((labels[:,0]==0) & (labels[:,1]==0))])
        else:
            print('num each column:')
            print(np.sum(labels, axis=0))
            
        print('Size of features array for this image set is:')
        print(self.features_in_prf.shape)
        
    def clear_big_features(self):
        
        print('Clearing features from memory')
        self.features_in_prf = None 
       
    def forward(self, image_inds, prf_params, prf_model_index, fitting_mode = True):

        if (not self.same_labels_all_prfs) or (self.features_in_prf is None):
            self.load_precomputed_features(image_inds, prf_model_index)
        
        features = self.features_in_prf
        
        assert(features.shape[0]==len(image_inds))
        print('Final size of feature matrix is:')
        print(features.shape)
        
        features = torch_utils._to_torch(features, self.device)
        
        feature_inds_defined = np.zeros((self.max_features,), dtype=bool)
        feature_inds_defined[0:features.shape[1]] = 1
            
        return features, feature_inds_defined
     
    