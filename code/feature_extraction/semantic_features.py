import numpy as np
import sys, os
import torch
import pandas as pd
import torch.nn as nn

from utils import torch_utils, default_paths, nsd_utils

class semantic_feature_extractor(nn.Module):
    
    def __init__(self, subject, feature_set, device, sessions=None, which_prf_grid=1):
        
        super(semantic_feature_extractor, self).__init__()
        
        self.subject = subject
        self.feature_set = feature_set  
        self.which_prf_grid = which_prf_grid

        self.get_trn_val_inds(sessions)
        
        if feature_set=='indoor_outdoor':
            self.features_file = os.path.join(default_paths.stim_labels_root, \
                                          'S%d_indoor_outdoor.csv'%self.subject)
            self.same_labels_all_prfs=True
            self.n_features = 2
        else:
            self.labels_folder = os.path.join(default_paths.stim_labels_root, \
                                         'S%d_within_prf_grid%d'%(self.subject, self.which_prf_grid))
            self.same_labels_all_prfs=False
            if feature_set=='natural_humanmade':            
                self.features_file = os.path.join(self.labels_folder, \
                                  'S%d_natural_humanmade_prf0.csv'%(self.subject))          
                self.n_features = 2
            elif 'coco_things' in feature_set:
                self.features_file = os.path.join(self.labels_folder, \
                                  'S%d_cocolabs_binary_prf0.csv'%(self.subject))
                if 'supcateg' in feature_set:                    
                    self.n_features = 12
                else:
                    self.n_features = 80
            elif 'coco_stuff' in feature_set:
                self.features_file = os.path.join(self.labels_folder, \
                                  'S%d_cocolabs_stuff_binary_prf0.csv'%(self.subject))
                if 'supcateg' in feature_set:                    
                    self.n_features = 16
                else:
                    self.n_features = 92           
            else:
                self.features_file = os.path.join(self.labels_folder, \
                                      'S%d_cocolabs_binary_prf0.csv'%(self.subject))
                self.n_features = 2

        if not os.path.exists(self.features_file):
            raise RuntimeError('Looking at %s for precomputed features, not found.'%self.features_file)

        self.device = device
        self.features_in_prf = None
        
    def get_trn_val_inds(self, sessions):
        
        # need to know which sessions we plan to work with, to see if any 
        # of the features will be missing in training set.
        # will then ignore those features always, no matter what trials we are 
        # currently working with.
        if sessions is None:
            self.sessions = np.arange(0,nsd_utils.max_sess_each_subj[self.subject-1]);
        else:
            self.sessions = np.array(sessions)
        print('working with these sessions for subject %d:'%self.subject)
        print(self.sessions)
        image_order = nsd_utils.get_master_image_order()
        session_inds = nsd_utils.get_session_inds_full()
        inds2use = np.isin(session_inds, self.sessions)
        image_order = image_order[inds2use]
        shared_1000_inds = image_order<1000   
        self.image_order_val = image_order[shared_1000_inds]
        self.image_order_trn = image_order[~shared_1000_inds]
        
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
            labels = np.array(coco_df)
            colnames = list(coco_df.keys())           
        else:
            if self.feature_set=='natural_humanmade':
                self.features_file = os.path.join(self.labels_folder, \
                                  'S%d_natural_humanmade_prf%d.csv'%(self.subject, prf_model_index))
                print('Loading pre-computed features from %s'%self.features_file)
                nat_hum_df = pd.read_csv(self.features_file, index_col=0)                
                labels = np.array(nat_hum_df)
                colnames = list(nat_hum_df.keys())
            elif 'coco_stuff' in self.feature_set:
                self.features_file = os.path.join(self.labels_folder, \
                                  'S%d_cocolabs_stuff_binary_prf%d.csv'%(self.subject, \
                                                                         prf_model_index))
                print('Loading pre-computed features from %s'%self.features_file)
                coco_df = pd.read_csv(self.features_file, index_col=0)
                if 'supcateg' in self.feature_set:
                    labels = np.array(coco_df)[:,0:16]
                    colnames = list(coco_df.keys())[0:16]
                else:
                    labels = np.array(coco_df)[:,16:]   
                    colnames = list(coco_df.keys())[16:]
            else:
                self.features_file = os.path.join(self.labels_folder, \
                                  'S%d_cocolabs_binary_prf%d.csv'%(self.subject, prf_model_index))
                print('Loading pre-computed features from %s'%self.features_file)
                coco_df = pd.read_csv(self.features_file, index_col=0)
                if 'supcateg' in self.feature_set:
                    labels = np.array(coco_df)[:,0:12]
                    colnames = list(coco_df.keys())[0:12]
                elif 'categ' in self.feature_set:
                    labels = np.array(coco_df)[:,12:92]   
                    colnames = list(coco_df.keys())[12:92]
                elif self.feature_set=='animacy':    
                    supcat_labels = np.array(coco_df)[:,0:12]
                    animate_supcats = [1,9]
                    inanimate_supcats = [ii for ii in range(12)\
                                         if ii not in animate_supcats]
                    has_animate = np.any(np.array([supcat_labels[:,ii]==1 \
                                                   for ii in animate_supcats]), axis=0)
                    has_inanimate = np.any(np.array([supcat_labels[:,ii]==1 \
                                                for ii in inanimate_supcats]), axis=0)
                    labels = np.concatenate([has_animate[:,np.newaxis], \
                                             has_inanimate[:,np.newaxis]], axis=1)
                    colnames = ['has_animate','has_inanimate']
                else:
                    has_label = np.any(np.array(coco_df)[:,0:12]==1, axis=1)
                    label1 = np.array(coco_df[self.feature_set])[:,np.newaxis]
                    label2 = (label1==0) & (has_label[:,np.newaxis])
                    labels = np.concatenate([label1, label2], axis=1)
                    colnames = ['has_%s'%self.feature_set, 'has_other']
                    
        print('using feature set: %s'%self.feature_set)
        assert(labels.shape[1]==self.n_features)
        print(colnames)
        
        # Check if any labels are missing in the training set
        missing_trn = np.sum(labels[self.image_order_trn,:], axis=0)==0
        print('%d columns are missing in training set for this pRF'%\
              (np.sum(missing_trn)));
        missing = missing_trn
        print('missing columns are:')
        print([colnames[cc] for cc in range(len(colnames)) if missing[cc]])
        # remove these so we won't get a degenerate matrix during fitting.
        labels = labels[:,~missing]
        self.is_defined_in_prf = ~missing   

        labels = labels[image_inds,:].astype(np.float32)           
        self.features_in_prf = labels;
        
        # print counts to verify things are working ok
        if self.n_features==2 and labels.shape[1]==2:
            print('num 1/1, 1/0, 0/1, 0/0:')
            print([np.sum((labels[:,0]==1) & (labels[:,1]==1)), \
                   np.sum((labels[:,0]==1) & (labels[:,1]==0)),\
                   np.sum((labels[:,0]==0) & (labels[:,1]==1)),\
                   np.sum((labels[:,0]==0) & (labels[:,1]==0))])
        else:
            print('num each column:')
            print(np.sum(labels, axis=0).astype(int))
            
        print('Size of features array for this image set is:')
        print(self.features_in_prf.shape)
        
    def clear_big_features(self):
        
        print('Clearing features from memory')
        self.features_in_prf = None 
       
    def forward(self, image_inds, prf_params, prf_model_index, fitting_mode = True):

        if (not self.same_labels_all_prfs) or (self.features_in_prf is None):
            self.load_precomputed_features(image_inds, prf_model_index)
        
        features = self.features_in_prf
        feature_inds_defined = self.is_defined_in_prf
        assert(len(feature_inds_defined)==self.max_features)
        
        assert(features.shape[0]==len(image_inds))
        print('Final size of feature matrix is:')
        print(features.shape)
        
        features = torch_utils._to_torch(features, self.device)
          
        return features, feature_inds_defined
     
    