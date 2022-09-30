"""
Run the model fitting for FWRF model. 
There are a few different versions of fitting in this script, the input arguments tell which kind of fitting to do.
"""

# import basic modules
import sys
import os
import time
import numpy as np
import argparse
import gc
import pandas as pd

# import custom modules
from utils import default_paths, floc_utils

import initialize_fitting, arg_parser
import fwrf_model

from analyze_fits import semantic_selectivity

try:
    device = initialize_fitting.init_cuda()
except:
    device = 'cpu:0'
    
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

print('numpy version: %s'%np.__version__)
#################################################################################################
        
    
def predict(args):

    model_name, fitting_types = initialize_fitting.get_full_save_name(args)
    save_fits_path = default_paths.save_fits_path
    subject_dir = os.path.join(save_fits_path, 'S%02d'%args.subject)
    date_str = initialize_fitting.most_recent_save(subject=args.subject, \
                                                   fitting_type=model_name)
    fn2load = os.path.join(subject_dir, model_name, date_str, 'all_fit_params.npy')
    if args.debug:
        fn2save = os.path.join(subject_dir, model_name, date_str, 'eval_on_%s_DEBUG.npy'%args.image_set)
    else:
        fn2save = os.path.join(subject_dir, model_name, date_str, 'eval_on_%s.npy'%args.image_set)
    
    print('loading saved model fit from %s'%fn2load)
    print('will save eval on %s image set to %s'%(args.image_set, fn2save))
    
    sys.stdout.flush()
    
    def save_all(fn2save):
    
        """
        Define all the important parameters that have to be saved
        """
        dict2save = {
        'model_name': model_name,
        'saved_model_fn': fn2load,
        'image_set': args.image_set,            
        'fitting_types': fitting_types, 
        'which_prf_grid': args.which_prf_grid,    
        'debug': args.debug,
        'best_layer_each_voxel': best_layer_each_voxel,
        'best_model_each_voxel': best_model_each_voxel,
        'sem_discrim_each_axis': sem_discrim_each_axis,
        'sem_corr_each_axis': sem_corr_each_axis,
        'discrim_type_list': discrim_type_list,
        'n_sem_samp_each_axis': n_sem_samp_each_axis,
        'mean_each_sem_level': mean_each_sem_level,
            }
        
        if np.any(['semantic' in ft for ft in fitting_types]):
            dict2save.update({
            'semantic_feature_set': args.semantic_feature_set,
            })
        if np.any(['sketch_tokens' in ft for ft in fitting_types]):
            dict2save.update({         
            'use_pca_st_feats': args.use_pca_st_feats,
            'use_residual_st_feats': args.use_residual_st_feats,
            })          
        if np.any(['pyramid' in ft for ft in fitting_types]):
            dict2save.update({
            'pyr_pca_type': args.pyr_pca_type,
            'group_all_hl_feats': args.group_all_hl_feats,
            'do_pyr_varpart': args.do_pyr_varpart,
            })            
        if np.any(['gabor' in ft for ft in fitting_types]):
            dict2save.update({
            'n_ori_gabor': args.n_ori_gabor,
            'n_sf_gabor': args.n_sf_gabor,
            'gabor_nonlin_fn': args.gabor_nonlin_fn,
            'use_pca_gabor_feats': args.use_pca_gabor_feats,
            })
        if np.any(['alexnet' in ft for ft in fitting_types]):
            dict2save.update({
            'alexnet_layer_name': args.alexnet_layer_name,
            'alexnet_padding_mode': args.alexnet_padding_mode,
            'use_pca_alexnet_feats': args.use_pca_alexnet_feats, 
            'alexnet_blurface': args.alexnet_blurface,
            })
        if np.any(['clip' in ft for ft in fitting_types]):
            dict2save.update({
            'clip_layer_name': args.resnet_layer_name,
            'clip_model_architecture': args.resnet_model_architecture,
            'use_pca_clip_feats': args.use_pca_resnet_feats,  
            'n_resnet_blocks_include': args.n_resnet_blocks_include,
            'clip_layers_use': dnn_layers_use,
            })
        if np.any(['resnet' in ft for ft in fitting_types]):
            dict2save.update({
            'resnet_layer_name': args.resnet_layer_name,
            'resnet_model_architecture': args.resnet_model_architecture,
            'use_pca_resnet_feats': args.use_pca_resnet_feats,  
            'n_resnet_blocks_include': args.n_resnet_blocks_include, 
            'resnet_blurface': args.resnet_blurface, 
            'resnet_layers_use': dnn_layers_use,
            })

        print('\nSaving to %s\n'%fn2save)
        print(dict2save.keys())
        np.save(fn2save, dict2save, allow_pickle=True)

        
    sem_discrim_each_axis = None
    sem_corr_each_axis = None
    discrim_type_list = None
    n_sem_samp_each_axis = None
    mean_each_sem_level = None
   
    if np.any(['alexnet' in ft for ft in fitting_types]):
        if args.alexnet_blurface: 
            dnn_model='alexnet_blurface'
        else:
            dnn_model='alexnet'
        n_dnn_layers = 5;
        dnn_layers_use = np.arange(5)
        assert(not np.any(['clip' in ft for ft in fitting_types]))
    elif np.any(['clip' in ft for ft in fitting_types]) or np.any(['resnet' in ft for ft in fitting_types]):
        if args.n_resnet_blocks_include==4:
            n_dnn_layers = 4;
            dnn_layers_use = [2,6,12,15]
        elif args.n_resnet_blocks_include==8:
            n_dnn_layers = 8;
            dnn_layers_use=np.arange(0,16,2)+1
        elif args.n_resnet_blocks_include==16:
            n_dnn_layers = 16;
            dnn_layers_use = np.arange(0,16,1)
        else:
            raise ValueError('n_resnet_blocks_include must be 4,8, or 16')
        if np.any(['clip' in ft for ft in fitting_types]):
            dnn_model='clip'
        elif np.any(['blurface' in ft for ft in fitting_types]):
            dnn_model='resnet_blurface'
        else:
            dnn_model='resnet'
        assert(not np.any(['alexnet' in ft for ft in fitting_types]))
    else:
        dnn_model = None
        dnn_layers_use=None
          
    ###### LOAD THE SAVED ENCODING MODEL ############################################################################
    
    print('\nLoading the results of training from %s\n'%fn2load)
    last_saved = np.load(fn2load, allow_pickle=True).item()

    assert(last_saved['which_prf_grid']==args.which_prf_grid)      
    assert(last_saved['zscore_features']==args.zscore_features)      
    assert(np.all(last_saved['voxel_subset_is_done_val']))

    # pull out the parameters of this model so we can reproduce it
    best_params = last_saved['best_params'] 
    best_prf_model_pars, best_weights, best_biases, \
                       features_mean, features_std, best_prf_models = best_params

    print('shape of saved weights:')
    print(best_weights.shape)
    
    best_model_each_voxel = best_prf_models[:,0]
    best_layer_each_voxel = last_saved['best_layer_each_voxel']
    
    n_voxels = best_weights.shape[0]
    n_prfs = last_saved['models'].shape[0]
        
    ########## INFO ABOUT THE EVALUATION IMAGES #################################################################
    if args.image_set=='floc':
        
        labels_file = os.path.join(default_paths.floc_image_root,'floc_image_labels.csv')
        labels = pd.read_csv(labels_file)
        
        image_inds_use = floc_utils.load_balanced_floc_set()
        image_inds_val = np.where(image_inds_use)[0]
        print('using %d images'%len(image_inds_val))
        sem_labels = np.array([labels['domain']==domain for domain in floc_utils.domains]).T.astype(int)
        unique_labs_each = [np.unique(sem_labels[:,dd]) for dd in range(sem_labels.shape[1])]
        labels_all = np.tile(sem_labels[:,:,None], [1,1,n_prfs])
        labels_all = labels_all[image_inds_val,:]
        discrim_type_list = ['%s > other domains'%domain for domain in floc_utils.domains]
        print(discrim_type_list)
        
    else:
        raise ValueError('image_set %s not recognized'%args.image_set)

    ####### DEFINE VOXEL SUBSETS TO LOOP OVER ###############################################################
    # also making feature loaders here
    
    if dnn_model is not None and (args.alexnet_layer_name=='best_layer' or args.resnet_layer_name=='best_layer'):

        assert(np.all(np.unique(best_layer_each_voxel)==np.arange(n_dnn_layers)))
        voxel_subset_masks = [best_layer_each_voxel==ll for ll in range(n_dnn_layers)]
       
        # Create feature loaders here
        feat_loader_full_list = [initialize_fitting.make_feature_loaders(args, fitting_types, vi=ll) \
                            for ll in dnn_layers_use]
        assert(len(feat_loader_full_list)==n_dnn_layers)
        
    else:
        # going to fit all voxels w same model
        voxel_subset_masks = [np.ones((n_voxels,), dtype=bool)]
        
        # Create feature loaders here
        feat_loader_full_list = [initialize_fitting.make_feature_loaders(args, fitting_types, \
                                                                         vi=0, dnn_layers_use=dnn_layers_use)]
        
    max_features_overall = np.max([fl.max_features for fl in feat_loader_full_list])      
   
    sys.stdout.flush()
       
    ########### LOOPING OVER VOXEL SUBSETS ######################################################
    for vi, voxel_subset_mask in enumerate(voxel_subset_masks):
        
        if best_model_each_voxel is not None:
            best_model_each_voxel_use = best_model_each_voxel[voxel_subset_mask]
        else:
            best_model_each_voxel_use = None
        print('\nStarting fitting for voxel mask %d of %d, number of voxels this loop=%d'%(vi, \
                                           len(voxel_subset_masks), np.sum(voxel_subset_mask)))
        if np.sum(voxel_subset_mask)==0:
            print('no voxels, continuing loop')
            continue
        
        # pull out my current feature loader
        feat_loader_full = feat_loader_full_list[vi]
        max_features = feat_loader_full.max_features 
        
        sys.stdout.flush()
            
        ########## INITIALIZE ENCODING MODEL ##################################################
        
        model = fwrf_model.encoding_model(feat_loader_full, \
                                            best_model_each_voxel = best_model_each_voxel_use, \
                                            zscore=args.zscore_features, \
                                            add_bias=True, \
                                            voxel_batch_size=args.voxel_batch_size,\
                                            sample_batch_size=args.sample_batch_size,\
                                            device=device,\
                                            shuffle_data = args.shuffle_data, \
                                            bootstrap_data = args.bootstrap_data, \
                                            dtype=np.float32, debug=args.debug)
                  
        
        best_weights_tmp = best_weights[voxel_subset_mask,0:max_features,:]
        best_biases_tmp = best_biases[voxel_subset_mask,:]
        best_prf_models_tmp = best_prf_models[voxel_subset_mask,:]
        features_mean_tmp = features_mean[:,0:max_features,vi]
        features_std_tmp = features_std[:,0:max_features,vi]

        # put the saved pars into the model so that we can evaluate it
        model.best_weights, model.best_biases, \
        model.best_prf_models, \
        model.features_mean, model.features_std = \
            best_weights_tmp, best_biases_tmp, \
            best_prf_models_tmp, \
            features_mean_tmp, features_std_tmp  
            
        print('shape of model.best_weights')
        print(model.best_weights.shape)
        ############### VALIDATE MODEL ##################################################################
    
        print('Starting validation (voxel subset %d of %d)...\n'%(vi, len(voxel_subset_masks)))
        sys.stdout.flush()

        ##TODO
        model.validate(val_voxel_data=None, image_inds_val=image_inds_val)

        voxel_data_val_pred = model.pred_voxel_data

        ########### ESTIMATE SEMANTIC DISCRIMINABILITY #######################################################
         
        print('\nStarting semantic discriminability analysis (voxel subset %d of %d)...\n'%(vi, len(voxel_subset_masks)))
        sys.stdout.flush()
        
        print('shape of pred data:')
        print(voxel_data_val_pred.shape)
        print('shape of labels:')
        print(labels_all.shape)
        
        discrim_tmp, corr_tmp, n_samp_tmp, mean_tmp = \
                semantic_selectivity.get_semantic_discrim(model.best_prf_models, \
                                                  labels_all, unique_labs_each, \
                                                  voxel_data_val_pred,\
                                                  debug=args.debug)
        if vi==0:
            sem_discrim_each_axis = np.zeros((n_voxels, discrim_tmp.shape[1]), \
                                             dtype=discrim_tmp.dtype) 
            sem_corr_each_axis = np.zeros((n_voxels, corr_tmp.shape[1]), \
                                             dtype=corr_tmp.dtype)
            n_sem_samp_each_axis = np.zeros((n_voxels, n_samp_tmp.shape[1], n_samp_tmp.shape[2]), \
                                             dtype=n_samp_tmp.dtype)
            mean_each_sem_level = np.zeros((n_voxels, mean_tmp.shape[1], mean_tmp.shape[2]), \
                                             dtype=mean_tmp.dtype)
        sem_discrim_each_axis[voxel_subset_mask,:] = discrim_tmp
        sem_corr_each_axis[voxel_subset_mask,:] = corr_tmp
        n_sem_samp_each_axis[voxel_subset_mask,:,:] = n_samp_tmp
        mean_each_sem_level[voxel_subset_mask,:,:] = mean_tmp

        save_all(fn2save)

        # Done!

if __name__ == '__main__':
    
    args = arg_parser.get_args()
    
    # overwrite some of these args to make sure things won't break
    args.do_varpart=False
    args.do_pyr_varpart=False
    args.shuffle_data=False
    args.bootstrap_data=False
    args.trial_subset='all'
    
    predict(args)
