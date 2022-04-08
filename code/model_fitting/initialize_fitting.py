
"""
These are all semi-general functions that are run before model fitting (file name to save, make feature extractors etc.)
"""

import torch
import time
import os
import numpy as np
import pandas as pd
from datetime import datetime

# import custom modules
from feature_extraction import gabor_feature_extractor
from utils import prf_utils, default_paths, nsd_utils

def init_cuda():
    
    # Set up CUDA stuff
    
    print ('#device:', torch.cuda.device_count())
    print ('device#:', torch.cuda.current_device())
    print ('device name:', torch.cuda.get_device_name(torch.cuda.current_device()))

    torch.manual_seed(time.time())
    device = torch.device("cuda:0") #cuda
    torch.backends.cudnn.enabled=True

    print ('\ntorch:', torch.__version__)
    print ('cuda: ', torch.version.cuda)
    print ('cudnn:', torch.backends.cudnn.version())
    print ('dtype:', torch.get_default_dtype())

    return device

def get_full_save_name(args):

    input_fitting_types = [args.fitting_type, args.fitting_type2, args.fitting_type3]
    input_fitting_types = [ft for ft in input_fitting_types if ft!='']
    fitting_types = [] # this is a list of all the feature types to include, used to create modules.
    model_name = '' # model_name is a string used to name the folder we will save to
    for fi, ft in enumerate(input_fitting_types):
        print(ft)
        if ft=='full_midlevel':
            fitting_types += ['gabor_solo', 'pyramid_texture','sketch_tokens']
            model_name += 'full_midlevel'
        elif ft=='semantic':
            if args.semantic_feature_set=='all_coco':
                fitting_types += ['semantic_coco_things_supcateg','semantic_coco_things_categ',\
                             'semantic_coco_stuff_supcateg','semantic_coco_stuff_categ']
                model_name += 'all_coco'
            elif args.semantic_feature_set=='all_coco_stuff':
                fitting_types += ['semantic_coco_stuff_supcateg','semantic_coco_stuff_categ']
                model_name += 'all_coco_stuff'
            elif args.semantic_feature_set=='all_coco_things':
                fitting_types += ['semantic_coco_things_supcateg','semantic_coco_things_categ']
                model_name += 'all_coco_things'
            elif args.semantic_feature_set=='all_coco_categ':
                fitting_types += ['semantic_coco_things_categ','semantic_coco_stuff_categ']
                model_name += 'all_coco_categ'
            elif args.semantic_feature_set=='all_coco_categ_pca':
                fitting_types += ['semantic_coco_things_categ_pca','semantic_coco_stuff_categ_pca']
                model_name += 'all_coco_categ_pca'
            elif args.semantic_feature_set=='highlevel_concat':
                fitting_types += ['semantic_indoor_outdoor', 'semantic_animacy', 'semantic_real_world_size']
                model_name += 'semantic_highlevel_concat'
            else:
                fitting_types += ['semantic_%s'%args.semantic_feature_set]
                model_name += 'semantic_%s'%args.semantic_feature_set
                
        elif 'texture_pyramid' in ft:
            fitting_types += [ft]
            model_name += 'texture_pyramid'
            if args.ridge==True:   
                model_name += '_ridge'
            else:
                model_name += '_OLS'
            model_name += '_%dori_%dsf'%(args.n_ori_pyr, args.n_sf_pyr)        
            if args.use_pca_pyr_feats_hl:
                model_name += '_pca_HL' 
            if not args.group_all_hl_feats:
                model_name += '_allsubsets'
        elif 'gabor_solo' in ft:     
            fitting_types += [ft]
            model_name += 'gabor_solo'
            if args.ridge==True:   
                model_name += '_ridge'
            else:
                model_name += '_OLS'
            model_name+='_%dori_%dsf'%(args.n_ori_gabor, args.n_sf_gabor)
            if args.use_pca_gabor_feats:
                model_name += '_pca'
            if not args.use_precomputed_prfs:
                model_name += '_fit_pRFs'
            if args.use_model_residuals:
                model_name += '_from_residuals'
        elif 'sketch_tokens' in ft:      
            fitting_types += [ft]
            if args.use_pca_st_feats==True:       
                model_name += 'sketch_tokens_pca'
            else:        
                model_name += 'sketch_tokens'
            if not args.use_precomputed_prfs:
                model_name += '_fit_pRFs'
        elif 'alexnet' in ft:
            fitting_types += [ft]
            if 'ReLU' in args.alexnet_layer_name:
                name = args.alexnet_layer_name.split('_')[0]
            else:
                name = args.alexnet_layer_name
            model_name += 'alexnet_%s'%name
            if args.use_pca_alexnet_feats:
                model_name += '_pca'
        elif 'clip' in ft:
            fitting_types += [ft]
            model_name += 'clip_%s_%s'%(args.clip_model_architecture, args.clip_layer_name)
            if args.use_pca_clip_feats:
                model_name += '_pca'
        else:
            raise ValueError('fitting type "%s" not recognized'%ft)
        
        if fi<len(input_fitting_types)-1:
            model_name+='_plus_'
            
    if args.trial_subset!='all':
        model_name += '_%s'%args.trial_subset

    print(fitting_types)
    print(model_name)
    
    return model_name, fitting_types


def get_save_path(model_name, args):
    
    # choose where to save results of fitting - always making a new file w current timestamp.
    # add these suffixes to the file name if it's one of the control analyses
    if args.shuffle_images==True:
        model_name = model_name + '_SHUFFLEIMAGES'
    if args.random_images==True:
        model_name = model_name + '_RANDOMIMAGES'
    if args.random_voxel_data==True:
        model_name = model_name + '_RANDOMVOXELDATA'
    
    save_fits_path = default_paths.save_fits_path
    if args.volume_space:
        subject_dir = os.path.join(save_fits_path, 'S%02d'%args.subject)
    else:
        subject_dir = os.path.join(save_fits_path,'S%02d_surface'%args.subject)
    
    if args.from_scratch:
        timestamp = time.strftime('%b-%d-%Y_%H%M_%S', time.localtime())
        make_new_folder = True
    elif (args.date_str=='') or (args.date_str=='0'):
        # load most recent file
        files_in_dir = os.listdir(os.path.join(subject_dir, model_name))       
        my_dates = [f for f in files_in_dir if 'ipynb' not in f and 'DEBUG' not in f]
        try:
            my_dates.sort(key=lambda date: datetime.strptime(date, "%b-%d-%Y_%H%M_%S"))
        except:
            my_dates.sort(key=lambda date: datetime.strptime(date, "%b-%d-%Y_%H%M"))
        most_recent_date = my_dates[-1]
        timestamp = most_recent_date
        make_new_folder = False
    else:
        # if you specified an existing timestamp, then won't try to make new folder, need to find existing one.
        timestamp = args.date_str
        make_new_folder = False
        
    print ("Time Stamp: %s" % timestamp)  
    
    if args.debug==True:
        output_dir = os.path.join(subject_dir,model_name,'%s_DEBUG/'%timestamp)
    else:
        output_dir = os.path.join(subject_dir,model_name,'%s'%timestamp)
    if not os.path.exists(output_dir): 
        if make_new_folder:
            os.makedirs(output_dir)
        else:
            raise ValueError('the path at %s does not exist yet!!'%output_dir)
        
    fn2save = os.path.join(output_dir,'all_fit_params.npy')
    print('\nWill save final output file to %s\n'%output_dir)
    
    return output_dir, fn2save
 
def get_lambdas(fitting_types, zscore_features=True, ridge=True):
    
    if ridge==False:
        lambdas = np.array([0.0,0.0])        
    else:
        if zscore_features==True:
            lambdas = np.logspace(np.log(0.01),np.log(10**5+0.01),9, dtype=np.float32, base=np.e) - 0.01
        else:
            # range of feature values are different if choosing not to z-score. the performance of these lambdas
            # will vary depending on actual feature value ranges, be sure to check the results carefully
            lambdas = np.logspace(np.log(0.01),np.log(10**1+0.01),10, dtype=np.float32, base=np.e) - 0.01

        if np.any(['clip' in ft for ft in fitting_types]) or np.any(['alexnet' in ft for ft in fitting_types]):
            
            lambdas = np.logspace(np.log(0.01),np.log(10**10+0.01),20, dtype=np.float32, base=np.e) - 0.01
            
        if np.any(['semantic' in ft for ft in fitting_types]):
            # the semantic models occasionally end up with a column of all zeros, so make sure we have a 
            # small value for lambda rather than zero, to prevent issues with inverse.
            lambdas[lambdas==0] = 10e-9
            
    print('\nPossible lambda values are:')
    print(lambdas)

    return lambdas

def get_prf_models(which_grid=5, verbose=False):

    # models is three columns, x, y, sigma
    if which_grid==1:
        smin, smax = np.float32(0.04), np.float32(0.4)
        n_sizes = 8
        aperture_rf_range=1.1
        models = prf_utils.model_space_pyramid(prf_utils.logspace(n_sizes)(smin, smax), min_spacing=1.4, aperture=aperture_rf_range)  
    
    elif which_grid==2 or which_grid==3:
        smin, smax = np.float32(0.04), np.float32(0.8)
        n_sizes = 9
        aperture_rf_range=1.1
        models = prf_utils.model_space_pyramid2(prf_utils.logspace(n_sizes)(smin, smax), min_spacing=1.4, aperture=aperture_rf_range)  
        
    elif which_grid==4:
        models = prf_utils.make_polar_angle_grid(sigma_range=[0.04, 1], n_sigma_steps=12, \
                              eccen_range=[0, 1.4], n_eccen_steps=12, n_angle_steps=16)
    elif which_grid==5:
        models = prf_utils.make_log_polar_grid(sigma_range=[0.02, 1], n_sigma_steps=10, \
                              eccen_range=[0, 7/8.4], n_eccen_steps=10, n_angle_steps=16)
    elif which_grid==6:
        models = prf_utils.make_log_polar_grid_scale_size_eccen(eccen_range=[0, 7/8.4], \
                              n_eccen_steps = 10, n_angle_steps = 16)
    elif which_grid==7:
        models = prf_utils.make_rect_grid(sigma_range=[0.04, 0.04], n_sigma_steps=1, min_grid_spacing=0.04)
    else:
        raise ValueError('prf grid number not recognized')

    if verbose:
        print('number of pRFs: %d'%len(models))
        print('most extreme RF positions:')
        print(models[0,:])
        print(models[-1,:])

    return models

def load_precomputed_prfs(subject):
    
    if subject==1:
        saved_prfs_fn=os.path.join(default_paths.save_fits_path,\
                'S01/alexnet_all_conv_pca/Apr-01-2022_1317_39/all_fit_params.npy')
        # saved_prfs_fn=os.path.join(default_paths.save_fits_path,\
        #         'S01/alexnet_all_conv_pca/Nov-23-2021_2247_09/all_fit_params.npy')    
    elif subject==2:
        saved_prfs_fn=os.path.join(default_paths.save_fits_path,\
                'S02/alexnet_all_conv_pca/Apr-02-2022_2104_46/all_fit_params.npy')                       
        # saved_prfs_fn=os.path.join(default_paths.save_fits_path,\
        #         'S02/alexnet_all_conv_pca/Jan-07-2022_1815_05/all_fit_params.npy')
    elif subject==3:
        saved_prfs_fn=os.path.join(default_paths.save_fits_path,\
                'S03/alexnet_all_conv_pca/Apr-04-2022_0349_08/all_fit_params.npy')
        # saved_prfs_fn=os.path.join(default_paths.save_fits_path,\
                # 'S03/alexnet_all_conv_pca/Jan-11-2022_0342_58/all_fit_params.npy')
    elif subject==4:
        saved_prfs_fn=os.path.join(default_paths.save_fits_path,\
                'S04/alexnet_all_conv_pca/Apr-05-2022_1052_06/all_fit_params.npy')
        # saved_prfs_fn=os.path.join(default_paths.save_fits_path,\
                # 'S04/alexnet_all_conv_pca/Jan-13-2022_1805_02/all_fit_params.npy')
    elif subject==5:
        saved_prfs_fn=os.path.join(default_paths.save_fits_path,\
                'S05/alexnet_all_conv_pca/Jan-15-2022_1936_46/all_fit_params.npy')
    elif subject==6:
        saved_prfs_fn=os.path.join(default_paths.save_fits_path,\
                'S06/alexnet_all_conv_pca/Jan-19-2022_1358_01/all_fit_params.npy')
    elif subject==7:
        saved_prfs_fn=os.path.join(default_paths.save_fits_path,\
                'S07/alexnet_all_conv_pca/Jan-21-2022_0313_37/all_fit_params.npy')
    elif subject==8:
        saved_prfs_fn=os.path.join(default_paths.save_fits_path,\
                'S08/alexnet_all_conv_pca/Jan-22-2022_1508_21/all_fit_params.npy')
    else:
        raise ValueError('trying to load pre-computed prfs, but prf params are not yet computed for this model')

    print('Loading pre-computed pRF estimates for all voxels from %s'%saved_prfs_fn)
    out = np.load(saved_prfs_fn, allow_pickle=True).item()
    best_model_each_voxel = out['best_params'][5][:,0]
    
    return best_model_each_voxel, saved_prfs_fn

def load_best_model_layers(subject, model):
    
    if subject==1:
        if model=='clip':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                 'S01/clip_RN50_all_resblocks_pca/Dec-12-2021_1407_50/all_fit_params.npy')
        elif model=='alexnet':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                'S01/alexnet_all_conv_pca/Apr-01-2022_1317_39/all_fit_params.npy')
            # saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
            #      'S01/alexnet_all_conv_pca/Nov-23-2021_2247_09/all_fit_params.npy')    
    elif subject==2:
        if model=='clip':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                 'S02/clip_RN50_all_resblocks_pca/Jan-13-2022_1121_18/all_fit_params.npy')
        elif model=='alexnet':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                'S02/alexnet_all_conv_pca/Apr-02-2022_2104_46/all_fit_params.npy') 
            # saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
            #     'S02/alexnet_all_conv_pca/Jan-07-2022_1815_05/all_fit_params.npy')
    elif subject==3:
        if model=='clip':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                 'S03/clip_RN50_all_resblocks_pca/Jan-18-2022_1156_04/all_fit_params.npy')
        elif model=='alexnet':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                'S03/alexnet_all_conv_pca/Apr-04-2022_0349_08/all_fit_params.npy')
            # saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
            #     'S03/alexnet_all_conv_pca/Jan-11-2022_0342_58/all_fit_params.npy')
    elif subject==4:
        if model=='clip':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                'S04/clip_RN50_all_resblocks_pca/Jan-23-2022_2137_23/all_fit_params.npy')               
        elif model=='alexnet':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                'S04/alexnet_all_conv_pca/Apr-05-2022_1052_06/all_fit_params.npy')
            # saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
            #     'S04/alexnet_all_conv_pca/Jan-13-2022_1805_02/all_fit_params.npy')
    elif subject==5:
        if model=='clip':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                'S05/clip_RN50_all_resblocks_pca/Jan-27-2022_1124_04/all_fit_params.npy')
        elif model=='alexnet':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                'S05/alexnet_all_conv_pca/Jan-15-2022_1936_46/all_fit_params.npy')
    elif subject==6:
        if model=='clip':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                'S06/clip_RN50_all_resblocks_pca/Jan-31-2022_0220_32/all_fit_params.npy')
        elif model=='alexnet':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                'S06/alexnet_all_conv_pca/Jan-19-2022_1358_01/all_fit_params.npy')
    elif subject==7:
        if model=='clip':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                'S07/clip_RN50_all_resblocks_pca/Feb-03-2022_2258_44/all_fit_params.npy')
        elif model=='alexnet':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                'S07/alexnet_all_conv_pca/Jan-21-2022_0313_37/all_fit_params.npy')
    elif subject==8:
        if model=='clip':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                'S08/clip_RN50_all_resblocks_pca/Feb-09-2022_1544_10/all_fit_params.npy')
        elif model=='alexnet':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                'S08/alexnet_all_conv_pca/Jan-22-2022_1508_21/all_fit_params.npy')
    else:
        raise ValueError('for S%d %s, best model layer not computed yet'%(subject, model))
    
    print('Loading best %s layer for all voxels from %s'%(model,saved_best_layer_fn))
    
    out = np.load(saved_best_layer_fn, allow_pickle=True).item()
    if model=='alexnet':
        layer_inds = [1,3,5,7,9]       
    elif model=='clip':
        layer_inds = np.arange(1,32,2)
      
    assert(np.all(['just_' in name for name in np.array(out['partial_version_names'])[layer_inds]]))
    best_layer_each_voxel = np.argmax(out['val_r2'][:,layer_inds], axis=1)
    unique, counts = np.unique(best_layer_each_voxel, return_counts=True)
    print('layer indices:')
    print(unique)
    print('num voxels:')
    print(counts)

    return best_layer_each_voxel, saved_best_layer_fn


def load_labels_each_prf(subject, which_prf_grid, image_inds, models, verbose=False, debug=False):

    """
    Load csv files containing spatially-specific category labels for coco images.
    Makes an array [n_trials x n_discrim_types x n_prfs]
    """

    labels_folder = os.path.join(default_paths.stim_labels_root, \
                                     'S%d_within_prf_grid%d'%(subject, which_prf_grid))
    groups = np.load(os.path.join(default_paths.stim_labels_root,\
                                  'All_concat_labelgroupnames.npy'), allow_pickle=True).item()
    col_names = groups['col_names_all']
    unique_labs_each = [np.arange(len(cn)) for cn in col_names]
    
    print('loading labels from folders at %s (will be slow...)'%(labels_folder))
  
    n_trials = image_inds.shape[0]
    n_prfs = models.shape[0]

    for prf_model_index in range(n_prfs):
        
        if debug and prf_model_index>1:
            continue
            
        fn2load = os.path.join(labels_folder, \
                                  'S%d_concat_prf%d.csv'%(subject, prf_model_index))
        concat_df = pd.read_csv(fn2load, index_col=0)
        labels = np.array(concat_df)
        labels = labels[image_inds,:]
        discrim_type_list = list(concat_df.keys())
        
        if prf_model_index==0:
            print(discrim_type_list)
            print('num nans each column (out of %d trials):'%n_trials)
            print(np.sum(np.isnan(labels), axis=0))
            n_sem_axes = len(discrim_type_list)
            labels_all = np.zeros((n_trials, n_sem_axes, n_prfs)).astype(np.float32)

        # put into big array
        
        labels_all[:,:,prf_model_index] = labels
    

    return labels_all, discrim_type_list, unique_labs_each


def get_trial_subsets(trn_image_order, val_image_order, prf_models, args):
    
    # going to work with a subset of trials only, defined by their semantic categories
    # note these definitions will be different for different pRFs 
    # this func will return an array [n_trials x n_prfs], masking out which trials to use
    # for each prf.
    # one for training trials, one for validation trials.
    
    n_prfs = prf_models.shape[0]
    
    labels_folder = os.path.join(default_paths.stim_labels_root, \
                         'S%d_within_prf_grid%d'%(args.subject, args.which_prf_grid))
    
    if 'indoor' in args.trial_subset or 'outdoor' in args.trial_subset:
        
        # load the labels for indoor vs outdoor (which is defined across entire images, not within pRF)
        in_out_labels_fn = os.path.join(default_paths.stim_labels_root, \
                                        'S%d_indoor_outdoor.csv'%args.subject)
        print('loading from %s'%in_out_labels_fn)
        in_out_df = pd.read_csv(in_out_labels_fn, index_col=0)
        
        has_one_label = np.sum(in_out_df, axis=1)==1
        if args.trial_subset=='indoor_only':
            trials_use = (np.array(in_out_df['has_indoor'])==1) & has_one_label
        elif args.trial_subset=='outdoor_only':
            trials_use = (np.array(in_out_df['has_outdoor'])==1) & has_one_label
        
        trials_use = np.tile(trials_use[:,np.newaxis], [1,n_prfs])
        
    elif 'animate' in args.trial_subset:
        
        trials_use = []
        for prf_model_index in range(n_prfs):
        
            if args.debug and prf_model_index>1:
                trials_use += [trials]
                continue
                
            coco_things_labels_fn = os.path.join(labels_folder, \
                                  'S%d_cocolabs_binary_prf%d.csv'%(args.subject, prf_model_index))
            if prf_model_index==0:
                print('loading from %s'%coco_things_labels_fn)
            coco_things_df = pd.read_csv(coco_things_labels_fn, index_col=0)
            supcat_labels = np.array(coco_things_df)[:,0:12]
            animate_supcats = [1,9]
            inanimate_supcats = [ii for ii in range(12)\
                                 if ii not in animate_supcats]
            has_animate = np.any(np.array([supcat_labels[:,ii]==1 \
                                           for ii in animate_supcats]), axis=0)
            has_inanimate = np.any(np.array([supcat_labels[:,ii]==1 \
                                        for ii in inanimate_supcats]), axis=0)
            has_one_label = (has_animate.astype(int) + has_inanimate.astype(int))==1
            
            if args.trial_subset=='animate_only':                
                trials = has_animate & has_one_label
            elif args.trial_subset=='inanimate_only':         
                trials = has_inanimate & has_one_label
                
            trials_use += [trials]
           
        trials_use = np.array(trials_use).T
        
        
    elif 'small' in args.trial_subset or 'large' in args.trial_subset:
        
        trials_use = []
        for prf_model_index in range(n_prfs):
        
            if args.debug and prf_model_index>1:
                trials_use += [trials]
                continue
                
            size_labels_fn = os.path.join(labels_folder, \
                                  'S%d_realworldsize_prf%d.csv'%(args.subject, prf_model_index))
            if prf_model_index==0:
                print('loading from %s'%size_labels_fn)
            size_df = pd.read_csv(size_labels_fn, index_col=0)
            size_df = size_df.iloc[:,[0,2]]
           
            has_one_label = np.sum(np.array(size_df), axis=1)==1
            
            if args.trial_subset=='small_only':                
                trials = (np.array(size_df['has_small'])==1) & has_one_label
            elif args.trial_subset=='large_only':         
                trials = (np.array(size_df['has_large'])==1) & has_one_label
                
            trials_use += [trials]
           
        trials_use = np.array(trials_use).T

    # put into correct order for different trial sets
    trn_trials_use = trials_use[trn_image_order,:]
    val_trials_use = trials_use[val_image_order,:]
        
    return trn_trials_use, val_trials_use

def load_model_residuals(args, sessions):

    subject_dir = os.path.join(default_paths.save_fits_path, 'S%02d'%args.subject)
    # load most recent file
    files_in_dir = os.listdir(os.path.join(subject_dir, args.residuals_model_name))   
    if args.debug:
        my_dates = [f for f in files_in_dir if 'ipynb' not in f and 'DEBUG' in f]
        my_dates = [dd.split('_DEBUG')[0] for dd in my_dates]
    else:
        my_dates = [f for f in files_in_dir if 'ipynb' not in f and 'DEBUG' not in f]
    try:
        my_dates.sort(key=lambda date: datetime.strptime(date, "%b-%d-%Y_%H%M_%S"))
    except:
        my_dates.sort(key=lambda date: datetime.strptime(date, "%b-%d-%Y_%H%M"))
    most_recent_date = my_dates[-1]
    if args.debug:
        most_recent_date += '_DEBUG'
    residuals_dir = os.path.join(subject_dir,args.residuals_model_name,'%s'%most_recent_date)
     
    fn2load = os.path.join(residuals_dir, 'residuals_all_trials.npy')
    print('Loading single trial residuals from %s'%fn2load)
    
    out = np.load(fn2load, allow_pickle=True).item()
    assert(out['model_name']==args.residuals_model_name)
    assert(out['average_image_reps']==args.average_image_reps)
    
    voxel_data = out['residuals']
    image_order = out['image_order']
    val_inds = out['val_inds']
    session_inds = out['session_inds']
    
    print('shape of residual voxel data is:')
    print(voxel_data.shape)
    
    # now double check that the right set of trials are here
    sessions_using = sessions[(sessions+1)<=nsd_utils.max_sess_each_subj[args.subject-1]]
    image_order_expected = nsd_utils.get_master_image_order()
    session_inds_expected = nsd_utils.get_session_inds_full()
    inds2use = np.isin(session_inds_expected, sessions_using)
    image_order_expected = image_order_expected[inds2use]
    session_inds_expected = session_inds_expected[inds2use]

    if args.average_image_reps:
        n_trials_expected = len(np.unique(image_order))
        assert(voxel_data.shape[0]==n_trials_expected)
        assert(np.all(np.unique(image_order_expected)==image_order))
    else:
        n_trials_expected = len(sessions_using)*nsd_utils.trials_per_sess
        assert(voxel_data.shape[0]==n_trials_expected)
        assert(np.all(session_inds==session_inds_expected))
        assert(np.all(image_order==image_order_expected))
    
    return voxel_data, image_order, val_inds, session_inds, fn2load



def save_model_residuals(voxel_data, voxel_data_pred, output_dir, model_name, \
                         image_order, val_inds, session_inds, \
                         all_dat_r2, args):
    
    residuals = voxel_data - voxel_data_pred
    residuals = residuals.astype(np.float32)
    
    fn2save = os.path.join(output_dir, 'residuals_all_trials.npy')
    print('Saving single trial residuals to %s'%fn2save)
    
    np.save(fn2save,{'residuals':residuals, \
                     'model_name':model_name, \
                    'image_order':image_order, \
                    'val_inds':val_inds, \
                    'session_inds':session_inds, \
                    'all_dat_r2': all_dat_r2, \
                    'average_image_reps': args.average_image_reps})