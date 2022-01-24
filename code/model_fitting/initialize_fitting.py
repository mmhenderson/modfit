
"""
These are all semi-general functions that are run before model fitting (file name to save, make feature extractors etc.)
"""

import torch
import time
import os
import numpy as np

# import custom modules
from feature_extraction import gabor_feature_extractor
from utils import prf_utils, default_paths

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
        elif 'gabor_texture' in ft:     
            fitting_types += [ft]
            model_name += 'texture_gabor'
            if args.ridge==True:   
                model_name += '_ridge'
            else:
                model_name += '_OLS'
            model_name+='_%dori_%dsf'%(args.n_ori_gabor, args.n_sf_gabor)
        elif 'gabor_solo' in ft:     
            fitting_types += [ft]
            model_name += 'gabor_solo'
            if args.ridge==True:   
                model_name += '_ridge'
            else:
                model_name += '_OLS'
            model_name+='_%dori_%dsf'%(args.n_ori_gabor, args.n_sf_gabor)
        elif 'sketch_tokens' in ft:      
            fitting_types += [ft]
            if args.use_pca_st_feats==True:       
                model_name += 'sketch_tokens_pca'
            elif args.use_lda_st_feats==True:
                model_name += 'sketch_tokens_lda_%s'%args.lda_discrim_type
            else:        
                model_name += 'sketch_tokens'
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

    print(fitting_types)
    print(model_name)
    
    return model_name, fitting_types


def get_save_path(subject, volume_space, model_name, shuffle_images, random_images, random_voxel_data, debug, date_str=None):
    
    # choose where to save results of fitting - always making a new file w current timestamp.
    
    # add these suffixes to the file name if it's one of the control analyses
    if shuffle_images==True:
        model_name = model_name + '_SHUFFLEIMAGES'
    if random_images==True:
        model_name = model_name + '_RANDOMIMAGES'
    if random_voxel_data==True:
        model_name = model_name + '_RANDOMVOXELDATA'
    
    if date_str is None:
        timestamp = time.strftime('%b-%d-%Y_%H%M_%S', time.localtime())
        make_new_folder = True
    else:
        # if you specified an existing timestamp, then won't try to make new folder, need to find existing one.
        timestamp = date_str
        make_new_folder = False
        
    print ("Time Stamp: %s" % timestamp)  
    save_fits_path = default_paths.save_fits_path
    if volume_space:
        subject_dir = os.path.join(save_fits_path, 'S%02d'%subject)
    else:
        subject_dir = os.path.join(save_fits_path,'S%02d_surface'%subject)
    if debug==True:
        output_dir = os.path.join(subject_dir,model_name,'%s_DEBUG/'%timestamp)
    else:
        output_dir = os.path.join(subject_dir,model_name,'%s'%timestamp)
    if not os.path.exists(output_dir): 
        if make_new_folder:
            os.makedirs(output_dir)
        else:
            raise ValueError('the path at %s does not exist yet!!'%output_dir)
        
    fn2save = os.path.join(output_dir,'all_fit_params')
    print('\nWill save final output file to %s\n'%output_dir)
    
    return output_dir, fn2save
 
def get_lambdas(zscore_features=True, ridge=True):
    
    if ridge==False:
        # putting in two zeros because the code might break with a singleton dimension for lambdas.
        lambdas = np.array([0.0,0.0])
        return lambdas
    
    if zscore_features==True:
        lambdas = np.logspace(np.log(0.01),np.log(10**5+0.01),9, dtype=np.float32, base=np.e) - 0.01
#         lambdas = np.logspace(np.log(0.01),np.log(10**7+0.01),10, dtype=np.float32, base=np.e) - 0.01
    else:
        # range of values are different if choosing not to z-score - note the performance of these lambdas
        # will vary depending on actual feature value ranges, be sure to check the results carefully
        lambdas = np.logspace(np.log(0.01),np.log(10**1+0.01),10, dtype=np.float32, base=np.e) - 0.01
        
    print('\nPossible lambda values are:')
    print(lambdas)
    
    return lambdas

def get_prf_models(which_grid=5):

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

    print('number of pRFs: %d'%len(models))
    print('most extreme RF positions:')
    print(models[0,:])
    print(models[-1,:])
    
    return models

def load_precomputed_prfs(subject):
    
    if subject==1:
        saved_prfs_fn=os.path.join(default_paths.save_fits_path,\
                'S01/alexnet_all_conv_pca/Nov-23-2021_2247_09/all_fit_params')    
    elif subject==2:
        saved_prfs_fn=os.path.join(default_paths.save_fits_path,\
                'S02/alexnet_all_conv_pca/Jan-07-2022_1815_05/all_fit_params')
    elif subject==3:
        saved_prfs_fn=os.path.join(default_paths.save_fits_path,\
                'S03/alexnet_all_conv_pca/Jan-11-2022_0342_58/all_fit_params')
    elif subject==4:
        saved_prfs_fn=os.path.join(default_paths.save_fits_path,\
                'S04/alexnet_all_conv_pca/Jan-13-2022_1805_02/all_fit_params')
    elif subject==5:
        saved_prfs_fn=os.path.join(default_paths.save_fits_path,\
                'S05/alexnet_all_conv_pca/Jan-15-2022_1936_46/all_fit_params')
    elif subject==6:
        saved_prfs_fn=os.path.join(default_paths.save_fits_path,\
                'S06/alexnet_all_conv_pca/Jan-19-2022_1358_01/all_fit_params')
    elif subject==7:
        saved_prfs_fn=os.path.join(default_paths.save_fits_path,\
                'S07/alexnet_all_conv_pca/Jan-21-2022_0313_37/all_fit_params')
    elif subject==8:
        saved_prfs_fn=os.path.join(default_paths.save_fits_path,\
                'S08/alexnet_all_conv_pca/Jan-22-2022_1508_21/all_fit_params')
    else:
        raise ValueError('trying to load pre-computed prfs, but prf params are not yet computed for this model')

    print('Loading pre-computed pRF estimates for all voxels from %s'%saved_prfs_fn)
    out = torch.load(saved_prfs_fn)
    best_model_each_voxel = out['best_params'][5][:,0]
    
    return best_model_each_voxel, saved_prfs_fn

def load_best_model_layers(subject, model):
    
    if subject==1:
        if model=='clip':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                 'S01/clip_RN50_all_resblocks_pca/Dec-12-2021_1407_50/all_fit_params')
        elif model=='alexnet':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                 'S01/alexnet_all_conv_pca/Nov-23-2021_2247_09/all_fit_params')    
    elif subject==2:
        if model=='clip':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                 'S02/clip_RN50_all_resblocks_pca/Jan-13-2022_1121_18/all_fit_params')
        elif model=='alexnet':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                'S02/alexnet_all_conv_pca/Jan-07-2022_1815_05/all_fit_params')
    elif subject==3:
        if model=='clip':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                 'S03/clip_RN50_all_resblocks_pca/Jan-18-2022_1156_04/all_fit_params')
        elif model=='alexnet':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                'S03/alexnet_all_conv_pca/Jan-11-2022_0342_58/all_fit_params')
    elif subject==4:
        if model=='clip':
            raise ValueError('for S%d %s, best model layer not computed yet'%(subject, model))
        elif model=='alexnet':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                'S04/alexnet_all_conv_pca/Jan-13-2022_1805_02/all_fit_params')
    elif subject==5:
        if model=='clip':
            raise ValueError('for S%d %s, best model layer not computed yet'%(subject, model))
        elif model=='alexnet':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                'S05/alexnet_all_conv_pca/Jan-15-2022_1936_46/all_fit_params')
    elif subject==6:
        if model=='clip':
            raise ValueError('for S%d %s, best model layer not computed yet'%(subject, model))
        elif model=='alexnet':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                'S06/alexnet_all_conv_pca/Jan-19-2022_1358_01/all_fit_params')
    elif subject==7:
        if model=='clip':
            raise ValueError('for S%d %s, best model layer not computed yet'%(subject, model))
        elif model=='alexnet':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                'S07/alexnet_all_conv_pca/Jan-21-2022_0313_37/all_fit_params')
    elif subject==8:
        if model=='clip':
            raise ValueError('for S%d %s, best model layer not computed yet'%(subject, model))
        elif model=='alexnet':
            saved_best_layer_fn=os.path.join(default_paths.save_fits_path,\
                'S08/alexnet_all_conv_pca/Jan-22-2022_1508_21/all_fit_params')
    else:
        raise ValueError('for S%d %s, best model layer not computed yet'%(subject, model))
    
    print('Loading best %s layer for all voxels from %s'%(model,saved_best_layer_fn))
    
    out = torch.load(saved_best_layer_fn)
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

def get_gabor_feature_map_fn(n_ori, n_sf, device, padding_mode='circular', nonlin_fn=False):
    
    """
    Creating first-level feature extractor modules for the Gabor models.
    If using 'gabor_solo' mode, then only the mean activations for 'complex' module here is actually used.
    """
    if nonlin_fn:
        # adding a nonlinearity to the filter activations
        print('\nAdding log(1+sqrt(x)) as nonlinearity fn...')
        nonlin = lambda x: torch.log(1+torch.sqrt(x))
    else:
        nonlin = None
        
    _gabor_ext_complex = gabor_feature_extractor.gabor_extractor_multi_scale(n_ori=n_ori, n_sf=n_sf, \
                             sf_range_cyc_per_stim = (3, 72), log_spacing = True, \
                             pix_per_cycle=4.13, cycles_per_radius=0.7, radii_per_filter=4, \
                             complex_cell=True, padding_mode = padding_mode, nonlin_fn=nonlin, \
                             RGB=False, device = device)

    _gabor_ext_simple = gabor_feature_extractor.gabor_extractor_multi_scale(n_ori=n_ori, n_sf=n_sf, \
                             sf_range_cyc_per_stim = (3, 72), log_spacing = True, \
                             pix_per_cycle=4.13, cycles_per_radius=0.7, radii_per_filter=4, \
                             complex_cell=False, padding_mode = padding_mode, nonlin_fn=nonlin, \
                             RGB=False, device = device)

    _fmaps_fn_complex = _gabor_ext_complex
    _fmaps_fn_simple = _gabor_ext_simple

    return  _gabor_ext_complex, _gabor_ext_simple, _fmaps_fn_complex, _fmaps_fn_simple
