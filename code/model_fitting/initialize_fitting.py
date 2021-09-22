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
 
    
def get_pyramid_model_name(ridge, n_ori, n_sf, do_pca_hl=False):

    if ridge==True:       
        # ridge regression, testing several positive lambda values (default)
        if do_pca_hl:
            model_name = 'texture_pyramid_pcaHL_ridge_%dori_%dsf'%(n_ori, n_sf)
        else:
            model_name = 'texture_pyramid_ridge_%dori_%dsf'%(n_ori, n_sf)
    else:        
        # fixing lambda at zero, so it turns into ordinary least squares
        if do_pca_hl:
            model_name = 'texture_pyramid_pcaHL_OLS_%dori_%dsf'%(n_ori, n_sf)
        else:       
            model_name = 'texture_pyramid_OLS_%dori_%dsf'%(n_ori, n_sf)
        
    return model_name

def get_gabor_texture_model_name(ridge, n_ori, n_sf):
    
    if ridge==True:       
        # ridge regression, testing several positive lambda values (default)
        model_name = 'texture_gabor_ridge_%dori_%dsf'%(n_ori, n_sf)
    else:        
        # fixing lambda at zero, so it turns into ordinary least squares
        model_name = 'texture_gabor_OLS_%dori_%dsf'%(n_ori, n_sf)

    return model_name

def get_gabor_solo_model_name(ridge, n_ori, n_sf):
    
    if ridge==True:       
        # ridge regression, testing several positive lambda values (default)
        model_name = 'gabor_solo_ridge_%dori_%dsf'%(n_ori, n_sf)
    else:        
        # fixing lambda at zero, so it turns into ordinary least squares
        model_name = 'gabor_solo_OLS_%dori_%dsf'%(n_ori, n_sf)

    return model_name

def get_sketch_tokens_model_name(do_pca):

    if do_pca==True:       
        model_name = 'sketch_tokens_pca'
    else:        
        model_name = 'sketch_tokens'
    
    return model_name


def get_bdcn_model_name(do_pca, map_ind):

    if do_pca==True:       
        # ridge regression, testing several positive lambda values (default)
        model_name = 'bdcn_pca'
    else:        
        # fixing lambda at zero, so it turns into ordinary least squares
        model_name = 'bdcn'
    
    if map_ind==-1:
        model_name += '_fused'
    else:
        model_name += '_map%d'%map_ind
        
    return model_name

        
    
def get_fitting_pars(trn_voxel_data, zscore_features=True, ridge=True, holdout_pct=0.10):

    holdout_size = int(np.ceil(np.shape(trn_voxel_data)[0]*holdout_pct))

    if ridge==True:
        if zscore_features==True:
            lambdas = np.logspace(0.,5.,9, dtype=np.float32)
        else:
            lambdas = np.logspace(-6., 1., 9).astype(np.float64)
    else:
        # putting in two zeros because the code might break with a singleton dimension for lambdas.
        lambdas = np.array([0.0,0.0])
        
    print('\nPossible lambda values are:')
    print(lambdas)

    return holdout_size, lambdas


def get_prf_models(aperture_rf_range=1.1):

    # sizes/centers; hard coded, taken from OSF code for fwrf fitting
    
    aperture = np.float32(1)
    smin, smax = np.float32(0.04), np.float32(0.4)
    n_sizes = 8

    # models is three columns, x, y, sigma
    models = prf_utils.model_space_pyramid(prf_utils.logspace(n_sizes)(smin, smax), min_spacing=1.4, aperture=aperture_rf_range*aperture)  
    print('most extreme RF positions:')
    print(models[0,:])
    print(models[-1,:])
    
    return aperture, models

def get_gabor_feature_map_fn(n_ori, n_sf, padding_mode, device, nonlin_fn):
    
    """
    Creating first-level feature extractor modules for the Gabor models.
    If using 'gabor_solo' mode, then only the mean activations for 'complex' module here is actually used.
    """
    _gabor_ext_complex = gabor_feature_extractor.gabor_extractor_multi_scale(n_ori=n_ori, n_sf=n_sf, sf_range_cyc_per_stim = (3, 72), \
                                                                             log_spacing = True, \
                 pix_per_cycle=4.13, cycles_per_radius=0.7, radii_per_filter=4, complex_cell=True, \
                                         padding_mode = 'circular', RGB=False, device = device)

    _gabor_ext_simple = gabor_feature_extractor.gabor_extractor_multi_scale(n_ori=n_ori, n_sf=n_sf, sf_range_cyc_per_stim = (3, 72), \
                                                                            log_spacing = True,
                 pix_per_cycle=4.13, cycles_per_radius=0.7, radii_per_filter=4, complex_cell=False, \
                                        padding_mode = padding_mode, RGB=False, device = device)
    
    if nonlin_fn:
        # adding a nonlinearity to the filter activations
        print('\nAdding log(1+sqrt(x)) as nonlinearity fn...')
        _fmaps_fn_complex = add_nonlinearity(_gabor_ext_complex, lambda x: torch.log(1+torch.sqrt(x)))
        _fmaps_fn_simple = add_nonlinearity(_gabor_ext_simple, lambda x: torch.log(1+torch.sqrt(x)))       
    else:
        _fmaps_fn_complex = _gabor_ext_complex
        _fmaps_fn_simple = _gabor_ext_simple
    
    return  _gabor_ext_complex, _gabor_ext_simple, _fmaps_fn_complex, _fmaps_fn_simple
