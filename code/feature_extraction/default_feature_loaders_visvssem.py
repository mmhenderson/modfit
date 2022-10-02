import numpy as np
import time, h5py
from utils import default_paths
from feature_extraction import fwrf_features, merge_features

def get_feature_loaders(image_set, feature_type, which_prf_grid=5):

    """
    Code to quickly create feature loader objects (from fwrf_features.fwrf_feature_loader)
    with some default params
    """

    if feature_type=='gabor_solo':
        path_to_load = default_paths.gabor_texture_feat_path
        feat_loaders = fwrf_features.fwrf_feature_loader(image_set=image_set,\
                                                         use_noavg=True, 
                                                         pca_subject=1, 
                                                         which_prf_grid=which_prf_grid,\
                                                         feature_type='gabor_solo', \
                                                         n_ori=12, n_sf=8, nonlin=True)
    elif 'pyramid_texture' in feature_type:
        path_to_load = default_paths.pyramid_texture_feat_path
        feat_loaders = fwrf_features.fwrf_feature_loader(image_set=image_set, 
                                                        which_prf_grid=which_prf_grid, \
                                                        feature_type='pyramid_texture',\
                                                        n_ori=4, n_sf=4,\
                                                        pca_subject=1, 
                                                        pca_type='pcaHL')     
    elif 'color' in feature_type:
        path_to_load = default_paths.color_feat_path
        feat_loaders = fwrf_features.fwrf_feature_loader(image_set=image_set,
                                                        which_prf_grid=which_prf_grid, \
                                                        feature_type='color',\
                                                        pca_subject=1, 
                                                        use_noavg=True)       
 
    elif 'sketch_tokens' in feature_type:
        if which_prf_grid==0:
            use_noavg = True
            st_use_avgpool = False
            st_pooling_size = 60
        else:
            use_noavg = False
            st_use_avgpool = None
            st_pooling_size = None
        path_to_load = default_paths.sketch_token_feat_path
        feat_loaders = fwrf_features.fwrf_feature_loader(image_set=image_set,
                                                        which_prf_grid=which_prf_grid, \
                                                        feature_type='sketch_tokens',\
                                                        use_pca_feats = False, \
                                                        use_noavg=use_noavg,
                                                        st_use_avgpool=st_use_avgpool,
                                                        st_pooling_size=st_pooling_size,
                                                        use_grayscale_st_feats=True,
                                                        use_residual_st_feats=False)

    elif 'alexnet' in feature_type:
        
        blurface = 'blurface' in feature_type
        if blurface:
            path_to_load = default_paths.alexnet_blurface_feat_path
        else:
            path_to_load = default_paths.alexnet_feat_path
        floaders = []
        layer_names = ['Conv%d_ReLU'%ll for ll in np.arange(1,6)]
        layer_names += ['FC%d_ReLU'%ll for ll in [6,7]]
        for layer_name in layer_names:
              
            f = fwrf_features.fwrf_feature_loader(image_set=image_set,
                                                which_prf_grid=which_prf_grid, \
                                                feature_type='alexnet',layer_name=layer_name,\
                                                padding_mode = 'reflect', \
                                                pca_subject=1, 
                                                use_noavg=True, blurface=blurface)
            floaders.append(f)
        feat_loaders = merge_features.combined_feature_loader(floaders, layer_names)
        

    elif 'clip' in feature_type or 'resnet' in feature_type:
        
        if 'clip' in feature_type:
            path_to_load = default_paths.clip_feat_path
            training_type='clip'
        elif 'blurface' in feature_type:
            path_to_load = default_paths.resnet50_blurface_feat_path
            training_type='blurface'
        else:
            path_to_load = default_paths.resnet50_feat_path
            training_type='imgnet'
        floaders = []
        layer_names=['block%d'%ll for ll in [2,6,12,15]]
        for layer_name in layer_names:
        
            f = fwrf_features.fwrf_feature_loader(image_set=image_set,
                                                        which_prf_grid=which_prf_grid, \
                                                        feature_type='resnet',layer_name=layer_name,\
                                                        model_architecture='RN50',
                                                        use_noavg=True,
                                                        pca_subject=1, 
                                                        training_type=training_type)
            floaders.append(f)
        feat_loaders = merge_features.combined_feature_loader(floaders, layer_names)
        
    else:
        raise RuntimeError('feature type %s not recognized'%feature_type)
    
    return feat_loaders, path_to_load
