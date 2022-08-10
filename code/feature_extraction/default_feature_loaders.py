import numpy as np
import time, h5py
from utils import default_paths
from feature_extraction import fwrf_features

def get_feature_loaders(subjects, feature_type, which_prf_grid=5):

    """
    Code to quickly create feature loader objects (from fwrf_features.fwrf_feature_loader)
    with some default params
    """

    if feature_type=='gabor_solo':
        path_to_load = default_paths.gabor_texture_feat_path
        feat_loaders = [fwrf_features.fwrf_feature_loader(subject=ss,\
                                                        which_prf_grid=which_prf_grid,\
                                                        feature_type='gabor_solo', \
                                                        n_ori=12, n_sf=8, nonlin=True) for ss in subjects]
    elif 'pyramid_texture' in feature_type:
        path_to_load = default_paths.pyramid_texture_feat_path
        if feature_type=='pyramid_texture_ll': 
            include_ll=True
            include_hl=False
            use_pca_feats_hl = False
        elif feature_type=='pyramid_texture_hl':
            include_ll=False
            include_hl=True
            use_pca_feats_hl = False
        elif feature_type=='pyramid_texture_hl_pca':
            assert(len(subjects)==1) # since these features are pca-ed within subject, can't concatenate.
            include_ll=False
            include_hl=True
            use_pca_feats_hl = True
        feat_loaders = [fwrf_features.fwrf_feature_loader(subject=ss,\
                                                        which_prf_grid=which_prf_grid, \
                                                        feature_type='pyramid_texture',\
                                                        n_ori=4, n_sf=4,\
                                                        pca_type='pcaHL') for ss in subjects]       
 
    elif 'sketch_tokens' in feature_type:
        path_to_load = default_paths.sketch_token_feat_path
        if 'residuals' in feature_type:
            use_residual_st_feats=True
        else:
            use_residual_st_feats=False
        feat_loaders = [fwrf_features.fwrf_feature_loader(subject=ss,\
                                                        which_prf_grid=which_prf_grid, \
                                                        feature_type='sketch_tokens',\
                                                        use_pca_feats = False, \
                                                        use_residual_st_feats=use_residual_st_feats) \
                                                            for ss in subjects]
    elif 'color' in feature_type:
        path_to_load = default_paths.color_feat_path
        feat_loaders = [fwrf_features.fwrf_feature_loader(subject=ss,\
                                                        which_prf_grid=which_prf_grid, \
                                                        feature_type='color') \
                                                            for ss in subjects]

    elif 'alexnet' in feature_type:
        assert(len(subjects)==1) # since these features are pca-ed within subject, can't concatenate.
        path_to_load = default_paths.alexnet_feat_path
        # if layer_name is None or layer_name=='':
        layer_name='Conv5_ReLU'
        blurface = 'blurface' in feature_type
        feat_loaders = [fwrf_features.fwrf_feature_loader(subject=ss,\
                                                        which_prf_grid=which_prf_grid, \
                                                        feature_type='alexnet',layer_name=layer_name,\
                                                        use_pca_feats = True, padding_mode = 'reflect', \
                                                        blurface=blurface) \
                                                        for ss in subjects]

    elif 'clip' in feature_type or 'resnet' in feature_type:
        assert(len(subjects)==1) # since these features are pca-ed within subject, can't concatenate.
        path_to_load = default_paths.clip_feat_path
        # if layer_name is None or layer_name=='':
        layer_name='block15'
        if 'clip' in feature_type:
            training_type='clip'
        elif 'blurface' in feature_type:
            training_type='blurface'
        else:
            training_type='imgnet'
        feat_loaders = [fwrf_features.fwrf_feature_loader(subject=ss,\
                                                        which_prf_grid=which_prf_grid, \
                                                        feature_type='resnet',layer_name=layer_name,\
                                                        model_architecture='RN50',use_pca_feats=True, \
                                                        training_type=training_type) \
                                                        for ss in subjects]

    else:
        raise RuntimeError('feature type %s not recognized'%feature_type)
    
    return feat_loaders, path_to_load
