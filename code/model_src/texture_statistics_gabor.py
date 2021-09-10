import numpy as np
import torch
import time
from collections import OrderedDict
import torch.nn as nn

from utils import numpy_utils, torch_utils, texture_utils, prf_utils
from model_src import fwrf_fit as fwrf_fit

class texture_feature_extractor(nn.Module):
    """
    Module to compute higher-order texture statistics of input images (similar to Portilla & Simoncelli style texture model), 
    within specified area of space.
    Builds off lower-level feature maps for various orientation/spatial frequency bands, extracted using the modules specified 
    in '_fmaps_fn_complex' and '_fmaps_fn_simple' (which should be Gabor filtering modules)
    Can specify different subsets of features to include/exclude (i.e. pixel-level stats, simple/complex cells, cross-correlations, 
    auto-correlations)
    
    Inputs to the forward pass are images and pRF parameters of interest [x,y,sigma].
    """
    def __init__(self,_fmaps_fn_complex, _fmaps_fn_simple, sample_batch_size=100, \
                 autocorr_output_pix=3, n_prf_sd_out=2, aperture=1.0, \
                 feature_types_exclude=None,  do_varpart=False, group_all_hl_feats=False, device=None):
        
        super(texture_feature_extractor, self).__init__()
        
        self.fmaps_fn_complex = _fmaps_fn_complex
        self.fmaps_fn_simple = _fmaps_fn_simple
#         dtype = torch_utils.get_value(next(_fmaps_fn_complex.parameters())).dtype 
        self.n_sf = _fmaps_fn_simple.n_sf
        self.n_ori = _fmaps_fn_simple.n_ori
        self.n_phases = _fmaps_fn_simple.n_phases

        self.sample_batch_size = sample_batch_size
        self.autocorr_output_pix = autocorr_output_pix
        self.n_prf_sd_out = n_prf_sd_out
        self.aperture = aperture
        self.device = device      
        
        self.do_varpart = do_varpart
        self.group_all_hl_feats = group_all_hl_feats       
        
        self.feature_types_exclude = feature_types_exclude
        self.update_feature_list()
        self.do_pca = False
        
        
    def init_for_fitting(self, image_size, models=None, dtype=None):

        """
        Additional initialization operations.
        """
        # These two methods make sure that the 'resolutions_each_sf' property of the two feature extractors
        # are populated with the correct feature maps sizes for this image size.
        self.fmaps_fn_complex.get_fmaps_sizes(image_size)
        self.fmaps_fn_simple.get_fmaps_sizes(image_size)      
        self.max_features = self.n_features_total            

    def update_feature_list(self):
        
        feature_types_all = ['pixel_stats', 'complex_feature_means', 'simple_feature_means',\
                         'complex_feature_autocorrs','simple_feature_autocorrs',\
                         'complex_within_scale_crosscorrs','simple_within_scale_crosscorrs',\
                         'complex_across_scale_crosscorrs','simple_across_scale_crosscorrs']
       
        ori_pairs = np.vstack([[[oo1, oo2] for oo2 in np.arange(oo1+1, self.n_ori)] for oo1 in range(self.n_ori) if oo1<self.n_ori-1])
        n_ori_pairs = np.shape(ori_pairs)[0]
        feature_type_dims = [4,self.n_ori*self.n_sf, self.n_ori*self.n_sf*self.n_phases, \
                              self.n_ori*self.n_sf*self.autocorr_output_pix**2, self.n_ori*self.n_sf*self.n_phases*self.autocorr_output_pix**2, \
                              self.n_sf*n_ori_pairs, self.n_sf*n_ori_pairs*self.n_phases, (self.n_sf-1)*self.n_ori**2, (self.n_sf-1)*self.n_ori**2*self.n_phases]
        
        # decide which features to ignore, or use all features
        
        # a few shorthands for ignoring sets of features at a time
        if 'crosscorrs' in self.feature_types_exclude:
            self.feature_types_exclude.extend(['complex_within_scale_crosscorrs','simple_within_scale_crosscorrs','complex_across_scale_crosscorrs','simple_across_scale_crosscorrs'])
        if 'autocorrs' in self.feature_types_exclude:
            self.feature_types_exclude.extend(['complex_feature_autocorrs','simple_feature_autocorrs'])
        if 'pixel' in self.feature_types_exclude:
            self.feature_types_exclude.extend(['pixel_stats'])

        self.feature_types_include  = [ff for ff in feature_types_all if not ff in self.feature_types_exclude]
        if len(self.feature_types_include)==0:
            raise ValueError('you have specified too many features to exclude, and now you have no features left! aborting.')
            
        feature_dims_include = [feature_type_dims[fi] for fi in range(len(feature_type_dims)) if not feature_types_all[fi] in self.feature_types_exclude]
        # how many features will be needed, in total?
        self.n_features_total = np.sum(feature_dims_include)
        
        # numbers that define which feature types are in which column
        self.feature_column_labels = np.squeeze(np.concatenate([fi*np.ones([1,feature_dims_include[fi]]) for fi in range(len(feature_dims_include))], axis=1).astype('int'))
        assert(np.size(self.feature_column_labels)==self.n_features_total)
        
        if self.group_all_hl_feats and len(self.feature_types_exclude)==0:
            # In this case pretend there are just two groups of features - the 'mean_magnitudes' which are first-level gabor-like
            # and all other features combined into a second group. Makes it simpler to do variance partition analysis.
            # if do_varpart=False, this does nothing.
            self.feature_column_labels[self.feature_column_labels != 1] = -1
            self.feature_column_labels[self.feature_column_labels==1] = 0
            self.feature_column_labels[self.feature_column_labels==-1] = 1
            self.feature_group_names = ['mean_magnitudes', 'all_other_texture_feats']
        else:
            self.feature_group_names = self.feature_types_include

    def get_partial_versions(self):
        
        if not hasattr(self, 'max_features'):
            raise RuntimeError('need to run init_for_fitting first')
            
        n_feature_types = len(self.feature_group_names)
        partial_version_names = ['full_model'] 
        masks = np.ones([1,self.n_features_total])
        
        if self.do_varpart and n_feature_types>1:
            
            # "Partial versions" will be listed as: [full model, model w only first set of features, model w only second set, ...             
            partial_version_names += ['just_%s'%ff for ff in self.feature_group_names]
            masks2 = np.concatenate([np.expand_dims(np.array(self.feature_column_labels==ff).astype('int'), axis=0) for ff in np.arange(0,n_feature_types)], axis=0)
            masks = np.concatenate((masks, masks2), axis=0)
            
            if n_feature_types > 2:
                # if more than two types, also include models where we leave out first set of features, leave out second set of features...]
                partial_version_names += ['leave_out_%s'%ff for ff in self.feature_group_names]           
                masks3 = np.concatenate([np.expand_dims(np.array(self.feature_column_labels!=ff).astype('int'), axis=0) for ff in np.arange(0,n_feature_types)], axis=0)
                masks = np.concatenate((masks, masks3), axis=0)           
        
        # masks always goes [n partial versions x n total features]
        return masks, partial_version_names
        
    def clear_maps(self):
        
        """
        Note this doesn't really do much here, but this method needs to exist for this module to work w fitting code.
        """
        print('Clear maps fn')
        
        
    def forward(self, images, prf_params, prf_model_index, fitting_mode=True):
        
        if isinstance(prf_params, torch.Tensor):
            prf_params = torch_utils.get_value(prf_params)
        assert(np.size(prf_params)==3)
        prf_params = np.squeeze(prf_params)
        if isinstance(images, torch.Tensor):
            images = torch_utils.get_value(images)
            
        if not hasattr(self.fmaps_fn_simple, 'resolutions_each_sf'):
            raise RuntimeError('Need to run init_for_fitting first')
                
       
        if 'pixel_stats' in self.feature_types_include:
            print('Computing pixel-level statistics...')    
            t=time.time()
            x,y,sigma = prf_params
            n_pix=np.shape(images)[2]
            g = prf_utils.make_gaussian_mass_stack([x], [y], [sigma], n_pix=n_pix, size=self.aperture, dtype=np.float32)
            spatial_weights = g[2][0]
            wmean, wvar, wskew, wkurt = texture_utils.get_weighted_pixel_features(images, spatial_weights, device=self.device)
            pix_feat = torch.cat((wmean, wvar, wskew, wkurt), axis=1)
            elapsed =  time.time() - t
            print('time elapsed = %.5f'%elapsed)
        else:
            pix_feat = None
            
        if 'complex_feature_means' in self.feature_types_include:
            print('Computing complex cell features...')
            t = time.time()
            complex_feature_means = get_avg_features_in_prf(self.fmaps_fn_complex, images, prf_params,\
                                                            sample_batch_size=self.sample_batch_size, \
                                                            aperture=self.aperture, device=self.device, to_numpy=False)
            elapsed =  time.time() - t
            print('time elapsed = %.5f'%elapsed)
        else:
            complex_feature_means = None
            
        if 'simple_feature_means' in self.feature_types_include:
            print('Computing simple cell features...')
            t = time.time()
            simple_feature_means = get_avg_features_in_prf(self.fmaps_fn_simple, images,  prf_params,\
                                                           sample_batch_size=self.sample_batch_size, \
                                                           aperture=self.aperture,  device=self.device, to_numpy=False)
            elapsed =  time.time() - t
            print('time elapsed = %.5f'%elapsed)
        else:
            simple_feature_means = None
            
        # To save time, decide now whether any autocorrelation or cross-correlation features are desired. If not, will skip a bunch of the slower computations.     
        self.include_crosscorrs = np.any(['crosscorr' in ff for ff in self.feature_types_include])
        self.include_autocorrs = np.any(['autocorr' in ff for ff in self.feature_types_include])
        
        if self.include_autocorrs and self.include_crosscorrs:
            print('Computing higher order correlations...')
        elif self.include_crosscorrs:
            print('Computing higher order correlations (SKIPPING AUTOCORRELATIONS)...')
        elif self.include_autocorrs:
            print('Computing higher order correlations (SKIPPING CROSSCORRELATIONS)...')
        else:
            print('SKIPPING HIGHER-ORDER CORRELATIONS...')    
        t = time.time()
        complex_feature_autocorrs, simple_feature_autocorrs, \
        complex_within_scale_crosscorrs, simple_within_scale_crosscorrs, \
        complex_across_scale_crosscorrs, simple_across_scale_crosscorrs = get_higher_order_features(self.fmaps_fn_complex, self.fmaps_fn_simple, \
                                                                                                    images, prf_params=prf_params, 
                                                                                                    sample_batch_size=self.sample_batch_size, \
                                                                                                    include_autocorrs=self.include_autocorrs, \
                                                                                                    include_crosscorrs=self.include_crosscorrs, 
                                                                                                    autocorr_output_pix=self.autocorr_output_pix, \
                                                                                                    n_prf_sd_out=self.n_prf_sd_out, 
                                                                                                    aperture=self.aperture,  device=self.device)
        elapsed =  time.time() - t
        print('time elapsed = %.5f'%elapsed)

        all_feat = OrderedDict({'pixel_stats': pix_feat, 'complex_feature_means':complex_feature_means, 'simple_feature_means':simple_feature_means, 
                    'complex_feature_autocorrs': complex_feature_autocorrs, 'simple_feature_autocorrs': simple_feature_autocorrs, 
                    'complex_within_scale_crosscorrs': complex_within_scale_crosscorrs, 'simple_within_scale_crosscorrs':simple_within_scale_crosscorrs,
                    'complex_across_scale_crosscorrs': complex_across_scale_crosscorrs, 'simple_across_scale_crosscorrs':simple_across_scale_crosscorrs})

        feature_names_full = list(all_feat.keys())
        feature_names = [fname for fname in feature_names_full if fname in self.feature_types_include]
        assert(feature_names==self.feature_types_include) # double check here that the order is correct
        
        for ff, feature_name in enumerate(feature_names):   
            assert(all_feat[feature_name] is not None)
            if ff==0:
                all_feat_concat = all_feat[feature_name]
            else:               
                all_feat_concat = torch.cat((all_feat_concat, all_feat[feature_name]), axis=1)

        assert(all_feat_concat.shape[1]==self.n_features_total)
        print('Final size of features concatenated is [%d x %d]'%(all_feat_concat.shape[0], all_feat_concat.shape[1]))
        print('Feature types included are:')
        print(feature_names)

        if torch.any(torch.isnan(all_feat_concat)):
            print('\nWARNING THERE ARE NANS IN FEATURES MATRIX\n')

        feature_inds_defined = np.ones((self.n_features_total,), dtype=bool)
            
        return all_feat_concat, feature_inds_defined
    
    

def get_avg_features_in_prf(_fmaps_fn, images, prf_params, sample_batch_size, aperture, device, to_numpy=True):
    
    """
    For a given set of images and a specified pRF position and size, compute the mean (weighted by pRF)
    in each feature map channel. Returns [nImages x nFeatures]
    This could be done inside the get_higher_order_features fn, but it is nice to keep them separate in case
    we just want to run this (faster) part.
    """
    
    dtype = images.dtype.type    
    x,y,sigma = prf_params
    n_trials = images.shape[0]
    n_features = _fmaps_fn.n_features
    fmaps_rez = _fmaps_fn.resolutions_each_sf

    features = np.zeros(shape=(n_trials, n_features), dtype=dtype)
    if to_numpy==False:
         features = torch_utils._to_torch(features, device=device)

    # Define the RF for this "model" version - at several resolutions.
    _prfs = [torch_utils._to_torch(prf_utils.make_gaussian_mass(x, y, sigma, n_pix, size=aperture, \
                              dtype=dtype)[2], device=device) for n_pix in fmaps_rez]

    # To make full design matrix for all trials, first looping over trials in batches to get the features
    # Only reason to loop is memory constraints, because all trials is big matrices.
    t = time.time()
    n_batches = np.ceil(n_trials/sample_batch_size)
    bb=-1
    for rt,rl in numpy_utils.iterate_range(0, n_trials, sample_batch_size):

        bb=bb+1

        # Multiplying feature maps by RFs here. 
        # Feature maps in _fm go [nTrials x nFeatures(orientations) x nPixels x nPixels]
        # Spatial RFs in _prfs go [nPixels x nPixels]
        # Once we multiply, get [nTrials x nFeatures]
        # note this is concatenating SFs together from low (smallest maps) to high (biggest maps). 
        # Cycles through all orient channels in order for first SF, then again for next SF, etc.
        _features = torch.cat([torch.tensordot(_fm, _prf, dims=[[2,3], [0,1]]) \
                               for _fm,_prf in zip(_fmaps_fn(torch_utils._to_torch(images[rt], \
                                       device=device)), _prfs)], dim=1) # [#samples, #features]

        # Add features for this batch to full design matrix over all trials
        if to_numpy:
            features[rt] = torch_utils.get_value(_features)
        else:
            features[rt] = _features

        elapsed = time.time() - t

    return features

 

def get_higher_order_features(_fmaps_fn_complex, _fmaps_fn_simple, images, prf_params, sample_batch_size=20, include_autocorrs=True, include_crosscorrs=True, autocorr_output_pix=7, n_prf_sd_out=2, aperture=1.0, device=None):

    """
    Compute all higher-order features (cross-spatial and cross-feature correlations) for a batch of images.
    Input the functions that define first level feature maps (simple and complex cells), and prf parameters.
    Returns arrays of each higher order feature.    
    """
    
    if device is None:
        device = torch.device('cpu:0')    
        
    n_trials = np.shape(images)[0]
    
    assert(np.mod(autocorr_output_pix,2)==1) # must be odd!

    n_features_simple = _fmaps_fn_simple.n_features
    n_features_complex = _fmaps_fn_complex.n_features 
    fmaps_rez = _fmaps_fn_simple.resolutions_each_sf
    
    n_sf = len(fmaps_rez)
    n_ori = int(n_features_complex/n_sf)
    n_phases = 2
    
    # all pairs of different orientation channels.
    ori_pairs = np.vstack([[[oo1, oo2] for oo2 in np.arange(oo1+1, n_ori)] for oo1 in range(n_ori) if oo1<n_ori-1])
    n_ori_pairs = np.shape(ori_pairs)[0]

    if include_autocorrs:
        complex_feature_autocorrs = torch.zeros([n_trials, n_sf, n_ori, autocorr_output_pix**2], device=device)
        simple_feature_autocorrs = torch.zeros([n_trials, n_sf, n_ori, n_phases, autocorr_output_pix**2], device=device)
    else:
        complex_feature_autocorrs = None
        simple_feature_autocorrs = None
    
    if include_crosscorrs:
        complex_within_scale_crosscorrs = torch.zeros([n_trials, n_sf, n_ori_pairs], device=device)
        simple_within_scale_crosscorrs = torch.zeros([n_trials, n_sf, n_phases, n_ori_pairs], device=device)
        complex_across_scale_crosscorrs = torch.zeros([n_trials, n_sf-1, n_ori, n_ori], device=device)
        simple_across_scale_crosscorrs = torch.zeros([n_trials, n_sf-1, n_phases, n_ori, n_ori], device=device) # only done for pairs of neighboring SF.
    else:
        complex_within_scale_crosscorrs = None
        simple_within_scale_crosscorrs = None
        complex_across_scale_crosscorrs = None
        simple_across_scale_crosscorrs = None
        
    if include_autocorrs or include_crosscorrs:
        
        x,y,sigma = prf_params

        bb=-1
        for batch_inds, batch_size_actual in numpy_utils.iterate_range(0, n_trials, sample_batch_size):
            bb=bb+1

            fmaps_complex = _fmaps_fn_complex(torch_utils._to_torch(images[batch_inds],device=device))   
            fmaps_simple =  _fmaps_fn_simple(torch_utils._to_torch(images[batch_inds],device=device))

            # First looping over frequency (scales)
            for ff in range(n_sf):

                # Scale specific things - get the prf at this resolution of interest
                n_pix = fmaps_rez[ff]
                g = prf_utils.make_gaussian_mass_stack([x], [y], [sigma], n_pix=n_pix, size=aperture, dtype=np.float32)
                spatial_weights = g[2][0]

                patch_bbox_rect = texture_utils.get_bbox_from_prf(prf_params, spatial_weights.shape, n_prf_sd_out, force_square=False)
                # for autocorrelation, forcing the input region to be square
                patch_bbox_square = texture_utils.get_bbox_from_prf(prf_params, spatial_weights.shape, n_prf_sd_out, force_square=True)

                # Loop over orientation channels
                xx=-1
                for oo1 in range(n_ori):       


                    # Simple cell responses - loop over two phases per orient.
                    for pp in range(n_phases):
                        filter_ind = n_phases*oo1+pp  # orients and phases are both listed in the same dimension of filters matrix               
                        simple1 = fmaps_simple[ff][:,filter_ind,:,:].view([batch_size_actual,1,n_pix,n_pix])

                        # Simple cell autocorrelations.
                        if include_autocorrs:
                            auto_corr = weighted_auto_corr_2d(simple1, spatial_weights, patch_bbox=patch_bbox_square, output_pix = autocorr_output_pix, subtract_patch_mean = True, enforce_size=True, device=device)
                            simple_feature_autocorrs[batch_inds,ff,oo1,pp,:] = torch.reshape(auto_corr, [batch_size_actual, autocorr_output_pix**2])

                    # Complex cell responses
                    complex1 = fmaps_complex[ff][:,oo1,:,:].view([batch_size_actual,1,n_pix,n_pix])

                    # Complex cell autocorrelation (correlation w spatially shifted versions of itself)
                    if include_autocorrs:
                        auto_corr = weighted_auto_corr_2d(complex1, spatial_weights, patch_bbox=patch_bbox_square, output_pix = autocorr_output_pix, subtract_patch_mean = True, enforce_size=True, device=device)       
                        complex_feature_autocorrs[batch_inds,ff,oo1,:] = torch.reshape(auto_corr, [batch_size_actual, autocorr_output_pix**2])

                    if include_crosscorrs:
                        # Within-scale correlations - compare resp at orient==oo1 to responses at all other orientations, same scale.
                        for oo2 in np.arange(oo1+1, n_ori):            
                            xx = xx+1 
                            assert(oo1==ori_pairs[xx,0] and oo2==ori_pairs[xx,1])

                            complex2 = fmaps_complex[ff][:,oo2,:,:].view([batch_size_actual,1,n_pix,n_pix])      

                            # Complex cell within-scale cross correlations
                            cross_corr = weighted_cross_corr_2d(complex1, complex2, spatial_weights, patch_bbox=patch_bbox_rect, subtract_patch_mean = True, device=device)

                            complex_within_scale_crosscorrs[batch_inds,ff,xx] = torch.squeeze(cross_corr);

                            # Simple cell within-scale cross correlations
                            for pp in range(n_phases):
                                filter_ind = n_phases*oo2+pp
                                simple2 = fmaps_simple[ff][:,filter_ind,:,:].view([batch_size_actual,1,n_pix,n_pix])

                                cross_corr = weighted_cross_corr_2d(simple1, simple2, spatial_weights, patch_bbox=patch_bbox_rect, subtract_patch_mean = True, device=device)
                                simple_within_scale_crosscorrs[batch_inds,ff,pp,xx] = torch.squeeze(cross_corr);

                        # Cross-scale correlations - for these we care about same ori to same ori, so looping over all ori.
                        # Only for neighboring scales, so the first level doesn't get one
                        if ff>0:

                            for oo2 in range(n_ori):

                                # Complex cell response for neighboring scale
                                complex2_neighborscale = fmaps_complex[ff-1][:,oo2,:,:].view([batch_size_actual,1,fmaps_rez[ff-1], -1])
                                # Resize so that it can be compared w current scale
                                complex2_neighborscale = torch.nn.functional.interpolate(complex2_neighborscale, [n_pix, n_pix], mode='bilinear', align_corners=True)

                                cross_corr = weighted_cross_corr_2d(complex1, complex2_neighborscale, spatial_weights, patch_bbox=patch_bbox_rect, subtract_patch_mean = True, device=device)
                                complex_across_scale_crosscorrs[batch_inds,ff-1, oo1, oo2] = torch.squeeze(cross_corr)

                                for pp in range(n_phases):
                                    filter_ind = n_phases*oo2+pp
                                    # Simple cell response for neighboring scale
                                    simple2_neighborscale = fmaps_simple[ff-1][:,filter_ind,:,:].view([batch_size_actual,1,fmaps_rez[ff-1], -1])
                                    simple2_neighborscale = torch.nn.functional.interpolate(simple2_neighborscale, [n_pix, n_pix], mode='bilinear', align_corners=True)

                                    cross_corr = weighted_cross_corr_2d(simple1, simple2_neighborscale, spatial_weights, patch_bbox=patch_bbox_rect, subtract_patch_mean = True, device=device)
                                    simple_across_scale_crosscorrs[batch_inds,ff-1, pp, oo1, oo2] = torch.squeeze(cross_corr)

    if include_crosscorrs:
        simple_within_scale_crosscorrs = torch.reshape(simple_within_scale_crosscorrs, [n_trials, -1])
        simple_across_scale_crosscorrs = torch.reshape(simple_across_scale_crosscorrs, [n_trials, -1])
        complex_within_scale_crosscorrs = torch.reshape(complex_within_scale_crosscorrs, [n_trials, -1])
        complex_across_scale_crosscorrs = torch.reshape(complex_across_scale_crosscorrs, [n_trials, -1])
    if include_autocorrs:
        simple_feature_autocorrs = torch.reshape(simple_feature_autocorrs, [n_trials, -1])
        complex_feature_autocorrs = torch.reshape(complex_feature_autocorrs, [n_trials, -1])

    return complex_feature_autocorrs, simple_feature_autocorrs, complex_within_scale_crosscorrs, simple_within_scale_crosscorrs, complex_across_scale_crosscorrs, simple_across_scale_crosscorrs



def weighted_auto_corr_2d(images, spatial_weights, patch_bbox=None, output_pix=None, subtract_patch_mean=False, enforce_size=False, device=None):

    """
    Compute autocorrelation of a batch of images, weighting the pixels based on the values in spatial_weights (could be for instance a pRF definition for a voxel).
    Can optionally specify a square patch of the image to compute over, based on "patch_bbox" params. Otherwise use whole image.
    Using fft method to compute, should be fast.
    Input parameters:
        patch_bbox: (optional) bounding box of the patch to use for this calculation. [xmin xmax ymin ymax], see get_bbox_from_prf
        output_pix: the size of the autocorrelation matrix output by this function. If this is an even number, the output size is this value +1. Achieved by cropping out the center of the final autocorrelation 
            matrix  (note that the full image patch is still used in computing the autocorrelation, but just the center values are returned).
            If None, then returns the full autocorrelation matrix (same size as image patch.)
        subtract_patch_mean: subtract weighted mean of image before computing autocorr?
        enforce_size: if image patch is smaller than desired output, should we pad w zeros so that it has to be same size?
    Returns:
        A matrix describing the correlation of the image and various spatially shifted versions of it.
    Note this version is slightly different than the one in texture_utils.
    """
    
    
    if device is None:
        device = torch.device('cpu:0')        
    if isinstance(images, np.ndarray):
        images = torch_utils._to_torch(images, device)
    if isinstance(spatial_weights, np.ndarray):
        spatial_weights = torch_utils._to_torch(spatial_weights, device)
            
    if len(np.shape(images))==2:
        # pretend the batch and channel dims exist, for 2D input only (3D won't work)
        single_image=True
        images = images.view([1,1,images.shape[0],-1])
    else:
        single_image=False
        
    # have to be same size
    assert(images.shape[2]==spatial_weights.shape[0] and images.shape[3]==spatial_weights.shape[1])
    # images is [batch_size x n_channels x nPix x nPix]
    batch_size = images.shape[0]
    n_channels = images.shape[1]    
   
    if patch_bbox is not None:    
        [xmin, xmax, ymin, ymax] = patch_bbox
        # first crop out the region of the image that's currently of interest
        images = images[:,:,xmin:xmax, ymin:ymax]
        # crop same region from spatial weights matrix
        spatial_weights = spatial_weights[xmin:xmax, ymin:ymax]

    # make sure these sum to 1
    if not torch.sum(spatial_weights)==0.0:
        spatial_weights = spatial_weights/torch.sum(spatial_weights)   
   
    spatial_weights = spatial_weights.view([1,1,spatial_weights.shape[0],-1]).expand([batch_size,n_channels,-1,-1]) # [batch_size x n_channels x nPix x nPix]    
    
    # compute autocorrelation of this image patch
    if subtract_patch_mean:

        wmean = torch.sum(torch.sum(images * spatial_weights, dim=3), dim=2) # size is [batch_size x 1]
        wmean = wmean.view([batch_size,-1,1,1]).expand([-1,-1,images.shape[2],images.shape[3]]) # [batch_size x n_channels x nPix x nPix]
        
        weighted_images = (images - wmean) * torch.sqrt(spatial_weights) # square root of the weights here because they will get squared again in next operation
        
        auto_corr = torch.fft.fftshift(torch.real(torch.fft.ifft2(torch.abs(torch.fft.fft2(weighted_images, dim=[2,3]))**2, dim=[2,3])), dim=[2,3]);
    else:
        weighted_images = images * torch.sqrt(spatial_weights)
        auto_corr = torch.fft.fftshift(torch.real(torch.fft.ifft2(torch.abs(torch.fft.fft2(weighted_images, dim=[2,3]))**2, dim=[2,3])), dim=[2,3]);

    if output_pix is not None:

        # crop out just the center region
        new_center = int(np.floor(auto_corr.shape[2]/2))
        n_pix_out = np.min([int(np.floor(output_pix/2)), np.min([new_center, auto_corr.shape[2]-new_center])])
        auto_corr = auto_corr[:,:,new_center-n_pix_out:new_center+n_pix_out+1, new_center-n_pix_out:new_center+n_pix_out+1]        
    
    if enforce_size and not (np.shape(auto_corr)[2]==output_pix or np.shape(auto_corr)[2]==output_pix+1):
        
        # just pad w zeros if want same size.
        pix_diff = output_pix - np.shape(auto_corr)[2]   
        auto_corr = torch.nn.functional.pad(auto_corr, [int(np.floor(pix_diff/2)), int(np.ceil(pix_diff/2)), int(np.floor(pix_diff/2)), int(np.ceil(pix_diff/2))], mode='constant', value=0)
        assert(np.shape(auto_corr)[2]==output_pix and np.shape(auto_corr)[3]==output_pix)

    if single_image:
        auto_corr = torch.squeeze(auto_corr)
        
    return auto_corr

def weighted_cross_corr_2d(images1, images2, spatial_weights, patch_bbox=None, subtract_patch_mean=True, device=None):

    """
    Compute cross-correlation of two identically-sized images, weighting the pixels based on the values in spatial_weights (could be for instance a pRF definition for a voxel).
    Can optionally specify a square patch of the image to compute over, based on "patch_bbox" params. Otherwise use whole image.
    Basically a dot product of image values.
    Input parameters:
        patch_bbox: (optional) bounding box of the patch to use for this calculation. [xmin xmax ymin ymax], see get_bbox_from_prf
        subtract_patch_mean: do you want to subtract the weighted mean of image patch before computing?
    Returns:
        A single value that captures correlation between images (zero spatial shift)
    Note this version is slightly different than the one in texture_utils.
    """
    
    if device is None:
        device = torch.device('cpu:0')  
    if isinstance(images1, np.ndarray):
        images1 = torch_utils._to_torch(images1, device)
    if isinstance(images2, np.ndarray):
        images2 = torch_utils._to_torch(images2, device)
    if isinstance(spatial_weights, np.ndarray):
        spatial_weights = torch_utils._to_torch(spatial_weights, device)      
    
    if len(np.shape(images1))==2:
        # pretend the batch and channel dims exist, for 2D input only (3D won't work)
        single_image=True
        images1 = images1.view([1,1,images1.shape[0],-1])
        images2 = images2.view([1,1,images2.shape[0],-1])
    else:
        single_image=False
        
    # have to be same size
    assert(images1.shape==images2.shape)
    assert(images1.shape[2]==spatial_weights.shape[0] and images1.shape[3]==spatial_weights.shape[1])
    assert(images2.shape[2]==spatial_weights.shape[0] and images2.shape[3]==spatial_weights.shape[1])
    # images is [batch_size x n_channels x nPix x nPix]
    batch_size = images1.shape[0]
    n_channels = images1.shape[1]
    

    if patch_bbox is not None:
        [xmin, xmax, ymin, ymax] = patch_bbox
        # first crop out the region of the image that's currently of interest
        images1 = images1[:,:,xmin:xmax, ymin:ymax]
        images2 = images2[:,:,xmin:xmax, ymin:ymax]
        # crop same region from spatial weights matrix
        spatial_weights = spatial_weights[xmin:xmax, ymin:ymax]
    
    # make sure the wts sum to 1
    if not torch.sum(spatial_weights)==0.0:
        spatial_weights = spatial_weights/torch.sum(spatial_weights)
    spatial_weights = spatial_weights.view([1,1,spatial_weights.shape[0],-1]).expand([batch_size,n_channels,-1,-1]) # [batch_size x n_channels x nPix x nPix]    
    
    # compute cross-correlation
    if subtract_patch_mean:
        # subtract mean of each weighted image patch and take their dot product.
        # this quantity is equal to weighted covariance (only true if mean-centered)
        wmean1 = torch.sum(torch.sum(images1 * spatial_weights, dim=3), dim=2) # size is [batch_size x 1]
        wmean1 = wmean1.view([batch_size,-1,1,1]).expand([-1,-1,images1.shape[2],images1.shape[3]]) # [batch_size x n_channels x nPix x nPix]
        wmean2 = torch.sum(torch.sum(images2 * spatial_weights, dim=3), dim=2) # size is [batch_size x 1]
        wmean2 = wmean2.view([batch_size,-1,1,1]).expand([-1,-1,images2.shape[2],images2.shape[3]]) # [batch_size x n_channels x nPix x nPix]
        weighted_images1 = (images1 - wmean1) * torch.sqrt(spatial_weights) # square root of the weights here because they will get squared again in dot product operation.
        weighted_images2 = (images2 - wmean2) * torch.sqrt(spatial_weights)

        cross_corr = torch.sum(torch.sum(weighted_images1 * weighted_images2, dim=3), dim=2)    

    else:
        # dot product of raw (weighted) values
        # this is closer to what scipy.signal.correlate2d will do (except this is weighted)
        weighted_images1 = images1 * torch.sqrt(spatial_weights)
        weighted_images2 = images2 * torch.sqrt(spatial_weights)
        cross_corr = torch.sum(torch.sum(weighted_images1 * weighted_images2, dim=3), dim=2)      
        
    if single_image:
        cross_corr = torch.squeeze(cross_corr)
        
    return cross_corr