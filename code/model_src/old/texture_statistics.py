import numpy as np
import torch
import time
from collections import OrderedDict
from utils import numpy_utility, torch_utils
from model_src import fwrf_fit

def compute_all_texture_features(_fmaps_fn_complex, _fmaps_fn_simple, images, prf_params, sample_batch_size=100, include_autocorrs=True, include_crosscorrs=True, autocorr_output_pix=3, n_prf_sd_out=2, aperture=1.0, device=None):

   
    if isinstance(prf_params, torch.Tensor):
        prf_params = torch_utils.get_value(prf_params)
    assert(np.size(prf_params)==3)
    prf_params = np.squeeze(prf_params)
    if isinstance(images, torch.Tensor):
        images = torch_utils.get_value(images)
   

    print('Computing pixel-level statistics...')    
    t=time.time()
    x,y,sigma = prf_params
    n_pix=np.shape(images)[2]
    g = numpy_utility.make_gaussian_mass_stack([x], [y], [sigma], n_pix=n_pix, size=aperture, dtype=np.float32)
    spatial_weights = g[2][0]
    wmean, wvar, wskew, wkurt = get_weighted_pixel_features(images, spatial_weights, device=device)
    elapsed =  time.time() - t
    print('time elapsed = %.5f'%elapsed)

    print('Computing complex cell features...')
    t = time.time()
    complex_feature_means = fwrf_fit.get_features_in_prf(prf_params, _fmaps_fn_complex , images=images, sample_batch_size=sample_batch_size, aperture=aperture, device=device, to_numpy=False)
    elapsed =  time.time() - t
    print('time elapsed = %.5f'%elapsed)

    print('Computing simple cell features...')
    t = time.time()
    simple_feature_means = fwrf_fit.get_features_in_prf(prf_params,  _fmaps_fn_simple, images=images, sample_batch_size=sample_batch_size, aperture=aperture,  device=device, to_numpy=False)
    elapsed =  time.time() - t
    print('time elapsed = %.5f'%elapsed)

    
    if include_autocorrs and include_crosscorrs:
        print('Computing higher order correlations...')
    elif include_crosscorrs:
        print('Computing higher order correlations (SKIPPING AUTOCORRELATIONS)...')
    elif include_autocorrs:
        print('Computing higher order correlations (SKIPPING CROSSCORRELATIONS)...')
    else:
        print('SKIPPING HIGHER-ORDER CORRELATIONS...')    
    t = time.time()
    complex_feature_autocorrs, simple_feature_autocorrs, \
    complex_within_scale_crosscorrs, simple_within_scale_crosscorrs, \
    complex_across_scale_crosscorrs, simple_across_scale_crosscorrs = get_higher_order_features(_fmaps_fn_complex, _fmaps_fn_simple, images, prf_params=prf_params, 
                                                                                                sample_batch_size=sample_batch_size, include_autocorrs=include_autocorrs, include_crosscorrs=include_crosscorrs, 
                                                                                                autocorr_output_pix=autocorr_output_pix, n_prf_sd_out=n_prf_sd_out, 
                                                                                                aperture=aperture,  device=device)
    elapsed =  time.time() - t
    print('time elapsed = %.5f'%elapsed)

    n_trials = len(wmean)
    if include_autocorrs and include_crosscorrs:
        all_feat = OrderedDict({'wmean': wmean, 'wvar':wvar, 'wskew':wskew, 'wkurt':wkurt, 'complex_feature_means':complex_feature_means, 'simple_feature_means':simple_feature_means, 
                'complex_feature_autocorrs': torch.reshape(complex_feature_autocorrs, [n_trials,-1]), 'simple_feature_autocorrs': torch.reshape(simple_feature_autocorrs, [n_trials,-1]), 
                'complex_within_scale_crosscorrs': torch.reshape(complex_within_scale_crosscorrs, [n_trials,-1]), 'simple_within_scale_crosscorrs': torch.reshape(simple_within_scale_crosscorrs, [n_trials,-1]),
                'complex_across_scale_crosscorrs': torch.reshape(complex_across_scale_crosscorrs, [n_trials,-1]), 'simple_across_scale_crosscorrs':torch.reshape(simple_across_scale_crosscorrs, [n_trials,-1])})
    elif include_crosscorrs:
        all_feat = OrderedDict({'wmean': wmean, 'wvar':wvar, 'wskew':wskew, 'wkurt':wkurt,  'complex_feature_means':complex_feature_means, 'simple_feature_means':simple_feature_means,                 
                'complex_within_scale_crosscorrs': torch.reshape(complex_within_scale_crosscorrs, [n_trials,-1]), 'simple_within_scale_crosscorrs': torch.reshape(simple_within_scale_crosscorrs, [n_trials,-1]),
                'complex_across_scale_crosscorrs': torch.reshape(complex_across_scale_crosscorrs, [n_trials,-1]), 'simple_across_scale_crosscorrs':torch.reshape(simple_across_scale_crosscorrs, [n_trials,-1])})
    elif include_autocorrs:
        all_feat = OrderedDict({'wmean': wmean, 'wvar':wvar, 'wskew':wskew, 'wkurt':wkurt, 'complex_feature_means':complex_feature_means, 'simple_feature_means':simple_feature_means, 
                'complex_feature_autocorrs': torch.reshape(complex_feature_autocorrs, [n_trials,-1]), 'simple_feature_autocorrs': torch.reshape(simple_feature_autocorrs, [n_trials,-1]) })
    else:
        all_feat = OrderedDict({'wmean': wmean, 'wvar':wvar, 'wskew':wskew, 'wkurt':wkurt, 'complex_feature_means':complex_feature_means, 'simple_feature_means':simple_feature_means})

    feature_names = list(all_feat.keys())

    for ff, feature_name in enumerate(feature_names): 
        if ff==0:
            all_feat_concat = all_feat[feature_name]
            feature_type_labels = ff*np.ones([1,all_feat[feature_name].shape[1]])
        else:               
            all_feat_concat = torch.cat((all_feat_concat, all_feat[feature_name]), axis=1)
            feature_type_labels = np.concatenate((feature_type_labels, ff*np.ones([1,all_feat[feature_name].shape[1]])), axis=1)
                
    print('Final size of features concatenated is [%d x %d]'%(all_feat_concat.shape[0], all_feat_concat.shape[1]))
    print('Feature types included are:')
    print(feature_names)
    
    if torch.any(torch.isnan(all_feat_concat)):
        print('\nWARNING THERE ARE NANS IN FEATURES MATRIX\n')
            
    return torch_utils.get_value(all_feat_concat), [feature_type_labels, feature_names]

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

    n_features_simple, fmaps_rez = fwrf_fit.get_fmaps_sizes(_fmaps_fn_simple, images[0:sample_batch_size], device)
    n_features_complex, fmaps_rez = fwrf_fit.get_fmaps_sizes(_fmaps_fn_complex, images[0:sample_batch_size], device)
    
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
        for batch_inds, batch_size_actual in numpy_utility.iterate_range(0, n_trials, sample_batch_size):
            bb=bb+1

            fmaps_complex = _fmaps_fn_complex(torch_utils._to_torch(images[batch_inds],device=device))   
            fmaps_simple =  _fmaps_fn_simple(torch_utils._to_torch(images[batch_inds],device=device))

            # First looping over frequency (scales)
            for ff in range(n_sf):

                # Scale specific things - get the prf at this resolution of interest
                n_pix = fmaps_rez[ff]
                g = numpy_utility.make_gaussian_mass_stack([x], [y], [sigma], n_pix=n_pix, size=aperture, dtype=np.float32)
                spatial_weights = g[2][0]

                patch_bbox_rect = get_bbox_from_prf(prf_params, spatial_weights.shape, n_prf_sd_out, force_square=False)
                # for autocorrelation, forcing the input region to be square
                patch_bbox_square = get_bbox_from_prf(prf_params, spatial_weights.shape, n_prf_sd_out, force_square=True)

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



    return complex_feature_autocorrs, simple_feature_autocorrs, complex_within_scale_crosscorrs, simple_within_scale_crosscorrs, complex_across_scale_crosscorrs, simple_across_scale_crosscorrs


def get_bbox_from_prf(prf_params, image_size, n_prf_sd_out=2, verbose=False, force_square=False):
    """
    For a given pRF center and size, calculate the square bounding box that captures a specified number of SDs from the center (default=2 SD)
    Returns [xmin, xmax, ymin, ymax]
    """
    x,y,sigma = prf_params
    n_pix = image_size[0]
    assert(image_size[1]==n_pix)
    assert(sigma>0 and n_prf_sd_out>0)
    
    # decide on the window to use for correlations, based on prf parameters. Patch goes # SD from the center (2 by default).
    # note this can't be < 1, even for the smallest choice of parameters (since rounding up). this way it won't be too small.
    pix_from_center = int(np.ceil(sigma*n_prf_sd_out*n_pix))

    # center goes [row ind, col ind]
    center = np.array((n_pix/2  - y*n_pix, x*n_pix + n_pix/2)) # note that the x/y dims get swapped here because of how pRF parameters are defined.

    # now defining the extent of the bbox. want to err on the side of making it too big, so taking floor/ceiling...
    xmin = int(np.floor(center[0]-pix_from_center))
    xmax = int(np.ceil(center[0]+pix_from_center))
    ymin = int(np.floor(center[1]-pix_from_center))
    ymax = int(np.ceil(center[1]+pix_from_center))

    # cropping it to within the image bounds. Can end up being a rectangle rather than square.
    [xmin, xmax, ymin, ymax] = np.maximum(np.minimum([xmin, xmax, ymin, ymax], n_pix), 0)

    # decide if we want square or are ok with a rectangle
    if force_square:
        minside = np.min([xmax-xmin, ymax-ymin])
        maxside = np.max([xmax-xmin, ymax-ymin])
        if minside!=maxside:

            if verbose:
                print('trimming bbox to make it square')
                print('original bbox was:')
                print([xmin, xmax, ymin, ymax])

            n2trim = [int(np.floor((maxside-minside)/2)), int(np.ceil((maxside-minside)/2))]

            if np.argmin([xmax-xmin, ymax-ymin])==0:
                ymin = ymin+n2trim[0]
                ymax = ymax-n2trim[1]
            else:
                xmin = xmin+n2trim[0]
                xmax = xmax-n2trim[1]

        assert((xmax-xmin)==(ymax-ymin))

    if verbose:
        print('final bbox will be:')
        print([xmin, xmax, ymin, ymax])
        
    # checking to see if the patch has become just one pixel. this can happen due to the cropping.
    # if this happens, cross-correlations will give zero.
    if ((xmax-xmin)<2 or (ymax-ymin)<2):
        print('Warning: your patch only has one pixel (for n_pix: %d and prf params: [%.2f, %.2f, %.2f])\n'%(n_pix,x,y,sigma))      
        
    return [xmin, xmax, ymin, ymax]


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



def get_weighted_pixel_features(image_batch, spatial_weights, device=None):
    """
    Compute mean, variance, skewness, kurtosis of luminance values for each of a batch of images.
    Input size is [batch_size x n_channels x npix x npix]
    Spatial weights describes a weighting function, [npix x npix]
    Returns [batch_size x n_channels] size array for each property.
    """
    
    if isinstance(image_batch, np.ndarray):
        image_batch = torch_utils._to_torch(image_batch, device)
    if isinstance(spatial_weights, np.ndarray):
        spatial_weights = torch_utils._to_torch(spatial_weights, device)
     
    assert(image_batch.shape[2]==spatial_weights.shape[0] and image_batch.shape[3]==spatial_weights.shape[1])
    assert(image_batch.shape[1]==1)
    
    batch_size = image_batch.shape[0]
    n_channels = image_batch.shape[1]
    n_pix = image_batch.shape[2]

    image_batch = image_batch.view([batch_size, n_channels, n_pix**2])
    spatial_weights = spatial_weights/torch.sum(spatial_weights)
    spatial_weights = spatial_weights.view([1,1,n_pix**2]).expand([batch_size,n_channels,-1]) # [batch_size x n_channels x nPix x nPix]    
   
    ims_weighted = image_batch * spatial_weights
   
    wmean = torch.sum(ims_weighted, axis=2).view([batch_size,-1,1])

    wvar = torch.sum(spatial_weights * (image_batch - wmean.expand([-1,-1,n_pix**2]))**2, axis=2).view([batch_size,-1,1])
    
    wskew = torch.sum(spatial_weights *(image_batch - wmean.expand([-1,-1,n_pix**2]))**3 / (wvar**(3/2)), axis=2).view([batch_size,-1,1])
    
    wkurt = torch.sum(spatial_weights *(image_batch - wmean.expand([-1,-1,n_pix**2]))**4 / (wvar**(2)), axis=2).view([batch_size,-1,1])
    
    # correct for nans/inf values which happen when variance is very small (denominator)
    wskew[torch.isnan(wskew)] = 0.0
    wkurt[torch.isnan(wkurt)] = 0.0
    wskew[torch.isinf(wskew)] = 0.0
    wkurt[torch.isinf(wkurt)] = 0.0
    
    return torch.squeeze(wmean, dim=2), torch.squeeze(wvar, dim=2), torch.squeeze(wskew, dim=2), torch.squeeze(wkurt, dim=2)



## OLD version

# def get_bbox_from_prf(prf_params, image_size, n_prf_sd_out=2, verbose=False):
#     """
#     For a given pRF center and size, calculate the square bounding box that captures a specified number of SDs from the center (default=2 SD)
#     Returns [xmin, xmax, ymin, ymax]
#     """
#     x,y,sigma = prf_params
#     n_pix = image_size[0]
#     assert(image_size[1]==n_pix)
#     assert(sigma>0 and n_prf_sd_out>0)
    
#     # decide on the window to use for correlations, based on prf parameters. Patch goes # SD from the center (2 by default).
#     pix_from_center = int(sigma*n_prf_sd_out*n_pix)
#     # center goes [row ind, col ind]
#     center = np.array((int(np.floor(n_pix/2  - y*n_pix)), int(np.floor(x*n_pix + n_pix/2)))) # note that the x/y dims get swapped here because of how pRF parameters are defined.
#     # ensure that the patch never tries to go outside image bounds...at the corners it initially will be outside. 
#     # TODO: figure out what to do with these edge cases, because padding will probably introduce artifacts. 
#     # For now just reducing the size of the region so that it's a square with all corners inside image bounds. 
#     center = np.minimum(np.maximum(center,0), n_pix-1) 
   
#     mindist2edge=np.min([n_pix-center[0]-1, n_pix-center[1]-1, center[0], center[1]])  
#     pix_from_center = np.minimum(pix_from_center, mindist2edge)    
#     if pix_from_center==0 and verbose:
#         print('Warning: your patch only has one pixel (for n_pix: %d and prf params: [%.2f, %.2f, %.2f])\n'%(n_pix,x,y,sigma))
        
#     assert(not np.any(np.array(center)<pix_from_center) and not np.any(image_size-np.array(center)<=pix_from_center))
#     xmin = center[0]-pix_from_center
#     xmax = center[0]+pix_from_center+1
#     ymin = center[1]-pix_from_center
#     ymax = center[1]+pix_from_center+1
        
#     return [xmin, xmax, ymin, ymax]