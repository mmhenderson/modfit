""" 
General code related to making Gabor filter banks.
"""

import numpy as np
import h5py
import warnings
import time
import torch
from tqdm import tqdm
from scipy import ndimage as ndi
from scipy import fft, ifft
from skimage import transform, color
from PIL import Image

class filter_bank:
    
    def __init__(self, orients_deg, freqs_cpp, spat_freq_bw=1, spat_aspect_ratio=1, n_sd_out=3, image_size=None):
        
        self.orients_deg = orients_deg;
        self.freqs_cpp = freqs_cpp;       
        self.n_filt_total = len(orients_deg)*len(freqs_cpp)
        self.spat_freq_bw=spat_freq_bw
        self.spat_aspect_ratio = spat_aspect_ratio
        self.n_sd_out = n_sd_out
        # this is the size at which images will be filtered (e.g. after any preprocessing is done)
        # if not specified, will automatically detect image size and not do any re-scaling.
        self.image_size = np.array(image_size)

    def get_filters_freq(self, image_size = None):
        
        if not hasattr(self, 'filters_freq'):
#             print('filters already made')
         
#         else:
#             print('making filters')
            if np.all(image_size==None):
                self.image_size = np.array(image_size)
            else:
                assert(np.all(self.image_size==np.array(image_size)))

            # first figure out how big biggest filters will be, so i can start preallocating array.
            sd_pix_x, sd_pix_y, patch_size = get_size_needed(np.min(self.freqs_cpp), self.spat_freq_bw, self.spat_aspect_ratio, self.n_sd_out)

            if not np.all(self.image_size==None):
                # must be bigger than original image, to avoid edge artifacts
    #             patch_size = np.array(np.ceil(patch_size/2) + self.image_size).astype('int')
                patch_size = np.maximum(patch_size, np.array(self.image_size)*3) 

            print('size of filter stack will be:')
            print((patch_size[0], patch_size[1] ,self.n_filt_total))
            filter_stack_freq_realpart = np.zeros((patch_size[0], patch_size[1] ,self.n_filt_total))
            filter_stack_freq_imagpart = np.zeros((patch_size[0], patch_size[1] ,self.n_filt_total))

            ii=-1
            orient_labs = np.zeros((self.n_filt_total,1))
            freq_labs = np.zeros((self.n_filt_total,1))

            for [oi, orient_deg] in enumerate(self.orients_deg):
                for [fi, freq_cpp] in enumerate(self.freqs_cpp):
                    ii=ii+1;
                    this_freq_filter = makeFreqGabor(orient_deg, freq_cpp, spat_freq_bw=self.spat_freq_bw,  
                                                     spat_aspect_ratio=self.spat_aspect_ratio,
                                                     n_sd_out = self.n_sd_out, patch_size=patch_size)

                    filter_stack_freq_realpart[:,:,ii] = np.real(this_freq_filter)
                    filter_stack_freq_imagpart[:,:,ii] = np.imag(this_freq_filter)
                    orient_labs[ii] = orient_deg
                    freq_labs[ii] = freq_cpp

            self.filters_freq = filter_stack_freq_realpart + 1j* filter_stack_freq_imagpart
            self.orient_labs = orient_labs
            self.freq_labs = freq_labs

        return self.filters_freq, self.orient_labs, self.freq_labs
    
    def get_filters_spat(self):
        
        if not hasattr(self, 'filters_spat'):
#             print('filters already made')
         
#         else:
#             print('making filters')
        
            filter_list = []
            ii=-1
            orient_labs = np.zeros((self.n_filt_total,1))
            freq_labs = np.zeros((self.n_filt_total,1))

            for [oi, orient_deg] in enumerate(self.orients_deg):
                for [fi, freq_cpp] in enumerate(self.freqs_cpp):
                    ii=ii+1;
                    this_spat_filter = makeSpatGabor(orient_deg, freq_cpp, spat_freq_bw=self.spat_freq_bw,  
                                                     spat_aspect_ratio=self.spat_aspect_ratio,
                                                     n_sd_out = self.n_sd_out)
                    filter_list.append(this_spat_filter)
                    orient_labs[ii] = orient_deg
                    freq_labs[ii] = freq_cpp
    
            self.filters_spat = filter_list
            self.orient_labs = orient_labs
            self.freq_labs = freq_labs

        return self.filters_spat, self.orient_labs, self.freq_labs
        
    
def get_rf_stack(x_centers, y_centers, sizes, image_size):
    """
    Create a grid of candidate RFs with the desired centers and sizes.
    """
    
    nrfs_total = len(x_centers)*len(y_centers)*len(sizes)
    rf_stack = np.zeros([image_size[0], image_size[1], nrfs_total])
    x_list = np.zeros([nrfs_total,1])
    y_list = np.zeros([nrfs_total,1])
    size_list = np.zeros([nrfs_total,1])
    
    ii=-1                         
    for xx in x_centers:
        for yy in y_centers:
            for size in sizes:
                              
                this_rf = gauss_2d(center_pix=[xx,yy], sd_pix=[size, size], patch_size=image_size)
                
                ii=ii+1
                rf_stack[:,:,ii] = this_rf
                size_list[ii,0] = size
                x_list[ii,0] = xx
                y_list[ii,0] = yy
       
    return rf_stack, x_list, y_list, size_list


def get_feats_by_rfs(feat_tensor, rfs_tensor):
    """
    Multiply each feature map by each candidate RF. 
    """
    feat_tensor = torch.tensor(feat_tensor)
    rfs_tensor = torch.tensor(rfs_tensor)

    nIms = feat_tensor.shape[0]
    nFeats = feat_tensor.shape[3]
    nRFs = rfs_tensor.shape[2]
    nPixTotal = feat_tensor.shape[1]*feat_tensor.shape[2]

    feats_reshaped = torch.moveaxis(torch.reshape(feat_tensor, [nIms, nPixTotal, nFeats]), 2, 1)
    rfs_reshaped = torch.reshape(rfs_tensor, [nPixTotal, nRFs])

    # this is [nImages x nFeatureMaps x nSpatialRFs]
    feats_by_rfs = torch.matmul(feats_reshaped, rfs_reshaped)

    return feats_by_rfs


def get_nsd_gabor_feat(nsd_inds, bank, nsd_brick_file='/lab_data/tarrlab/common/datasets/NSD/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5'):
    """
    Get a stack of feature maps corresponding to the specified NSD images.
    Use the orientation filters specified in bank.
    """
    nsd_inds = np.array(nsd_inds).astype('int')
    
    for ii in tqdm(range(len(nsd_inds))):

        nsd_ind = nsd_inds[ii]
#         print('loading image %d of %d'%(ii, len(nsd_inds)))

        with h5py.File(nsd_brick_file, "r") as f:
            nsd_im = f['imgBrick'][nsd_ind,:,:,:]  

#         print('filtering image %d of %d'%(ii, len(nsd_inds)))

        if np.all(bank.image_size==None):
            nsd_im_preproc = np.mean(nsd_im, axis=2)
        else:
            nsd_im_preproc = transform.resize(np.mean(nsd_im, axis=2), bank.image_size)
            
        t = time.time()
        image_stats, filters = filter_whole_image_freq(nsd_im_preproc, bank)
    #     image_stats, filters = filter_whole_image_spat(nsd_im_preproc, bank)
        feat = image_stats['mag']
        elapsed = time.time() - t
#         print('time elapsed: %.2f s'%elapsed)
        feat_map_size = np.shape(feat)

        if ii==0:
            all_feat = np.zeros([len(nsd_inds), feat_map_size[0], feat_map_size[1], feat_map_size[2]])

        all_feat[ii,:,:,:] = feat
        
    return all_feat


# def preproc_for_filt(image, desired_size=None, grayscale=True):
#     """ 
#     General use preprocessing for images (just resize and grayscale etc.)
#     """
#     if np.all(desired_size==None):
#         desired_size = np.shape(image)
        
#     if grayscale:
#         if len(np.shape(image))>2:
#             image = color.rgb2gray(image)
#         if len(desired_size)>2:
#             desired_size=desired_size[0:2]
       
#     else:
#         if len(desired_size)==2:
#             desired_size = np.array([desired_size[0], desired_size[1], 3])
            
#     image_preproc = transform.resize(image, desired_size)

#     return image_preproc.astype('float32')
  
def preproc_for_filt(image, desired_size=None, grayscale=True):
    """ 
    General use preprocessing for images (just resize and grayscale etc.)
    """
    if np.all(desired_size==None):
        desired_size = np.shape(image)
    if len(desired_size)>2:
        desired_size = desired_size[0:2]
        
    image_preproc = np.asarray(Image.fromarray(image).resize(desired_size, resample=Image.BILINEAR))   
    
    image_preproc = image_preproc.astype('float32')/255

    if (grayscale and len(np.shape(image_preproc))>2):

        newdat = image_preproc
        newdat[:,:,0] = image_preproc[:,:,0]*0.2126
        newdat[:,:,1] = image_preproc[:,:,1]*0.7152
        newdat[:,:,2] = image_preproc[:,:,2]*0.0722

        image_preproc = np.sum(newdat,axis=2)
        
    return image_preproc
    
    
def filter_whole_image_spat(image, bank):
    """
    Apply filter bank to entire image. In spatial domain.
    """
    
    filters_spat, orient_labs, freq_labs = bank.get_filters_spat()
    
    image = preproc_for_filt(image, bank.image_size, grayscale=True)

    # make sure it's the size we expect
    assert(bank.image_size[0]==np.shape(image)[0] & bank.image_size[1]==np.shape(image)[1])
    orig_size = np.shape(image)
    
    out = 1j*np.zeros((orig_size[0], orig_size[1],bank.n_filt_total))
    for ii in range(bank.n_filt_total):
        
        # run the convolution here
        out[:,:,ii] = ndi.convolve(image, filters_spat[ii], mode='reflect')
        
    mag = np.abs(out);
    phase = np.angle(out);
    
    #  add all this info to my structure
    image_stats={}
                                      
    image_stats['phase'] = phase;
    image_stats['mag'] = mag;
    image_stats['mean_phase'] = np.squeeze(np.mean(np.mean(phase,1),0));
    image_stats['mean_mag'] = np.squeeze(np.mean(np.mean(mag,1),0));
    image_stats['orient_labs'] = orient_labs;
    image_stats['freq_labs'] = freq_labs;
    image_stats['orig_size'] = orig_size;
                                     
    return image_stats, filters_spat
    
    
    
def filter_whole_image_freq(image, bank):
    """
    Apply filter bank to entire image. In frequency domain.
    """
    
    image = preproc_for_filt(image, bank.image_size, grayscale=True)

    # make sure it's the size we expect
    assert(bank.image_size[0]==np.shape(image)[0] & bank.image_size[1]==np.shape(image)[1])
    orig_size = np.shape(image)
        
    filters_freq, orient_labs, freq_labs = bank.get_filters_freq(image_size = orig_size)
   
    size_after_pad = np.array([np.shape(filters_freq)[0], np.shape(filters_freq)[1]])
            
    # Zero-pad the image for filtering
    pad_by = (size_after_pad - orig_size)/2   
    n2pad = np.transpose(np.array([np.floor(pad_by),np.ceil(pad_by)]).astype('int'));   
    # n2pad goes [[x1, x2],[y1, y2]]
    # padding with reflection instead of zeros, avoid edge artifacts!
    image_padded = np.pad(image,n2pad,mode='reflect')
    
    padded_size = np.shape(image_padded);
    assert(padded_size[0]==size_after_pad[0] and padded_size[1]==size_after_pad[1])

    # Filtering:
    # fft into frequency domain
    image_fft = fft.fftshift(fft.fft2(image_padded,axes=[0,1]));

    # Apply all my filters all at once
    filtered_freq_domain = np.tile(np.expand_dims(image_fft,2),[1,1,np.shape(filters_freq)[2]])*filters_freq;

    # get back to the spatial domain
    out_full = fft.ifft2(filtered_freq_domain,axes=[0,1]);

    # un-pad the image (back to its down-sampled size)
    out = out_full[n2pad[0,0] : n2pad[0,0]+orig_size[0], n2pad[1,0]:n2pad[1,0]+orig_size[1],:];
    assert(np.shape(out)[0]==orig_size[0] and np.shape(out)[1]==orig_size[1])

    mag = np.abs(out);
    phase = np.angle(out);

    #  add all this info to my structure
    image_stats={}
                                      
    image_stats['phase'] = phase;
    image_stats['mag'] = mag;
    image_stats['mean_phase'] = np.squeeze(np.mean(np.mean(phase,1),0));
    image_stats['mean_mag'] = np.squeeze(np.mean(np.mean(mag,1),0));
    image_stats['orient_labs'] = orient_labs;
    image_stats['freq_labs'] = freq_labs;
    image_stats['orig_size'] = orig_size;
    image_stats['padded_size'] = padded_size;
                                      
    return image_stats, filters_freq


def get_size_needed(freq_cpp, spat_freq_bw=1, spat_aspect_ratio=1, n_sd_out=3):
    """
    Figure out how big the filter needs to be, for a desired frequency and bandwidth.
    Bandwidth sets the spatial frequency bandwidth in "x" direction, the spatial aspect ratio 
    determines how the "y" direction is scaled relative to x. 
    """
    
    wavelength_pix = 1/freq_cpp;
    # Solve for the standard deviation of gaussian for filter, in pixels. 
    # Based on wavelength and spatial frequency bandwidth already defined. 
    # From relationship in "Nonlinear Operator in Oriented Texture", Kruizinga,
    # Petkov, 1999.
    sigma_x = wavelength_pix/np.pi*np.sqrt(np.log(2)/2)*(2**spat_freq_bw+1)/(2**spat_freq_bw-1);
    sigma_y = sigma_x/spat_aspect_ratio
       
    # now that we know this value, can figure out how big the patch needs to be.
    # 3 SD out to each side is probably fine.
    patch_size = np.array([np.ceil(sigma_x*n_sd_out*2), np.ceil(sigma_y*n_sd_out*2)])
    # want this always to be an odd number
    patch_size = patch_size + 1 - np.mod(patch_size,2)

    return sigma_x, sigma_y, patch_size.astype('int')


def gauss_2d(center_pix, sd_pix, patch_size, orient_deg=0):
    """
     Making a little gaussian blob. Can be elongated in one direction relative to other.
     [sd_pix] is the x and y standard devs in pix, respectively. 
     """
    if len(sd_pix)==1:
        sd_pix = np.array([sd_pix, sd_pix])
        
    aspect_ratio = sd_pix[0] / sd_pix[1]
    orient_rad = orient_deg/180*np.pi
    
    # first meshgrid over image space
    x,y = np.meshgrid(np.arange(0,patch_size[0],1), np.arange(0,patch_size[1],1))
    x_centered = x-center_pix[0]
    y_centered = y-center_pix[1]
    
    # rotate the axes to match desired orientation (if orient=0, this is just regular x and y)
    x_prime = x_centered * np.cos(orient_rad) + y_centered * np.sin(orient_rad)
    y_prime = y_centered * np.cos(orient_rad) - x_centered * np.sin(orient_rad)

    # make my gaussian w the desired size/eccentricity
    gauss = np.exp(-((x_prime)**2 + aspect_ratio**2 * (y_prime)**2)/(2*sd_pix[0]**2))

    return gauss


def complex_gabor_patch(orient_deg, freq_cpp, sd_pix, patch_size, center_pix=None, phase_deg=270):
    """
    Making a complex Gabor. Quadrature pair (two sine waves 90 deg apart in phase).
    Magnitude of the complex number gives the spatial envelope.
    """
    
    if len(sd_pix)==1:
        sd_pix = np.array([sd_pix, sd_pix])
    if center_pix==None:
        center_pix = np.array(patch_size)/2 - 0.5 
    if sd_pix[0]*4 > patch_size[0] or sd_pix[1]*4 > patch_size[1]:
        warnings.warn('Warning: Patch size is too small to fit 2 SD of the gaussian blob')       
    if freq_cpp>=0.5:
        warnings.warn('Warning: Frequency is at or above the upper Nyquist limit (2 pixels/cycle)')
     
    
    gauss = gauss_2d(center_pix, sd_pix, patch_size, orient_deg)
    
    orient_rad = orient_deg/180*np.pi
    phase_rad = phase_deg/180*np.pi
    phase_conj_rad = phase_rad + np.pi/2   # define the conjugate phase pairs, 90 deg apart.
    
    x,y = np.meshgrid(np.arange(0,patch_size[0]), (np.arange(0,patch_size[1])))
    x = x-np.mean(x)
    y = y-np.mean(y)
  
    sine = (np.sin(freq_cpp*2*np.pi*(y*np.sin(orient_rad)+x*np.cos(orient_rad))-phase_rad));
    
    sine_conj = (np.sin(freq_cpp*2*np.pi*(y*np.sin(orient_rad)+x*np.cos(orient_rad))-phase_conj_rad));
    
    gabor = (sine + 1j*sine_conj) * gauss
    
    return gabor

        
def makeSpatGabor(orient_deg, freq_cpp, spat_freq_bw=1, spat_aspect_ratio=1, n_sd_out=3, patch_size=None):
        
    """
    Construct a spatial kernel for Gabor filter.
    """ 
   
    if np.all(patch_size==None):
        sd_pix_x, sd_pix_y, patch_size = get_size_needed(freq_cpp, spat_freq_bw, spat_aspect_ratio, n_sd_out)
    else:
        sd_pix_x, sd_pix_y, patch_size_req = get_size_needed(freq_cpp, spat_freq_bw, spat_aspect_ratio, n_sd_out)
        if patch_size_req[0] > patch_size[0] or patch_size_req[1] > patch_size[0]:
            warnings.warn('your manually specified patch size is smaller than the calculated required patch size. proceeding anyway...')
    
    patch_size = np.array([np.max(patch_size), np.max(patch_size)])
    gabor_spat = complex_gabor_patch(orient_deg, freq_cpp, [sd_pix_x, sd_pix_y], patch_size);
    
    return gabor_spat


def makeFreqGabor(orient_deg, freq_cpp, spat_freq_bw=1, spat_aspect_ratio=1, n_sd_out=3, patch_size=None):
        
    """ 
    Directly construct frequency domain transfer function of
    Gabor filter. based on (Jain, Farrokhnia, "Unsupervised Texture
    Segmentation Using Gabor Filters", 1991)
    """

    # First set some parameters
    orient_rad = orient_deg/180*np.pi
    wavelength_pix = 1/freq_cpp

    if np.all(patch_size==None):
        sd_pix_x, sd_pix_y, patch_size = get_size_needed(freq_cpp, spat_freq_bw, spat_aspect_ratio, n_sd_out)
    else:
        sd_pix_x, sd_pix_y, patch_size_req = get_size_needed(freq_cpp, spat_freq_bw, spat_aspect_ratio, n_sd_out)
        if patch_size_req[0] > patch_size[0] or patch_size_req[1] > patch_size[0]:
            warnings.warn('your manually specified patch size is smaller than the calculated required patch size. proceeding anyway...')

    patch_size = np.array([np.max(patch_size), np.max(patch_size)])
    # first meshgrid over image space
    u,v = np.meshgrid(np.arange(0,patch_size[0],1), np.arange(0,patch_size[1],1))
    center_pix = np.array(patch_size)/2 - 0.5 
    u_centered = (u-center_pix[0])/center_pix[0]/2
    v_centered = (v-center_pix[1])/center_pix[1]/2
 
    # rotate the axes to match desired orientation (if orient=0, this is just regular x and y)
    # Uprime defines the axis where SF varies, Vprime defines axis where orient varies.
    Uprime = u_centered * np.cos(orient_rad) + v_centered * np.sin(orient_rad)
    Vprime = v_centered * np.cos(orient_rad) - u_centered * np.sin(orient_rad)

    sigmau = 1/(2*np.pi*sd_pix_x); 
    sigmav = 1/(2*np.pi*sd_pix_y);

    A = 2*np.pi*sd_pix_x*sd_pix_y;

    # create the Gabor in freq domain (complex number)
    # This is from Jain & Farrokhnia (1991) Pattern Recognition
    # note this is not symmetric across the origin
    gabor_freq_magnitude = 1 * np.exp(-0.5*( ((Uprime-freq_cpp)**2)/sigmau**2 + Vprime**2 / sigmav**2))

    # now making it a complex number, with 0 phase.
    gabor_freq = gabor_freq_magnitude  + 0*1j

    return gabor_freq