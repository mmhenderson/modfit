import matplotlib.pyplot as plt

from skimage.color import rgb2gray, rgba2rgb
import skimage.io as skio
from skimage.util.shape import view_as_windows
from skimage.transform import resize

import numpy as np
import pandas as pd 
from scipy import stats  
import scipy.stats
from einops import rearrange, reduce, repeat

import itertools
import os
import sys
import glob
import warnings
import tqdm
import copy

from PIL import Image

from utils import default_paths, nsd_utils
from model_fitting import initialize_fitting


class bent_gabor_feature_bank():
    
    def __init__(self, freq_values=None, bend_values=None, orient_values=None, image_size=128):
        
        self.image_size = image_size;
 
        self.__set_kernel_params__(freq_values, bend_values, orient_values)
        self.__generate_kernels__()
        
    def __set_kernel_params__(self, freq_values, bend_values, orient_values):
        
        """
        Set some default params for the banana kernels.
        sigmaXbend:  sigma for the bent gaussian in x-direction
        sigmaYbend:  sigma for the bent gaussian in y-direction
        xA_shift:    center shift in x direction
        yA_shift:    center shift in y direction   
        
        freq_values: freq of filters, cyc/image
        orient_values: orientation of filters, 0-2pi
        bend_values: control bending of filters
        """
        
        self.sigmaXbend = 2;
#         self.sigmaXbend = 3;
        self.sigmaYbend = 6;
        self.kernel_size = self.image_size
        self.xA_shift   = 0
        self.yA_shift   = 0
        
        if freq_values is None:
            self.freq_values = [64, 32, 16, 8]
        else:
            self.freq_values = freq_values
            
        nyquist = 0.5*self.kernel_size
        if any(np.array(self.freq_values)>nyquist):
            raise ValueError('for image of size %d x %d, must have freqs < %.2f'%\
                            (self.kernel_size, self.kernel_size, nyquist))
        self.kA = np.array(self.freq_values)*2*np.pi / self.kernel_size
        self.scale_values = np.log(2*np.pi/self.kA)/np.log(np.sqrt(2))
        
        if orient_values is None:
            self.orient_values = np.linspace(0,2*np.pi, 9)[0:8]
        else:
            self.orient_values = orient_values
        if bend_values is None:
            self.bend_values = [0, 0.02,0.07,0.10,0.18,0.45]
        else:
            self.bend_values = bend_values
            
        print('freq values')
        print(self.freq_values)
        print('scale values')
        print(self.scale_values)
        print('bend values')
        print(self.bend_values)
        print('orient values')
        print(self.orient_values)

    def __make_bananakernel__(self, kA, bA, alphaA, is_curved):
        
        """
        Generate banana wavelet kernels.  The kernels
        can be used to filter a image to quantify curvatures.

        kA:          scale param, length of the wave vector K
                     kA =  2*np.pi/((np.sqrt(2))**scale)
                     filter frequency: (cycle/object) = kA*kernel_size / (2*pi)
        bA:          bending value b (arbitrary, roughly between 0-0.5)
        alphaA:      direction of the wave vector (i.e. orientation in rad)
        is_curved:   Are we making a curved gabor? If false, making a sharp angle detector.
                     Note if bA==0, then these are the same. 

        return SpaceKernel, FreqKernel

        """
        
        assert not (isinstance(bA, complex))
       
        kernel_size = self.kernel_size
        if kernel_size%2 !=0:
            kernel_size = kernel_size + 1
        [xA, yA] = np.meshgrid(np.arange(-kernel_size/2, kernel_size/2,1),np.arange(-kernel_size/2, kernel_size/2,1)) 
        xA = xA - self.xA_shift
        yA = yA - self.yA_shift

        xRotL = np.cos(alphaA)*xA + np.sin(alphaA)*yA 
        yRotL = np.cos(alphaA)*yA - np.sin(alphaA)*xA

        if is_curved:
            # make a curved "banana" gabor.
            scale = np.log(2*np.pi/kA)/np.log(np.sqrt(2))
            xRotBendL = xRotL + bA/scale * (yRotL)**2
        else:
            # otherwise making a sharp angle detector, use abs instead of squaring.
            # adjusting the constant here to make the bA values ~similar across curved/angle filters.
            xRotBendL = xRotL + bA*4 * np.abs(yRotL)
            
        yRotBendL = yRotL

        """make the DC free""" 
        tmpgaussPartA = np.exp(-0.5*(kA)**2*((xRotBendL/self.sigmaXbend)**2 + (yRotBendL/(self.sigmaYbend))**2))
        tmprealteilL  = 1*tmpgaussPartA*(np.cos(kA*xRotBendL) - 0)
        tmpimagteilL  = 1*tmpgaussPartA*(np.sin(kA*xRotBendL) - 0)

        numeratorRealL = np.sum(tmprealteilL)
        numeratorImagL = np.sum(tmpimagteilL)
        denominatorL   = np.sum(tmpgaussPartA)

        DCValueAnalysis = np.exp(-0.5 * self.sigmaXbend * self.sigmaXbend)
        if denominatorL==0:
            DCPartRealA = DCValueAnalysis
            DCPartImagA = 0
        else:    
            DCPartRealA = numeratorRealL/denominatorL
            DCPartImagA = numeratorImagL/denominatorL
            if DCPartRealA < DCValueAnalysis:
                DCPartRealA = DCValueAnalysis
                DCPartImagA = 0

        """generate a space kernel""" 
        preFactorA = kA**2
        gaussPartA = np.exp(-0.5*(kA)**2*((xRotBendL/self.sigmaXbend)**2 + (yRotBendL/(self.sigmaYbend))**2))
        realteilL  = preFactorA*gaussPartA*(np.cos(kA*xRotBendL) - DCPartRealA)
        imagteilL  = preFactorA*gaussPartA*(np.sin(kA*xRotBendL) - DCPartImagA)

        """normalize the kernel"""  
        normRealL   = np.sqrt(np.sum(realteilL**2))
        normImagL   = np.sqrt(np.sum(imagteilL**2))
        normFactorL = kA**2

        total_std = normRealL + normImagL
        if total_std == 0:
            total_std = 10**20
        norm_realteilL = realteilL*normFactorL/(0.5*total_std)
        norm_imagteilL = imagteilL*normFactorL/(0.5*total_std)
        
        space_kernel = norm_realteilL + norm_imagteilL*1j
        freq_kernel = np.fft.ifft2(space_kernel)
        
        return space_kernel, freq_kernel
 
    def __generate_kernels__(self):
        
        """
        Make the bank of filters.
        """
        
        n_scales = len(self.scale_values)
        n_orients = len(self.orient_values)
        n_bends = len(self.bend_values)
        
        curv_freq_kernels,rect_freq_kernels,lin_freq_kernels, \
            curv_space,rect_space,lin_space = [],[],[],[],[],[]
        
        curv_kernel_pars = np.zeros((n_scales*(n_bends-1)*n_orients, 4))
        rect_kernel_pars = np.zeros((n_scales*(n_bends-1)*n_orients, 4))
        lin_kernel_pars = np.zeros((n_scales*n_orients, 4))
        
        ci=-1; ri=-1; li=-1
            
        for is_curved in [True, False]:

            for kA, bA, alphaA in itertools.product(self.kA, self.bend_values, self.orient_values):

                space_kernel, freq_kernel = self.__make_bananakernel__(kA, bA, alphaA, is_curved)

                if bA == 0:
                    if not is_curved:
                        # the linear kernels each get defined twice (once with is_curv=True and False)
                        # only counting one occurence of each.
                        lin_freq_kernels.append(freq_kernel)
                        lin_space.append(space_kernel.real) 
                        li+=1
                        lin_kernel_pars[li,:] = [kA, bA, alphaA, is_curved]
                    else:
                        continue
                elif is_curved:
                    # this is a curved banana filter
                    curv_freq_kernels.append(freq_kernel)
                    curv_space.append(space_kernel.real)
                    ci+=1
                    curv_kernel_pars[ci,:] = [kA, bA, alphaA, is_curved]
                else:
                    # this is a second-order rectilinear filter
                    rect_freq_kernels.append(freq_kernel)
                    rect_space.append(space_kernel.real)
                    ri+=1
                    rect_kernel_pars[ri,:] = [kA, bA, alphaA, is_curved]
                    

        self.kernels = {'curv_freq':curv_freq_kernels, 'curv_space':curv_space,
                        'rect_freq':rect_freq_kernels, 'rect_space':rect_space, 
                        'lin_freq':lin_freq_kernels, 'lin_space':lin_space}
        self.rect_kernel_pars = rect_kernel_pars
        self.curv_kernel_pars = curv_kernel_pars
        self.lin_kernel_pars = lin_kernel_pars
        
    def plot_kernel_bends(self, ori_ind=0, scale_ind=0):

        rect_kernel_pars = self.rect_kernel_pars
        curv_kernel_pars = self.curv_kernel_pars
        lin_kernel_pars = self.lin_kernel_pars
        rect_spat_kernel_list = self.kernels['rect_space']
        curv_spat_kernel_list = self.kernels['curv_space']
        lin_spat_kernel_list = self.kernels['lin_space']

        plt.figure(figsize=(20,12))
        npx = 3;
        npy = len(self.bend_values)

        ori = self.orient_values[ori_ind]
        sc = self.kA[scale_ind]

        kk2plot = np.where((rect_kernel_pars[:,2]==ori) & (rect_kernel_pars[:,0]==sc))[0]
        for ki, kk in enumerate(kk2plot):
            plt.subplot(npx, npy, ki+1)
            plt.pcolormesh(rect_spat_kernel_list[kk])
            plt.axis('square')
            plt.gca().invert_yaxis()
            plt.axis('off')
            plt.title('bend=%.2f'%(rect_kernel_pars[kk,1]))

        kk2plot = np.where((curv_kernel_pars[:,2]==ori) & (curv_kernel_pars[:,0]==sc))[0]
        for ki, kk in enumerate(kk2plot):
            plt.subplot(npx, npy,ki+npy+1)
            plt.pcolormesh(curv_spat_kernel_list[kk])
            plt.axis('square')
            plt.gca().invert_yaxis()
            plt.axis('off')
            plt.title('bend=%.2f'%(curv_kernel_pars[kk,1]))

        kk2plot = np.where((lin_kernel_pars[:,2]==ori) & (lin_kernel_pars[:,0]==sc))[0]
        for ki, kk in enumerate(kk2plot):
            plt.subplot(npx, npy ,ki+npy*2+1)
            plt.pcolormesh(lin_spat_kernel_list[kk])
            plt.axis('square')
            plt.gca().invert_yaxis()
            plt.axis('off')
            plt.title('bend=%.2f'%(lin_kernel_pars[kk,1]))

        plt.suptitle('ori=%.2f rad, freq=%.2f cyc/im'%(ori, self.freq_values[scale_ind]))
        
    def filter_image_batch(self, image_batch, which_kernels='curv'):
        
        
        if which_kernels=='curv':
            kernel_list = self.kernels['curv_freq']
        elif which_kernels=='rect':
            kernel_list = self.kernels['rect_freq']
        elif which_kernels=='linear':
            kernel_list = self.kernels['lin_freq']
        else:
            raise ValueError('which_kernels must be one of [curv, rect, linear]')

        """image x, image y, kernel dimension, all images (4D array)"""
        all_kernels = np.dstack(kernel_list)
        
        """calculate kernel norm for normalization"""
        all_kernels_power =  np.einsum('ijk,ijk->k',np.abs(all_kernels),np.abs(all_kernels))
        all_kernels_power =  np.sqrt(all_kernels_power)

        """stack fft image list to a 3d array"""
        image_batch_fft = np.fft.fft2(image_batch, axes=(0,1))
        
        all_conved_images = np.abs(np.fft.ifft2(image_batch_fft[:,:,np.newaxis,:]*all_kernels[:,:,:,np.newaxis],axes=(0,1)))
        all_conved_images = np.power(all_conved_images,1/2) ## power correction
        all_conved_images = all_conved_images/all_kernels_power[np.newaxis, np.newaxis,:,np.newaxis]
    
        return np.fft.fftshift(all_conved_images, axes=(0,1))
 
            
def measure_curvrect_stats(bank, file_list, batch_size=20, \
                           resize=True, patchnorm=False, crop_one=False):
    
    n_images = len(file_list)
    n_batches = int(np.ceil(n_images/batch_size))

    bend_values = bank.bend_values
    scale_values = bank.scale_values
    image_size = bank.image_size
    
    curv_score_overall = np.zeros((n_images,))
    curv_score_each_bend = np.zeros((n_images, len(bend_values)-1))
    curv_score_each_scale = np.zeros((n_images, len(scale_values)))
    rect_score_overall = np.zeros((n_images,))
    rect_score_each_bend = np.zeros((n_images, len(bend_values)-1))
    rect_score_each_scale = np.zeros((n_images, len(scale_values)))
    lin_score_overall = np.zeros((n_images,))
    
    n_feats = (len(bend_values)-1)*len(scale_values)*len(bank.orient_values)
    mean_curv_over_space = np.zeros((n_images,n_feats))
    mean_rect_over_space = np.zeros((n_images,n_feats))    
    mean_lin_over_space = np.zeros((n_images,len(scale_values)*len(bank.orient_values)))
    
    for bb in range(n_batches):

        batch_inds = np.arange(batch_size*bb, np.min([batch_size*(bb+1), n_images]))

        file_list_batch = np.array(file_list)[batch_inds]
        image_list = []
        print('loading images, batch %d of %d'%(bb, n_batches))

        for fi, filename in enumerate(file_list_batch):

            image_raw = skio.imread(filename)
            
            if crop_one:
                # this means that original image size was odd, so will trim off one row of pixels.
                assert(image_raw.shape[0]==(image_size+1) and image_raw.shape[1]==(image_size+1))
                image_raw = image_raw[0:image_size, 0:image_size]
                
            if resize:
                image_resized = resize(image_raw, [image_size, image_size])
            else:
                assert(image_raw.shape[0]==image_size and image_raw.shape[1]==image_size)
                image_resized = image_raw
                
            if len(image_resized.shape)>2:              
                image_gray = rgb2gray(image_resized)
            else:
                image_gray = image_resized
                
            if patchnorm:
                image_normed = patchnorm(image_gray)
            else:
                image_normed = image_gray
            
                
            image_list.append(image_normed)

        image_batch = np.dstack(image_list)

        print('processing images w filter bank')
        sys.stdout.flush()
        all_curv_filt_coeffs = bank.filter_image_batch(image_batch, which_kernels='curv')
        all_rect_filt_coeffs = bank.filter_image_batch(image_batch, which_kernels='rect')
        all_lin_filt_coeffs = bank.filter_image_batch(image_batch, which_kernels='linear')

        print('computing summary stats')
        # Compute some summary stats (trying to give many options here)
        max_curv_images = np.max(all_curv_filt_coeffs, axis=2)
        max_rect_images = np.max(all_rect_filt_coeffs, axis=2)
        max_lin_images = np.max(all_lin_filt_coeffs, axis=2)

        unique_curv_inds = ((max_curv_images>max_rect_images) & (max_curv_images>max_lin_images))
        unique_rect_inds = ((max_rect_images>max_curv_images) & (max_rect_images>max_lin_images))
        unique_lin_inds = ((max_lin_images>max_curv_images) & (max_lin_images>max_rect_images))

        unique_curv_ims = copy.deepcopy(max_curv_images)
        unique_curv_ims[~unique_curv_inds] = 0.0
        unique_rect_ims = copy.deepcopy(max_rect_images)
        unique_rect_ims[~unique_rect_inds] = 0.0
        unique_lin_ims = copy.deepcopy(max_lin_images)
        unique_lin_ims[~unique_lin_inds] = 0.0

        curv_score_overall[batch_inds] = np.mean(np.mean(unique_curv_ims, axis=0), axis=0);
        rect_score_overall[batch_inds] = np.mean(np.mean(unique_rect_ims, axis=0), axis=0);
        lin_score_overall[batch_inds] = np.mean(np.mean(unique_lin_ims, axis=0), axis=0);


        
        for bb,bend in enumerate(bank.bend_values[1:]):

            kernel_inds = bank.curv_kernel_pars[:,1]==bend

            max_curv_images = np.max(all_curv_filt_coeffs[:,:,kernel_inds,:], axis=2)
            max_rect_images = np.max(all_rect_filt_coeffs[:,:,kernel_inds,:], axis=2)
            max_lin_images = np.max(all_lin_filt_coeffs, axis=2)
        
            unique_curv_inds = ((max_curv_images>max_rect_images) & (max_curv_images>max_lin_images))
            unique_rect_inds = ((max_rect_images>max_curv_images) & (max_rect_images>max_lin_images))

            unique_curv_ims = copy.deepcopy(max_curv_images)
            unique_curv_ims[~unique_curv_inds] = 0.0
            unique_rect_ims = copy.deepcopy(max_rect_images)
            unique_rect_ims[~unique_rect_inds] = 0.0

            curv_score_each_bend[batch_inds,bb] = np.mean(np.mean(unique_curv_ims, axis=0), axis=0);
            rect_score_each_bend[batch_inds,bb] = np.mean(np.mean(unique_rect_ims, axis=0), axis=0);


        for sc,scale in enumerate(bank.scale_values):

            kernel_inds = bank.curv_kernel_pars[:,0]==bank.kA[sc]

            max_curv_images = np.max(all_curv_filt_coeffs[:,:,kernel_inds,:], axis=2)
            max_rect_images = np.max(all_rect_filt_coeffs[:,:,kernel_inds,:], axis=2)

            lin_kernel_inds = bank.lin_kernel_pars[:,0]==bank.kA[sc]

            max_lin_images = np.max(all_lin_filt_coeffs[:,:,lin_kernel_inds,:], axis=2)

            unique_curv_inds = ((max_curv_images>max_rect_images) & (max_curv_images>max_lin_images))
            unique_rect_inds = ((max_rect_images>max_curv_images) & (max_rect_images>max_lin_images))

            unique_curv_ims = copy.deepcopy(max_curv_images)
            unique_curv_ims[~unique_curv_inds] = 0.0
            unique_rect_ims = copy.deepcopy(max_rect_images)
            unique_rect_ims[~unique_rect_inds] = 0.0

            curv_score_each_scale[batch_inds,sc] = np.mean(np.mean(unique_curv_ims, axis=0), axis=0);
            rect_score_each_scale[batch_inds,sc] = np.mean(np.mean(unique_rect_ims, axis=0), axis=0);

        # averaging over image dimensions        
        mean_curv_over_space[batch_inds] = np.mean(np.mean(all_curv_filt_coeffs, axis=0), axis=0).T
        mean_rect_over_space[batch_inds] = np.mean(np.mean(all_rect_filt_coeffs, axis=0), axis=0).T
        mean_lin_over_space[batch_inds] = np.mean(np.mean(all_lin_filt_coeffs, axis=0), axis=0).T
 
    best_curv_kernel = np.argmax(mean_curv_over_space, axis=1)
    best_rect_kernel = np.argmax(mean_rect_over_space, axis=1)
    best_lin_kernel = np.argmax(mean_lin_over_space, axis=1)
    
    curv_z = scipy.stats.zscore(mean_curv_over_space, axis=0)
    rect_z = scipy.stats.zscore(mean_rect_over_space, axis=0)
    lin_z = scipy.stats.zscore(mean_lin_over_space, axis=0)
    
    best_curv_kernel_z = np.argmax(curv_z, axis=1)
    best_rect_kernel_z = np.argmax(rect_z, axis=1)
    best_lin_kernel_z = np.argmax(lin_z, axis=1)
    
    mean_curv_z = np.mean(curv_z, axis=1)
    mean_rect_z = np.mean(rect_z, axis=1)
    mean_lin_z = np.mean(lin_z, axis=1)
    
    curv_rect_index = (mean_curv_z - mean_rect_z - mean_lin_z) / \
                      (mean_curv_z + mean_rect_z + mean_lin_z)

    curvrect = {'file_list': file_list, 
                 'curv_score_overall': curv_score_overall, 
                 'rect_score_overall': rect_score_overall, 
                 'curv_score_each_bend': curv_score_each_bend, 
                 'rect_score_each_bend': rect_score_each_bend, 
                 'curv_score_each_scale': curv_score_each_scale, 
                 'rect_score_each_scale': rect_score_each_scale,
                 'lin_score_overall': lin_score_overall, 
                 'mean_curv_over_space': mean_curv_over_space,
                 'mean_rect_over_space': mean_rect_over_space,
                 'mean_lin_over_space': mean_lin_over_space,  
                 'mean_curv_z': mean_curv_z,
                 'mean_rect_z': mean_rect_z,
                 'mean_lin_z': mean_lin_z,  
                 'best_curv_kernel': best_curv_kernel, 
                 'best_rect_kernel': best_rect_kernel, 
                 'best_lin_kernel': best_lin_kernel, 
                 'best_curv_kernel_z': best_curv_kernel_z, 
                 'best_rect_kernel_z': best_rect_kernel_z, 
                 'best_lin_kernel_z': best_lin_kernel_z, 
                 'curv_rect_index': curv_rect_index,
                 'curv_kernel_pars': bank.curv_kernel_pars, 
                 'rect_kernel_pars': bank.rect_kernel_pars, 
                 'lin_kernel_pars': bank.lin_kernel_pars}
    
    return curvrect

    
def patchnorm(image):

    if image.shape[0]%3 == 0:
        patch_size = 3
    else:
        patch_size = 4

    """create patches with the patch_size"""
    patches = view_as_windows(image, (patch_size,patch_size), patch_size)

    """ caculate norm of the local patches """ 
    local_norm = np.sqrt(np.einsum('ijkl->ij',patches**2))
    local_norm[local_norm<1] = 1

    """normalize local patches """ 
    normed_patches = patches/local_norm[:,:,np.newaxis,np.newaxis]

    """reshape normalized local patch to original shape """ 
    local_normed_image = rearrange(normed_patches,'h w c d -> (h c) (w d)')

    return local_normed_image

    
def measure_sketch_tokens_top_ims_curvrect(debug=False):
    
    freq_values_cyc_per_pix = [0.5, 0.25, 0.125, 0.0625, 0.03125]
    bend_values = [0, 0.04, 0.08, 0.16, 0.32, 0.64]
    orient_values = np.linspace(0,np.pi*2, 9)[0:8]
    
    folder2load = os.path.join(default_paths.sketch_token_feat_path, 'top_im_patches')
    fn2save = os.path.join(default_paths.sketch_token_feat_path, 'feature_curvrect_stats.npy')
    
    top_n_images = 96;
    subjects = np.arange(1,9)
    top_n_each_subj = int(np.ceil(top_n_images/len(subjects)))
    
    which_prf_grid=5
    prf_models = initialize_fitting.get_prf_models(which_prf_grid)
    n_prfs = prf_models.shape[0]
    
    sublist = np.repeat(subjects, top_n_each_subj)
    ranklist = np.tile(np.arange(top_n_each_subj), [len(subjects),])
    
    n_features = 150
    curv_rect_index = np.zeros((top_n_images, n_prfs, n_features))
    
    mean_curv = np.zeros((top_n_images, n_prfs, n_features))
    mean_rect = np.zeros((top_n_images, n_prfs, n_features))
    mean_lin = np.zeros((top_n_images, n_prfs, n_features))
    mean_curv_z = np.zeros((top_n_images, n_prfs, n_features))
    mean_rect_z = np.zeros((top_n_images, n_prfs, n_features))
    mean_lin_z = np.zeros((top_n_images, n_prfs, n_features))
    
    curv_score_overall = np.zeros((top_n_images, n_prfs, n_features))
    rect_score_overall = np.zeros((top_n_images, n_prfs, n_features))
    lin_score_overall = np.zeros((top_n_images, n_prfs, n_features))
    
    best_curv_kernel = np.zeros((top_n_images, n_prfs, n_features))    
    best_rect_kernel = np.zeros((top_n_images, n_prfs, n_features))    
    best_lin_kernel = np.zeros((top_n_images, n_prfs, n_features))    
    best_curv_kernel_z = np.zeros((top_n_images, n_prfs, n_features))    
    best_rect_kernel_z = np.zeros((top_n_images, n_prfs, n_features))    
    best_lin_kernel_z = np.zeros((top_n_images, n_prfs, n_features))    
    
    
    for mm in range(n_prfs):
        
        if debug and mm>1:
            continue
        print('Processing pRF %d of %d'%(mm, n_prfs))
        
        # getting the size from first file, all others are same size.
        imfn = os.path.join(folder2load, 'S1_prf%d_feature0_ranked0.jpg'%mm)
        image_size = skio.imread(imfn).shape[0]
        if np.mod(image_size,2)!=0:
            image_size-=1
            crop_one = True
        else:
            crop_one = False
        # adjusting the freqs so that they are constant cycles/pixel. 
        # since these images were cropped out of larger images at a fixed size, 
        # want this to be as if we filtered the entire image and then cropped.
        freq_values_cyc_per_image = np.array(freq_values_cyc_per_pix)*image_size
        bank = bent_gabor_feature_bank(freq_values = freq_values_cyc_per_image, \
                                       bend_values = bend_values, \
                                       orient_values = orient_values, \
                                       image_size=image_size)
    
        for ff in range(n_features):
            
            if debug and ff>1:
                continue
            print('Processing feat %d of %d'%(ff, n_features))
        
            # get all n filenames
            file_list = []
            for ii in range(top_n_images):               
                imfn = os.path.join(folder2load, 'S%d_prf%d_feature%d_ranked%d.jpg'%\
                                           (sublist[ii], mm, ff, ranklist[ii]))
                file_list.append(imfn)
                
            print('first image is %s'%file_list[0])
            print('last image is %s'%file_list[-1])
            curvrect = measure_curvrect_stats(bank, file_list, batch_size=20, \
                                              resize=False, patchnorm=False, \
                                              crop_one=crop_one)
          
            curv_rect_index[:,mm,ff] = curvrect['curv_rect_index']
            mean_curv[:,mm,ff] = np.mean(curvrect['mean_curv_over_space'], axis=1)
            mean_rect[:,mm,ff] = np.mean(curvrect['mean_rect_over_space'], axis=1)
            mean_lin[:,mm,ff] = np.mean(curvrect['mean_lin_over_space'], axis=1)
            mean_curv_z[:,mm,ff] = curvrect['mean_curv_z']
            mean_rect_z[:,mm,ff] = curvrect['mean_rect_z']
            mean_lin_z[:,mm,ff] = curvrect['mean_lin_z']
            curv_score_overall[:,mm,ff] = curvrect['curv_score_overall']
            rect_score_overall[:,mm,ff] = curvrect['rect_score_overall']
            lin_score_overall[:,mm,ff] = curvrect['lin_score_overall']
            best_curv_kernel[:,mm,ff] = curvrect['best_curv_kernel']
            best_rect_kernel[:,mm,ff] = curvrect['best_rect_kernel']
            best_lin_kernel[:,mm,ff] = curvrect['best_lin_kernel']
            best_curv_kernel_z[:,mm,ff] = curvrect['best_curv_kernel_z']
            best_rect_kernel_z[:,mm,ff] = curvrect['best_rect_kernel_z']
            best_lin_kernel_z[:,mm,ff] = curvrect['best_lin_kernel_z']
            
            
            
    dict2save = {'curv_rect_index': curv_rect_index, \
                 'mean_curv': mean_curv, \
                 'mean_rect': mean_rect, \
                 'mean_lin': mean_lin, \
                 'mean_curv_z': mean_curv_z, \
                 'mean_rect_z': mean_rect_z, \
                 'mean_lin_z': mean_lin_z, \
                 'curv_score_overall': curv_score_overall, \
                 'rect_score_overall': rect_score_overall, \
                 'lin_score_overall': lin_score_overall, \
                 'best_curv_kernel': best_curv_kernel, \
                 'best_rect_kernel': best_rect_kernel, \
                 'best_lin_kernel': best_lin_kernel, \
                 'best_curv_kernel_z': best_curv_kernel_z, \
                 'best_rect_kernel_z': best_rect_kernel_z, \
                 'best_lin_kernel_z': best_lin_kernel_z, \
                }
    print('saving to %s'%fn2save)
    np.save(fn2save, dict2save)   

def run_test():
     
    image_folder = '/lab_data/tarrlab/common/datasets/NSD_images/images/';
    
    subject_df = nsd_utils.get_subj_df(1);
    n_images = 1000
    image_inds = np.random.choice(np.arange(0,10000), n_images, replace=False)
    coco_id = np.array(subject_df['cocoId'])[image_inds]
    file_list = [os.path.join(image_folder,'%d.jpg'%cid) for cid in coco_id]

    
    # save_filename = os.path.join(default_paths.sketch_token_feat_path, 'test_random_cocoims_curv_rect_values.csv')
    fn2save = os.path.join(default_paths.sketch_token_feat_path, 'test_random_cocoims_curv_rect_values_wlocalnorm.npy')
    print('will save to %s\n'%(fn2save)) 
    
    
    image_size = 128;
    scale_values = np.linspace(2,8,4)
    bend_values= [0, 0.04, 0.08, 0.16, 0.32, 0.64]

    bank = bent_gabor_feature_bank(scale_values = scale_values, bend_values=bend_values, image_size=image_size)
    
    curvrect = measure_curvrect_stats(bank, file_list, batch_size=20)

    curvrect['image_inds'] = image_inds
              
    print('saving to %s\n'%(fn2save))    
    np.save(fn2save, curvrect)

    
if __name__ == '__main__':
    
    
#     run_test()

    measure_sketch_tokens_top_ims_curvrect(debug=False)