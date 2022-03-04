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

import torch

import itertools
import os
import sys
import glob
import warnings
import tqdm
import copy
import time

from PIL import Image

import argparse

from utils import default_paths, nsd_utils, texture_utils
from model_fitting import initialize_fitting
from feature_extraction import fwrf_features

try:
    device = initialize_fitting.init_cuda()
except:
    device = 'cpu:0'
print('Found device:')
print(device)

class bent_gabor_feature_bank():
    
    def __init__(self, freq_values=None, bend_values=None, orient_values=None, \
                 image_size=128, device=None):
        
        self.image_size = image_size;
        if device is None:
            self.device = 'cpu:0'
        else:
            self.device = device
            
        print('initialized feature bank, device is: %s'%self.device)
 
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
        self.n_rect_filters = self.rect_kernel_pars.shape[0]
        self.n_curv_filters = self.curv_kernel_pars.shape[0]      
        self.n_lin_filters = self.lin_kernel_pars.shape[0]
        
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
    
    
    def filter_image_batch_pytorch(self, image_batch, which_kernels='curv', to_numpy=True):

        # This should behave like filter_image_batch, but is much faster when using 
        # a GPU (specify in self.device)
        
        if which_kernels=='curv':
            kernel_list = self.kernels['curv_freq']
        elif which_kernels=='rect':
            kernel_list = self.kernels['rect_freq']
        elif which_kernels=='linear':
            kernel_list = self.kernels['lin_freq']
        elif which_kernels=='all':
            kernel_list = self.kernels['curv_freq']+self.kernels['rect_freq']+self.kernels['lin_freq']
        else:
            raise ValueError('which_kernels must be one of [curv, rect, linear, all]')

        # stack all the filters together, [self.kernel_size, self.kernel_size, n_filters]
        # and send to specified device.
        all_kernels = np.dstack(kernel_list)
        all_kernels_tensor = torch.complex(torch.Tensor(np.real(all_kernels)), \
                                           torch.Tensor(np.imag(all_kernels))).to(self.device)

        # Compute power of each kernel, will use to normalize the convolution result.
        all_kernels_power =  torch.sum(torch.sum(torch.pow(torch.abs(all_kernels_tensor), 2), \
                                                 axis=0), axis=0)
        all_kernels_power =  torch.sqrt(all_kernels_power)

        # send image batch to device [self.image_size, self.image_size, n_images]
        image_batch_tensor = torch.Tensor(image_batch).to(self.device)
        # get frequency domain representation of images
        image_batch_fft = torch.fft.fftn(image_batch_tensor, dim=(0,1))

        # apply the filters by multiplying all at once
        mult = image_batch_fft.view([self.image_size, self.image_size,1,-1]) * \
                all_kernels_tensor.view([self.image_size, self.image_size,-1,1])
        # back to spatial domain 
        all_conved_images = torch.abs(torch.fft.ifftn(mult,dim=(0,1)))

        # power correction
        all_conved_images = torch.pow(all_conved_images,1/2) 
        all_conved_images = all_conved_images/ \
                    all_kernels_power.view([1,1,all_kernels_power.shape[0],1])
        
        # shift back to original spatial configuration
        all_conved_images = torch.fft.fftshift(all_conved_images, dim=(0,1))

        if to_numpy:
            all_conved_images = all_conved_images.detach().cpu().numpy()
            
        return all_conved_images
 
            
def measure_curvrect_stats(bank, image_brick, batch_size=20, \
                           resize=True, patchnorm=False):
    
    # image brick should be [n_images x h x w]
    
    n_images = image_brick.shape[0]
    
    n_batches = int(np.ceil(n_images/batch_size))

    bend_values = bank.bend_values
    scale_values = bank.scale_values
    image_size = bank.image_size
    print(image_brick.shape)
    print(image_brick.shape[1:3])
    print(image_size)
    assert(np.all(image_size==np.array(image_brick.shape[1:3])))
    assert(np.mod(image_brick.shape[1],2)==0) # must have even n pixels
    
    curv_score_method1 = np.zeros((n_images,))
    lin_score_method1 = np.zeros((n_images,))   
    
    curv_score_method2 = np.zeros((n_images,))
    rect_score_method2 = np.zeros((n_images,))
    lin_score_method2 = np.zeros((n_images,))
    
    n_feats = (len(bend_values)-1)*len(scale_values)*len(bank.orient_values)
    mean_curv_over_space = np.zeros((n_images,n_feats))
    mean_rect_over_space = np.zeros((n_images,n_feats))    
    mean_lin_over_space = np.zeros((n_images,len(scale_values)*len(bank.orient_values)))
    
    for bb in range(n_batches):

        batch_inds = np.arange(batch_size*bb, np.min([batch_size*(bb+1), n_images]))

        image_batch = np.moveaxis(image_brick[batch_inds,:,:], [0,1,2], [2,0,1])
        
        print('processing images w filter bank')
        sys.stdout.flush()
        st = time.time()
        all_curv_filt_coeffs = bank.filter_image_batch_pytorch(image_batch, which_kernels='curv')
        all_rect_filt_coeffs = bank.filter_image_batch_pytorch(image_batch, which_kernels='rect')
        all_lin_filt_coeffs = bank.filter_image_batch_pytorch(image_batch, which_kernels='linear')
        
        elapsed = time.time() - st
        print('took %.5f sec to process batch of %d images (image size %d pix)'\
                  %(elapsed, len(batch_inds), image_batch.shape[0]))
 
        print('computing summary stats')
        # Compute some summary stats (trying to give many options here)
        max_curv_images = np.max(all_curv_filt_coeffs, axis=2)
        max_rect_images = np.max(all_rect_filt_coeffs, axis=2)
        max_lin_images = np.max(all_lin_filt_coeffs, axis=2)
        
        # method 1 - compare curved filters vs linear filters.
        unique_curv_inds = max_curv_images>max_lin_images
        unique_lin_inds = max_lin_images>max_curv_images
        
        unique_curv_ims = copy.deepcopy(max_curv_images)
        unique_curv_ims[~unique_curv_inds] = 0.0        
        unique_lin_ims = copy.deepcopy(max_lin_images)
        unique_lin_ims[~unique_lin_inds] = 0.0

        curv_score_method1[batch_inds] = np.mean(np.mean(unique_curv_ims, axis=0), axis=0);
        lin_score_method1[batch_inds] = np.mean(np.mean(unique_lin_ims, axis=0), axis=0);
        
        # method 2 - compare curved against both angular (rect) and linear filters.
        unique_curv_inds = ((max_curv_images>max_rect_images) & (max_curv_images>max_lin_images))
        unique_rect_inds = ((max_rect_images>max_curv_images) & (max_rect_images>max_lin_images))
        unique_lin_inds = ((max_lin_images>max_curv_images) & (max_lin_images>max_rect_images))

        unique_curv_ims = copy.deepcopy(max_curv_images)
        unique_curv_ims[~unique_curv_inds] = 0.0
        unique_rect_ims = copy.deepcopy(max_rect_images)
        unique_rect_ims[~unique_rect_inds] = 0.0
        unique_lin_ims = copy.deepcopy(max_lin_images)
        unique_lin_ims[~unique_lin_inds] = 0.0

        curv_score_method2[batch_inds] = np.mean(np.mean(unique_curv_ims, axis=0), axis=0);
        rect_score_method2[batch_inds] = np.mean(np.mean(unique_rect_ims, axis=0), axis=0);
        lin_score_method2[batch_inds] = np.mean(np.mean(unique_lin_ims, axis=0), axis=0);

        # averaging power over image dimensions        
        mean_curv_over_space[batch_inds,:] = np.mean(np.mean(all_curv_filt_coeffs, axis=0), axis=0).T
        mean_rect_over_space[batch_inds,:] = np.mean(np.mean(all_rect_filt_coeffs, axis=0), axis=0).T
        mean_lin_over_space[batch_inds,:] = np.mean(np.mean(all_lin_filt_coeffs, axis=0), axis=0).T
 
    curvrect = {'curv_score_method1': curv_score_method1, 
                'lin_score_method1': lin_score_method1, 
                'curv_score_method2': curv_score_method2, 
                 'rect_score_method2': rect_score_method2, 
                 'lin_score_method2': lin_score_method2,                 
                 'mean_curv_over_space': mean_curv_over_space,
                 'mean_rect_over_space': mean_rect_over_space,
                 'mean_lin_over_space': mean_lin_over_space,  
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

    
def measure_sketch_tokens_top_ims_curvrect(debug=False, which_prf_grid=5, batch_size=20):
    
    freq_values_cyc_per_pix = [0.5, 0.25, 0.125, 0.0625, 0.03125]
    bend_values = [0, 0.04, 0.08, 0.16, 0.32, 0.64]
    orient_values = np.linspace(0,np.pi*2, 9)[0:8]
    
    if debug:
        subjects = [1]
    else:
        subjects = np.arange(1,9)
    path_to_load = default_paths.sketch_token_feat_path
    feat_loaders = [fwrf_features.fwrf_feature_loader(subject=ss,\
                                which_prf_grid=which_prf_grid, \
                                feature_type='sketch_tokens',\
                                use_pca_feats = False) for ss in subjects]
    n_features = feat_loaders[0].max_features
    
    val_inds_ss = np.array(nsd_utils.get_subj_df(subject=1)['shared1000'])
    trninds_ss = np.where(val_inds_ss==False)[0]
        
    ims_list = []
    for ss in subjects:
        images = nsd_utils.get_image_data(subject=ss)
        images = images[trninds_ss,:,:,:]
        image_size = images.shape[2:4]
        images = nsd_utils.image_uncolorize_fn(images)
        ims_list.append(images)

    prf_models = initialize_fitting.get_prf_models(which_prf_grid) 

    # compute bounding boxes for each pRF
    bboxes = np.array([ texture_utils.get_bbox_from_prf(prf_models[mm,:], \
                                   image_size, n_prf_sd_out=2, \
                                   min_pix=None, verbose=False, \
                                   force_square=True) \
                for mm in range(prf_models.shape[0]) ])
    # some pRFs might have the exact same bounding box as others, even if they
    # are not totally identical. Get rid of duplicate bounding boxes now.
    bboxes = np.unique(bboxes, axis=0)
    # put the biggest pRFs first, in case they cause out of memory errors 
    bboxes = bboxes[np.flip(np.argsort(bboxes[:,1]-bboxes[:,0])),:]
    n_prfs = bboxes.shape[0]
    
    print(bboxes)
    print(n_prfs)
    
    fn2save = os.path.join(default_paths.sketch_token_feat_path, 'Sketch_token_feature_curvrect_stats.npy')
    
    top_n_images = 96;
    top_n_each_subj = int(np.ceil(top_n_images/len(subjects)))
 
    curv_score_method1 = np.zeros((top_n_images, n_prfs, n_features))
    lin_score_method1 = np.zeros((top_n_images, n_prfs, n_features))
    
    curv_score_method2 = np.zeros((top_n_images, n_prfs, n_features))
    rect_score_method2 = np.zeros((top_n_images, n_prfs, n_features))
    lin_score_method2 = np.zeros((top_n_images, n_prfs, n_features))
    
    mean_curv = np.zeros((top_n_images, n_prfs, n_features))
    mean_rect = np.zeros((top_n_images, n_prfs, n_features))
    mean_lin = np.zeros((top_n_images, n_prfs, n_features))
    mean_curv_z = np.zeros((top_n_images, n_prfs, n_features))
    mean_rect_z = np.zeros((top_n_images, n_prfs, n_features))
    mean_lin_z = np.zeros((top_n_images, n_prfs, n_features))
    
    max_curv = np.zeros((top_n_images, n_prfs, n_features))
    max_rect = np.zeros((top_n_images, n_prfs, n_features))
    max_lin = np.zeros((top_n_images, n_prfs, n_features))
    max_curv_z = np.zeros((top_n_images, n_prfs, n_features))
    max_rect_z = np.zeros((top_n_images, n_prfs, n_features))
    max_lin_z = np.zeros((top_n_images, n_prfs, n_features))

    best_curv_kernel = np.zeros((top_n_images, n_prfs, n_features))    
    best_rect_kernel = np.zeros((top_n_images, n_prfs, n_features))    
    best_lin_kernel = np.zeros((top_n_images, n_prfs, n_features))    
    best_curv_kernel_z = np.zeros((top_n_images, n_prfs, n_features))    
    best_rect_kernel_z = np.zeros((top_n_images, n_prfs, n_features))    
    best_lin_kernel_z = np.zeros((top_n_images, n_prfs, n_features))    
          
    for mm in range(n_prfs):
        
        print('Processing pRF %d of %d\n'%(mm, n_prfs))
        
        st = time.time()

        bbox = bboxes[mm,:]
        
        cropped_size = bbox[1]-bbox[0]
        if np.mod(cropped_size,2)!=0:
            # needs to be even n pixels, so shave one pixel off if needed
            cropped_size-=1
            bbox[1] = bbox[1]-1
            bbox[3] = bbox[3]-1
            
        print(bbox)
        print(cropped_size)
        print('\n')
        
        # adjusting the freqs so that they are constant cycles/pixel. 
        # since these images were cropped out of larger images at a fixed size, 
        # want this to be as if we filtered the entire image and then cropped.
        # but it is faster just to filter the crops.
        freq_values_cyc_per_image = np.array(freq_values_cyc_per_pix)*cropped_size
        bank = bent_gabor_feature_bank(freq_values = freq_values_cyc_per_image, \
                                       bend_values = bend_values, \
                                       orient_values = orient_values, \
                                       image_size=cropped_size, \
                                       device = device)
        
        for ff in range(n_features):
            
            if debug and ff>1:
                continue
            print('Processing feature %d of %d'%(ff, n_features))
            
            # making a stack of images to analyze, across all subs
            top_images_cropped = []
            
            for si, ss in enumerate(subjects):

                # get sketch tokens model response to each image at this pRF position
                feat, _ = feat_loaders[si].load(trninds_ss, prf_model_index=mm)
                assert(feat.shape[0]==ims_list[si].shape[0])

                # sort in descending order, to get top n images
                sorted_order = np.flip(np.argsort(feat[:,ff]))
            
                top_images = ims_list[si][sorted_order[0:top_n_each_subj],:,:,:]

                # taking just the patch around this pRF, because this is the region that 
                # contributed to computing the sketch tokens feature
                top_cropped = top_images[:,0,bbox[0]:bbox[1], bbox[2]:bbox[3]]

                top_images_cropped.append(top_cropped)
                
            top_images_cropped = np.concatenate(top_images_cropped, axis=0)
                
            assert(top_images_cropped.shape[0]==top_n_images)
            assert(top_images_cropped.shape[2]==cropped_size)
          
            curvrect = measure_curvrect_stats(bank, image_brick=top_images_cropped, \
                                              batch_size=batch_size, \
                                              resize=False, patchnorm=False)
          
            curv_score_method1[:,mm,ff] = curvrect['curv_score_method1']
            lin_score_method1[:,mm,ff] = curvrect['lin_score_method1']
            
            curv_score_method2[:,mm,ff] = curvrect['curv_score_method2']
            rect_score_method2[:,mm,ff] = curvrect['rect_score_method2']
            lin_score_method2[:,mm,ff] = curvrect['lin_score_method2']
            
            mean_curv[:,mm,ff] = np.mean(curvrect['mean_curv_over_space'], axis=1)
            mean_rect[:,mm,ff] = np.mean(curvrect['mean_rect_over_space'], axis=1)
            mean_lin[:,mm,ff] = np.mean(curvrect['mean_lin_over_space'], axis=1)
            
            max_curv[:,mm,ff] = np.max(curvrect['mean_curv_over_space'], axis=1)
            max_rect[:,mm,ff] = np.max(curvrect['mean_rect_over_space'], axis=1)
            max_lin[:,mm,ff] = np.max(curvrect['mean_lin_over_space'], axis=1)

            best_curv_kernel[:,mm,ff] = np.argmax(curvrect['mean_curv_over_space'], axis=1)
            best_rect_kernel[:,mm,ff] = np.argmax(curvrect['mean_rect_over_space'], axis=1)
            best_lin_kernel[:,mm,ff] = np.argmax(curvrect['mean_lin_over_space'], axis=1)

            # try z-scoring, see if stats make more sense
            curv_z = scipy.stats.zscore(curvrect['mean_curv_over_space'], axis=0)
            rect_z = scipy.stats.zscore(curvrect['mean_rect_over_space'], axis=0)
            lin_z = scipy.stats.zscore(curvrect['mean_lin_over_space'], axis=0)
 
            mean_curv_z[:,mm,ff] = np.mean(curv_z, axis=1)
            mean_rect_z[:,mm,ff] = np.mean(rect_z, axis=1)
            mean_lin_z[:,mm,ff] = np.mean(lin_z, axis=1)
            
            max_curv_z[:,mm,ff] = np.max(curv_z, axis=1)
            max_rect_z[:,mm,ff] = np.max(rect_z, axis=1)
            max_lin_z[:,mm,ff] = np.max(lin_z, axis=1)
            
            best_curv_kernel_z[:,mm,ff] = np.argmax(curv_z, axis=1)
            best_rect_kernel_z[:,mm,ff] = np.argmax(rect_z, axis=1)
            best_lin_kernel_z[:,mm,ff] = np.argmax(lin_z, axis=1)
        
        elapsed = time.time() - st;
        print('\nTook %.5f sec to do pRF %d (patch size %d pix)\n'%(elapsed, mm, cropped_size))
            
            
    dict2save = {'curv_score_method1': curv_score_method1, \
                 'lin_score_method1': lin_score_method1, \
                 
                 'curv_score_method2': curv_score_method2, \
                 'rect_score_method2': rect_score_method2, \
                 'lin_score_method2': lin_score_method2, \

                 'mean_curv': mean_curv, \
                 'mean_rect': mean_rect, \
                 'mean_lin': mean_lin, \
                 'mean_curv_z': mean_curv_z, \
                 'mean_rect_z': mean_rect_z, \
                 'mean_lin_z': mean_lin_z, \
                 
                 'max_curv': max_curv, \
                 'max_rect': max_rect, \
                 'max_lin': max_lin, \
                 'max_curv_z': max_curv_z, \
                 'max_rect_z': max_rect_z, \
                 'max_lin_z': max_lin_z, \
                 
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
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    
    args = parser.parse_args()

    if args.debug:
        print('DEBUG MODE\n')
        
#     run_test()

    measure_sketch_tokens_top_ims_curvrect(debug=args.debug==1)