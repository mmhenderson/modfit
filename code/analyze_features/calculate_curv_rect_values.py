import matplotlib.pyplot as plt

from skimage.color import rgb2gray, rgba2rgb
import skimage.io as skio
from skimage.util.shape import view_as_windows
from skimage.transform import resize

import numpy as np
import pandas as pd 
from scipy import stats  
from einops import rearrange, reduce, repeat

import itertools
import os
import glob
import warnings
import tqdm

from utils import default_paths

"""
Code to perform curvature analysis on images corresponding to each sketch token 
feature (clusters in curvature space).
Original version from Xiaomin Yue, related to paper:
"Curvature processing in human visual cortical areas" (2020), Neuroimage.
Modified for this project by MMH, including adding option for angle detection.
Saves out a csv file listing the approx curvature params of each feature.
"""

class CurvRectValues:  
    def __init__(self, proc_cluster_ims=False, file_block_size=10):
        self.files           = []
        self.image_size      = 128 
        self.file_block_size = file_block_size
        self.images          = []
        self.fft_images      = []
        self.curv_values     = []
        self.rect_values     = []
        self.curv_max        = []
        self.rect_max        = []
        self.curv_unique     = []
        self.rect_unique     = []
        self.proc_cluster_ims= proc_cluster_ims
        self.__set_kernel_params__()
        self.__generate_kernels__()
        
    def __set_kernel_params__(self):
        
        """
        Set some default params for the banana kernels.
        mA:          magnitude value m
        xA_half:     x-size
        yA_half:     y-size
        sigmaXbend:  sigma for the bent gaussian in x-direction
        sigmaYbend:  sigma for the bent gaussian in y-direction
        xA_shift:    center shift in x direction
        yA_shift:    center shift in y direction
        
        """
        
        self.mA = 3;
        self.sigmaXbend = 2
        self.sigmaYbend = 2
        self.xA_half    = self.image_size/2
        self.yA_half    = self.image_size/2
        self.xA_shift   = 0
        self.yA_shift   = 0
        
        # now setting values that define the bank of filters we will create. 
        if self.proc_cluster_ims:
            # special case where i am working with sketch tokens "cluster" images, 
            # not natural image patches. 
            # want really fine orientation resolution/coarser scale resolution.
            self.scale_values = [9,12]
#             self.orient_values = np.linspace(0,2*np.pi,73)[0:72]
            self.orient_values = np.linspace(0,2*np.pi,25)[0:24]
        else:
            # usual case, natural image patches (more scales/fewer orients).
            self.scale_values = [1,5,8]
            self.orient_values = np.linspace(0,2*pi, 9)[0:8]
            
        self.scale_values = 2*np.pi / (np.sqrt(2)**self.scale_values)
        
        self.bend_values = [0, 0.02,0.07,0.10,0.18,0.45]
    
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

        kA:          length of the wave vector K (i.e. scale)
                     filter frequency: (cycle/object) = xA*kA/(2*pi*mA)
        bA:          bending value b (arbitrary, roughly between 0-0.5)
        alphaA:      direction of the wave vector (i.e. orientation in rad)
        is_curved:   Are we making a curved gabor? If false, making a sharp angle detector.
                     Note if bA==0, then these are the same. 

        return SpaceKernel, FreqKernel

        """
        
        if isinstance(bA, complex):
            print('bA has to be real number. However your input is a complex number')
            bA = np.real(bA)

        if any(x<=0 for x in np.array([kA, self.mA])) or any(np.isnan(np.array([kA, bA, alphaA, self.mA]))):     
            out_range_value = 10**-20
            SpaceKernel = np.ones((2*self.xA_half,2*self.yA_half))*out_range_value
            FreqKernel   = np.ones((2*self.xA_half,2*self.yA_half))*out_range_value
            return SpaceKernel, FreqKernel

        kernel_size = 2*self.xA_half
        if kernel_size%2 !=0:
            kernel_size = kernel_size + 1
        [xA, yA] = np.meshgrid(np.arange(-kernel_size/2, kernel_size/2,1),np.arange(-kernel_size/2, kernel_size/2,1)) 
        xA = xA - self.xA_shift
        yA = yA - self.yA_shift

        xRotL = np.cos(alphaA)*xA + np.sin(alphaA)*yA 
        yRotL = np.cos(alphaA)*yA - np.sin(alphaA)*xA

        if is_curved:
            # default behavior, make a curved "banana" gabor.
            xRotBendL = xRotL + bA/8 * (yRotL)**2
        else:
            # otherwise making a sharp angle detector, use abs instead of squaring.
            # adjusting the constant here to make the bA values ~similar across curved/angle filters.
            xRotBendL = xRotL + bA*3 * np.abs(yRotL)
            
        yRotBendL = yRotL

        """make the DC free""" 
        tmpgaussPartA = np.exp(-0.5*(kA)**2*((xRotBendL/self.sigmaXbend)**2 + (yRotBendL/(self.mA*self.sigmaYbend))**2))
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
        gaussPartA = np.exp(-0.5*(kA)**2*((xRotBendL/self.sigmaXbend)**2 + (yRotBendL/(self.mA*self.sigmaYbend))**2))
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

            for kA, bA, alphaA in itertools.product(self.scale_values, self.bend_values, self.orient_values):

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
                        'lin_freq':rect_freq_kernels, 'lin_space':rect_space}
        self.rect_kernel_pars = rect_kernel_pars
        self.curv_kernel_pars = curv_kernel_pars
        self.lin_kernel_pars = lin_kernel_pars
        
    
    def __patchnorm__(self,image):
        """ make sure it is gray scale image in range from 0 - 255 """ 
        if np.max(image) <=1:
            image = 255*image/np.max(image)
        orig_size = image.shape
        
        if image.shape[0]%3 == 0:
            patch_size = 3
        else:
            patch_size = 4
        self.patch_size = patch_size
        
        """create patches with the patch_size"""
        patches = view_as_windows(image, (patch_size,patch_size), patch_size)

        """ caculate norm of the local patches """ 
        local_norm = np.sqrt(np.einsum('ijkl->ij',patches**2))
        local_norm[local_norm<1] = 1

        """normalize local patches """ 
        normed_patches = patches/local_norm[:,:,np.newaxis,np.newaxis]

        """reshape normalized local patch to original shape """ 
#         local_normed_image = normed_patches.transpose(0,2,1,3).reshape(-1,normed_patches.shape[1]*normed_patches.shape[3])
        local_normed_image = rearrange(normed_patches,'h w c d -> (h c) (w d)')
        
        return {'local_norm':local_norm, 'local_normed_image':local_normed_image, 
                'total_local_norm':np.sqrt(local_norm.sum())}

    
    def process_images(self, files, save_filename):
        images_list, fft_images_list = [],[]
        self.files = files
        self.curv_values     = []
        self.rect_values     = []
        self.curv_max        = []
        self.rect_max        = []
        self.curv_unique     = []
        self.rect_unique     = []    
        folder_name = os.path.dirname(files[0])
        print(f'processing {len(files)} images...')
        best_kernel_each_image = []
        for i in tqdm.tqdm(range(0, len(files), self.file_block_size)):
            block_files   = files[i:i + self.file_block_size]
            _, fft_images = self.__read_images__(block_files)
            self.__calculate_curv_rect_values__(fft_images)
            bk = self.__get_best_kernel__(fft_images)
            best_kernel_each_image.extend(bk)
            
        self.curv_max    = np.dstack(self.curv_max)
        self.curv_unique = np.dstack(self.curv_unique)
        
        self.rect_max    = np.dstack(self.rect_max)
        self.rect_unique = np.dstack(self.rect_unique)
        
        self.best_kernel_pars = self.all_kernel_pars[best_kernel_each_image,:]
        
        df = pd.DataFrame({'files':self.files, 
                           'curv_values':self.curv_values, 
                           'rect_values':self.rect_values,
                           'best_scale': self.best_kernel_pars[:,0],
                           'best_bend': self.best_kernel_pars[:,1],
                           'best_orient': self.best_kernel_pars[:,2], 
                           'best_is_curved': self.best_kernel_pars[:,3]})    
        df.to_csv(save_filename, index=False)
        
        
    def __read_images__(self,file_block):  
        images_list,fft_images_list = [],[]    
        for image_name in file_block:
            orig_image = skio.imread(image_name)
            image = resize(orig_image, (self.image_size,self.image_size))

            if not self.proc_cluster_ims:
                image = rgb2gray(image)*255            
                patch_processed_image = self.__patchnorm__(image)
                output_image          = patch_processed_image['local_normed_image']
            else:
                output_image = image
                
            fft_image = np.fft.fft2(output_image)
            images_list.append(output_image)
            fft_images_list.append(fft_image)  
            
        self.images.append(images_list) 
        self.fft_images.append(fft_images_list) 
        return images_list, fft_images_list

    def __get_max_image__(self,fft_image_list,kernel_list):
        """image x, image y, kernel dimension, all images (4D array)"""
        all_kernels = np.dstack(kernel_list)
        
        """calculate kernel norm for normalization"""
        all_kernels_power =  np.einsum('ijk,ijk->k',np.abs(all_kernels),np.abs(all_kernels))
        all_kernels_power =  np.sqrt(all_kernels_power)

        """stack fft image list to a 3d array"""
        fft_images        = np.dstack(fft_image_list)
        all_conved_images = np.abs(np.fft.ifft2(fft_images[:,:,np.newaxis,:]*all_kernels[:,:,:,np.newaxis],axes=(0,1)))
        all_conved_images = np.power(all_conved_images,1/2) ## power correction
        all_conved_images = all_conved_images/all_kernels_power[np.newaxis, np.newaxis,:,np.newaxis]
    
        max_images = np.max(all_conved_images,axis=2)
        return max_images  
 
    
    def __calculate_curv_rect_values__(self, fft_image_list):
        curv_max_response = self.__get_max_image__(fft_image_list, self.kernels['curv_freq'])
        rect_max_response = self.__get_max_image__(fft_image_list, self.kernels['rect_freq'])
        
        x, y,_ = curv_max_response.shape        
        self.curv_max.append(curv_max_response) 
        self.rect_max.append(rect_max_response) 
        
        curv_unique = np.where(curv_max_response>rect_max_response, curv_max_response, 0)
        curv_values = np.einsum('ijk->k',curv_unique)
        self.curv_unique.append(curv_unique)
        
        rect_unique = np.where(rect_max_response>curv_max_response, rect_max_response, 0)
        rect_values = np.einsum('ijk->k',rect_unique)
        self.rect_unique.append(rect_unique)
        
        self.curv_values.extend(curv_values/(2*(x/2)**2))
        self.rect_values.extend(rect_values/(2*(x/2)**2))
        
        
    def __get_best_kernel__(self,fft_image_list):
                
        rect_kernel_list = self.kernels['rect_freq']       
        curv_kernel_list = self.kernels['curv_freq']
        lin_kernel_list = self.kernels['lin_freq']
        
        rect_kernel_pars = self.rect_kernel_pars
        curv_kernel_pars = self.curv_kernel_pars        
        lin_kernel_pars = self.lin_kernel_pars
        
        self.all_kernel_pars = np.concatenate([rect_kernel_pars, curv_kernel_pars, lin_kernel_pars], axis=0)

        """image x, image y, kernel dimension, all images (4D array)"""
        all_kernels = np.concatenate([np.dstack(rect_kernel_list), \
                                      np.dstack(curv_kernel_list), \
                                      np.dstack(lin_kernel_list)], axis=2)
        
        """calculate kernel norm for normalization"""
        all_kernels_power =  np.einsum('ijk,ijk->k',np.abs(all_kernels),np.abs(all_kernels))
        all_kernels_power =  np.sqrt(all_kernels_power)

        """stack fft image list to a 3d array"""
        fft_images        = np.dstack(fft_image_list)
        all_conved_images = np.abs(np.fft.ifft2(fft_images[:,:,np.newaxis,:]*all_kernels[:,:,:,np.newaxis],axes=(0,1)))
        all_conved_images = np.power(all_conved_images,1/2) ## power correction
        all_conved_images = all_conved_images/all_kernels_power[np.newaxis, np.newaxis,:,np.newaxis]
    
        # take max of each convolved image, and choose which kernel gave the biggest max activation.
        max_each_kernel = np.max(np.max(all_conved_images, 0),0)
        best_kernel_each_image = np.argmax(max_each_kernel, axis=0)
            
        return best_kernel_each_image
 
        
    def show_kernel_example(self):
        fig,ax = plt.subplots(2,2,figsize=(8,6))
        ax = ax.flat
        ax[0].imshow(self.kernels['curv_space'][100])
        ax[0].set(title='curvilinear kernel')
        ax[1].imshow((np.log(np.abs(self.kernels['curv_freq'][100]))))
        ax[1].set(title='power of the curvilinear kernel')
                                
        ax[2].imshow(self.kernels['rect_space'][15])
        ax[2].set(title='rectilinear kernel')
        ax[3].imshow((np.log(np.abs(self.kernels['rect_freq'][15]))))
        ax[3].set(title='power of the curvilinear kernel')
        
        plt.tight_layout()
        plt.show()

    def show_curv_rect_example(self):
        fig, ax = plt.subplots(2,2,figsize=(8,6))
        ax = ax.flat
        ax[0].imshow(np.fft.fftshift(self.curv_max[:,:,0]))
        ax[0].set(title='a max curvilinear image')
        
        ax[1].imshow(np.fft.fftshift(self.rect_max[:,:,0]))
        ax[1].set(title='a max rectilinear image')
        
        ax[2].imshow(np.fft.fftshift(self.curv_unique[:,:,0]))
        ax[2].set(title='a unique curvilinear image')
        
        ax[3].imshow(np.fft.fftshift(self.rect_unique[:,:,0]))
        ax[3].set(title='a unique rectilinear image')
        plt.tight_layout()
        plt.show()
                               
            
    def show_correlation(self):
        tmp_corr = stats.pearsonr(self.curv_values, self.rect_values)
        fig,ax = plt.subplots(1,1)
        ax.scatter(self.curv_values,self.rect_values)
        ax.set(xlabel='curvilinear values',
               ylabel='rectilinear values',
               title=f'correlation:{tmp_corr[0]:.3f}')
        
        plt.tight_layout()
        plt.show()
        
        
if __name__ == ('__main__'):
    
    if proc_cluster_ims:

        folder = os.path.join(default_paths.sketch_token_feat_path, 'cluster_ims')
        file_list = [os.path.join(folder,'clust%d.png'%ii) for ii in range(150)]
        save_filename = os.path.join(default_paths.sketch_token_feat_path, 'cluster_ims_curv_rect_values.csv')

    else:

        image_folder = '/lab_data/tarrlab/common/datasets/NSD_images/images/';
        from utils import nsd_utils
        subject_df = nsd_utils.get_subj_df(1);
        image_ind = np.random.choice(np.arange(0,10000), 100)
        coco_id = np.array(subject_df['cocoId'])[image_ind]
        file_list = [os.path.join(image_folder,'%d.jpg'%cid) for cid in coco_id]

        save_filename = os.path.join(default_paths.sketch_token_feat_path, 'test_random_cocoims_curv_rect_values.csv')


    curvrect_score= CurvRectValues(proc_cluster_ims)
    curvrect_score.process_images(file_list, save_filename)