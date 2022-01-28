
import skimage.io as skio
from skimage.transform import resize

import numpy as np
import pandas as pd 
from einops import rearrange, reduce, repeat

import itertools
import os
import tqdm

from utils import default_paths

"""
Code to perform curvature analysis on images corresponding to each sketch token 
feature (clusters in curvature space).
Original version from Xiaomin Yue, related to paper:
"Curvature processing in human visual cortical areas" (2020), Neuroimage.
Modified for this project by MMH.
Saves out a csv file listing the approx curvature params of each feature.
"""

class CurvRectValues:  
   
    def __init__(self):
        self.files           = []
        self.image_size      = 128
        self.randstate       = 49
        self.file_block_size = 20
        self.images          = []
        self.fft_images      = []
        self.cg_parameters   = {'kA':0.3,'bend':0.01,'orientation':45*np.pi/180}
        self.curv_values     = []
        self.rect_values     = []
        self.curv_max        = []
        self.rect_max        = []
        self.curv_unique     = []
        self.rect_unique     = []
        kA_scales = [9,12]
        self.__generate_kernels__(kA_scales)
            
    def __bananakernel__(self):
        """
        input: cg_parameters is a dictionary, including kA, bA, alphaA, mA, sigmaXbend, sigmaYbend, 
                                               xA_half, yA_half, xA_shift, yA_shift
                                               
        the function is used to generate banana wavelet kernels.  The kernels
        can be used to filter a image to quantify curvatures.

        kA:          length of the wave vector K
        bA:          bending value b
        alphaA:      direction of the wave vector
        mA:          magnitude value m
        xA_half:     x-size
        yA_half:     y-size
        xA_shift:    center shift in x direction
        yA_shift:    center shift in y direction

        for references:
        preFactorA:  pre-factor p
        DCPartRealA: real dc-part
        DCPartImagA: imaginary dc-part
        gaussPartA:  Gaussian part    

        filter requency: (cycle/object) = xA*kA/(2*pi*mA)
        kernel size: 2*4*sigmaYbend*mA*(1/kA)

        return SpaceKernel, FreKernel

        last updated 5/23/2021
        last updated 3/23/3021
        """
        kA         = self.cg_parameters['kA']
        bA         = self.cg_parameters['bend']    
        alphaA     = self.cg_parameters.get('orientation',45*np.pi/180)
        mA         = self.cg_parameters.get('mA',3)
        sigmaXbend = self.cg_parameters.get('sigmaXbend',2)
        sigmaYbend = self.cg_parameters.get('sigmaYbend',2)
        xA_half    = self.cg_parameters.get('xA_half',self.image_size/2)
        yA_half    = self.cg_parameters.get('yA_half',self.image_size/2)
        xA_shift   = self.cg_parameters.get('x_shift',0)
        yA_shift   = self.cg_parameters.get('y_shift',0)
        
        if isinstance(bA, complex):
            print('bA has to be real number. However your input is a complex number')
            bA = np.real(bA)

        if any(x<=0 for x in np.array([kA, mA])) or any(np.isnan(np.array([kA, bA, alphaA, mA]))):     
            out_ranage_value = 10**-20
            SpaceKernel = np.ones((2*xA_half,2*yA_half))*out_ranage_value
            FreKernel   = np.ones((2*xA_half,2*yA_half))*out_ranage_value
            return SpaceKernel, FreKernel

        kernel_size = 2*xA_half
        if kernel_size%2 !=0:
            kernel_size = kernel_size + 1
        [xA, yA] = np.meshgrid(np.arange(-kernel_size/2, kernel_size/2,1),np.arange(-kernel_size/2, kernel_size/2,1)) 
        xA = xA - xA_shift
        yA = yA - yA_shift

        xRotL = np.cos(alphaA)*xA + np.sin(alphaA)*yA 
        yRotL = np.cos(alphaA)*yA - np.sin(alphaA)*xA

        xRotBendL = xRotL + bA * (yRotL)**2
        yRotBendL = yRotL

        """make the DC free""" 
        tmpgaussPartA = np.exp(-0.5*(kA)**2*((xRotBendL/sigmaXbend)**2 + (yRotBendL/(mA*sigmaYbend))**2))
        tmprealteilL  = 1*tmpgaussPartA*(np.cos(kA*xRotBendL) - 0)
        tmpimagteilL  = 1*tmpgaussPartA*(np.sin(kA*xRotBendL) - 0)

        numeratorRealL = np.sum(tmprealteilL)
        numeratorImagL = np.sum(tmpimagteilL)
        denominatorL   = np.sum(tmpgaussPartA)

        DCValueAnalysis = np.exp(-0.5 * sigmaXbend * sigmaXbend)
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
        gaussPartA = np.exp(-0.5*(kA)**2*((xRotBendL/sigmaXbend)**2 + (yRotBendL/(mA*sigmaYbend))**2))
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
                                  
        
    def __generate_kernels__(self, kA_scales=[1,5,8]):
        
        all_kA = [2*np.pi/((np.sqrt(2))**x) for x in kA_scales]
        bends = [0, 0.02,0.07,0.10,0.18,0.45]
        alphaA = np.linspace(0,2*np.pi,73)[0:72]
        
        curv_freq_kernels,rect_freq_kernels,curv_space,rect_space = [],[],[],[]
        
        curv_kernel_pars = np.zeros((len(all_kA)*(len(bends)-1)*len(alphaA), 3))
        rect_kernel_pars = np.zeros((len(all_kA)*(1)*len(alphaA), 3))
        
        ci=-1; ri=-1
        for kA, bA, orien in itertools.product(all_kA, bends, alphaA):
            self.cg_parameters['kA']          = kA
            self.cg_parameters['bend']        = bA/8
            self.cg_parameters['orientation'] = orien
            
            neuron, Freq_kernel = self.__bananakernel__()
            if bA == 0:
                rect_freq_kernels.append(Freq_kernel)
                rect_space.append(neuron.real) 
                ri+=1
                rect_kernel_pars[ri,:] = [kA, bA/8, orien]
            else:
                curv_freq_kernels.append(Freq_kernel)
                curv_space.append(neuron.real)
                ci+=1
                curv_kernel_pars[ci,:] = [kA, bA/8, orien]

        self.kernels = {'curv_freq':curv_freq_kernels, 'curv_space':curv_space,
                        'rect_freq':rect_freq_kernels, 'rect_space':rect_space}
        self.rect_kernel_pars = rect_kernel_pars
        self.curv_kernel_pars = curv_kernel_pars
      
    def process_images(self, files):
        images_list, fft_images_list = [],[]
        self.files = files
        folder_name = os.path.dirname(files[0])
        print(f'processing {len(files)} images...')
        best_kernel_each_image = []
        for i in tqdm.tqdm(range(0, len(files), self.file_block_size)):
            block_files   = files[i:i + self.file_block_size]
            _, fft_images = self.__read_images__(block_files)
            bk = self.__get_best_kernel__(fft_images)
            best_kernel_each_image.extend(bk)
            
        best_kernel_pars = self.all_kernel_pars[best_kernel_each_image,:]
        
        df = pd.DataFrame({'files':self.files, 
                           'best_scale': best_kernel_pars[:,0],
                           'best_bend': best_kernel_pars[:,1],
                           'best_orient': best_kernel_pars[:,2]})
        
        df.to_csv(f'{folder_name}_curv_rect_values.csv', index=False)
        
        
    def __read_images__(self,file_block):  
        images_list,fft_images_list = [],[]    
        for image_name in file_block:
            
            orig_image = skio.imread(image_name)
            image = resize(orig_image, (self.image_size,self.image_size))
            output_image = image
            fft_image = np.fft.fft2(output_image)
            images_list.append(output_image)
            fft_images_list.append(fft_image)  
            
        self.images.append(images_list) 
        self.fft_images.append(fft_images_list) 
        
        return images_list, fft_images_list

    def __get_best_kernel__(self,fft_image_list):
                
        rect_kernel_list = self.kernels['rect_freq']
        rect_spat_kernel_list = self.kernels['rect_space']
        curv_kernel_list = self.kernels['curv_freq']
        curv_spat_kernel_list = self.kernels['curv_space']

        curv_kernel_pars = self.curv_kernel_pars
        rect_kernel_pars = self.rect_kernel_pars
        self.all_kernel_pars = np.concatenate([rect_kernel_pars, curv_kernel_pars], axis=0)

        """image x, image y, kernel dimension, all images (4D array)"""
        all_kernels = np.concatenate([np.dstack(rect_kernel_list), np.dstack(curv_kernel_list)], axis=2)
        all_spat_kernels = np.concatenate([np.dstack(rect_spat_kernel_list), np.dstack(curv_spat_kernel_list)], axis=2)
        
        """calcuate kernel norm for normalization"""
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
 
        
if __name__ == '__main__':
    
    folder = os.path.join(default_paths.sketch_token_feat_path, 'cluster_ims')
    file_list = [os.path.join(folder,'clust%d.png'%ii) for ii in range(150)]
    curvrect_score= CurvRectValues()
    curvrect_score.process_images(file_list)
    