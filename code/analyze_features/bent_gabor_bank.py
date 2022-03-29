import numpy as np
import torch
import itertools
import matplotlib.pyplot as plt

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
            Orientation axis starts at vertical==0 and rotates counter-clockwise.
            Orientations span a full 0-360 space, because the bend values make them asymmetric. 
            For linear kernels, the filters 180 deg apart are identical.
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
 