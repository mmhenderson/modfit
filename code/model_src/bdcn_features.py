import numpy as np
import sys
import torch
import time
import torch.nn as nn
from utils import prf_utils, torch_utils, texture_utils, nms_utils
bdcn_path = '/user_data/mmhender/toolboxes/BDCN/'
sys.path.append(bdcn_path)
import bdcn

class bdcn_feature_extractor(nn.Module):
    
    def __init__(self, pretrained_model_file, device, aperture_rf_range = 1.1, n_prf_sd_out = 2, \
                 batch_size=10, map_ind = -1, mult_patch_by_prf=True, downsample_factor = 1, do_nms = False):
        
        super(bdcn_feature_extractor, self).__init__()
        
        self.pretrained_model_file = pretrained_model_file
        self.device = device
        self.load_model_file()
          
        self.aperture_rf_range = aperture_rf_range
        self.n_prf_sd_out = n_prf_sd_out
        self.batch_size = batch_size
        self.map_ind = map_ind
        self.mult_patch_by_prf = mult_patch_by_prf
        if downsample_factor<1:
            raise ValueError('downsample factor must be >= 1')
        self.downsample_factor = downsample_factor
        self.do_nms = do_nms        
        self.fmaps = None
             
    def load_model_file(self):
        
        model = bdcn.BDCN()
        model.load_state_dict(torch.load(self.pretrained_model_file))
        model = model.to(self.device)

        self.model = model
        
    def get_maps(self, images):
        
        print('Running BDCN contour feature extraction...')
        print('Images array shape is:')
        print(images.shape)
        t = time.time()
       
        maps_each_scale, names = get_bdcn_maps(self.model, images, self.batch_size, self.map_ind)
        maps = torch_utils._to_torch(maps_each_scale[0], device=self.device)

        if not self.downsample_factor==1:            
            orig_size = np.array(maps.shape[2:4])
            downsampled_size = np.ceil(orig_size/self.downsample_factor).astype('int')
            resized_maps = torch.nn.functional.interpolate(maps, \
                                                           size=(downsampled_size[0], downsampled_size[1]), \
                                                           mode = 'bilinear')
            maps = resized_maps
            print('Downsampled by factor of %.2f, new size is:'%self.downsample_factor)
            print(maps.shape)
                    
        maps = torch.sigmoid(maps)
        
        if self.do_nms:
            print('Applying non-maximum suppression to edge maps (can be slow...)')
            maps = nms_utils.apply_nms(maps)
        
        self.fmaps = maps
        
        print('Final array shape is:')
        print(maps.shape)
            
        elapsed =  time.time() - t
        print('time elapsed = %.5f'%elapsed)        
        
    def clear_maps(self):
        
        print('Clearing BDCN contour features from memory.')
        self.fmaps = None    
    
    def forward(self, images, prf_params):
        
        if self.fmaps is None:
            self.get_maps(images)
        else:
            assert(images.shape[0]==self.fmaps.shape[0])

        maps = self.fmaps  
        x,y,sigma = prf_params
        print('pRF [x,y,sigma]:')
        print([x,y,sigma])
        n_pix = maps.shape[2]

         # Define the RF for this "model" version
        prf = torch_utils._to_torch(prf_utils.make_gaussian_mass(x, y, sigma, n_pix, size=self.aperture_rf_range, \
                                  dtype=np.float32)[2], device=self.device)

        if self.mult_patch_by_prf:
            minval = torch.min(prf)
            maxval = torch.max(prf-minval)
            prf_scaled = (prf - minval)/maxval
            # Multiply the feature map by gaussian pRF weights, before cropping
            maps = maps * prf_scaled
        
        # Crop the patch +/- n SD away from center
        bbox = texture_utils.get_bbox_from_prf(prf_params, prf.shape, self.n_prf_sd_out, min_pix=None, verbose=False, force_square=False)
        print('bbox to crop is:')
        print(bbox)
        maps_cropped = maps[:,:,bbox[0]:bbox[1], bbox[2]:bbox[3]]
        
        print('[min max] of first image patch is:')
        print([torch.min(maps_cropped[0,0,:,:]), torch.max(maps_cropped[0,0,:,:])])
        
        # return [ntrials x nfeatures]
        # Note this reshaping goes in "C" style order by default
        features = torch.reshape(maps_cropped, [maps_cropped.shape[0], np.prod(maps_cropped.shape[1:])])
        
        return features
    

def get_bdcn_maps(model, images, batch_size=10, map_inds=None):
            
    device = list(model.parameters())[0].device
 
    if map_inds is not None:
        if np.isscalar(map_inds):
            map_inds = [map_inds]
    else:
        map_inds = np.arange(0,11)
        
    n_images = images.shape[0]
    n_batches = int(np.ceil(n_images/batch_size))

    for bb in range(n_batches):

        batch_inds = np.arange(batch_size * bb, np.min([batch_size * (bb+1), n_images]))
        image_batch = images[batch_inds,:,:,:]
        
        out = model(prep_for_bdcn(image_batch, device))    
        out = [oo.detach().cpu().numpy() for oo in out]

        p1_1, p2_1, p3_1, p4_1, p5_1, p1_2, p2_2, p3_2, p4_2, p5_2, fuse = out
        
        # undoing the sums they do at end of forward pass, to get out the raw feature maps at each scale
        # p1_1 = s1
        # p2_1 = s2 + o1
        # p3_1 = s3 + o2 + o1
        # p4_1 = s4 + o3 + o2 + o1
        # p5_1 = s5 + o4 + o3 + o2 + o1

        # p1_2 = s11 + o21 + o31 + o41 + o51
        # p2_2 = s21 + o31 + o41 + o51
        # p3_2 = s31 + o41 + o51
        # p4_2 = s41 + o51
        # p5_2 = s51

        s1_1 = p1_1
        s2_1 = p2_1 - s1_1
        s3_1 = p3_1 - s1_1 - s2_1
        s4_1 = p4_1 - s1_1 - s2_1 - s3_1
        s5_1 = p5_1 - s1_1 - s2_1 - s3_1 - s4_1

        s5_2 = p5_2
        s4_2 = p4_2 - s5_2
        s3_2 = p3_2 - s5_2 - s4_2
        s2_2 = p2_2 - s5_2 - s4_2 - s3_2
        s1_2 = p1_2 - s5_2 - s4_2 - s3_2 - s2_2

        maps_each_scale_this_batch = [s1_1, s2_1, s3_1, s4_1, s5_1, s1_2, s2_2, s3_2, s4_2, s5_2, fuse]

        if bb==0:
            maps_each_scale = [maps_each_scale_this_batch[mi] for mi in map_inds]
        else:
            for ii, mi in enumerate(map_inds):                    
                maps_each_scale[ii] = np.concatenate((maps_each_scale[ii], maps_each_scale_this_batch[mi]), axis=0)
            

    names1 = ['s2d_%d'%(ii+1) for ii in range(5)]
    names2 = ['d2s_%d'%(ii+1) for ii in range(5)]
    names = list(np.concatenate([names1, names2, ['BDCN: fused']]))

    return maps_each_scale, names


def prep_for_bdcn(image_data, device):
    
    if image_data.shape[1]==1:
        image_data = np.tile(image_data, [1,3,1,1])
    else:
        # RGB to BGR
        image_data = image_data[:,-1::,:,:]   
    mean_bgr = np.expand_dims(np.array([104.00699, 116.66877, 122.67892]), [0,2,3])
    dat = image_data * 255 - mean_bgr
    dat = torch.tensor(dat, dtype=torch.float32).to(device)

    return dat


