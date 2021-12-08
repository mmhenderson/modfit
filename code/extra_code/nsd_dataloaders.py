# import basic modules
import os
import time
import numpy as np
import h5py
import pickle
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from utils import default_paths, torch_utils

"""
Some OLD code to create pytorch dataloaders from NSD images.
Note I am not actually using any of this for imStat right now, but could be useful in the future.
"""


class nsd_dataset(Dataset):
    
    """
    A custom PyTorch dataset to work with NSD data. Load batches of images and labels at a time.    
    Can get the dataset from http://naturalscenesdataset.org/    
    """
    
    def __init__(self, nsd_brick_file=None, nsd_meta_file=None, transform=None, device=None, nsd_inds_include=None):

        super(nsd_dataset, self).__init__()

        if nsd_brick_file is None:
            nsd_brick_file = os.path.join(default_paths.nsd_path,'nsddata_stimuli','stimuli','nsd','nsd_stimuli.hdf5')
        self.nsd_brick_file = nsd_brick_file
        
        if nsd_meta_file is None:
            nsd_meta_file = os.path.join(default_paths.nsd_path,'nsddata','experiments','nsd','nsd_stim_info_merged.pkl')               
        self.stim_info = get_nsd_info(nsd_meta_file)
        
        self.transform = transform
        
        if device is None:
            device=torch.device('cpu:0')
        self.device=device
        
        # can choose just a subset of the images to include in the dataset, if desired. 
        # use the indices into 73000 NSD images to specify which ims to include.
        if nsd_inds_include is None:
            nsd_inds_include = np.arange(0,len(self.stim_info))            
        self.nsd_inds_include = np.array(nsd_inds_include)
#         print(self.nsd_inds_include)
        
    def __len__(self):

        return len(self.nsd_inds_include)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        nsd_indices = self.nsd_inds_include[idx]
#         print(nsd_indices)
        image_batch = load_nsd_ims(nsd_indices=nsd_indices, nsd_brick_file=self.nsd_brick_file)
        image_batch = np.moveaxis(image_batch, [3],[1])   
        image_batch = torch_utils._to_torch(image_batch, device=self.device).to(torch.float64)/255
        
        print('\nBefore transformation:')
        print ('block size:', image_batch.shape, ', dtype:', image_batch.dtype, ', device:', image_batch.device,', value range: (',\
        torch_utils.get_value(torch.min(image_batch[0])), torch_utils.get_value(torch.max(image_batch[0])), ')')
                
        if self.transform:
            image_batch = self.transform(image_batch)
            
        print('\nAfter transformation:')
        print ('block size:', image_batch.shape, ', dtype:', image_batch.dtype, ', device:', image_batch.device,', value range: (',\
        torch_utils.get_value(torch.min(image_batch[0])), torch_utils.get_value(torch.max(image_batch[0])), ')')
  
        coco_id_batch = self.stim_info['cocoId'][nsd_indices]
        nsd_id_batch = self.stim_info['nsdId'][nsd_indices]
        
        item = {'image':image_batch, 'coco_id': coco_id_batch, 'nsd_id': nsd_id_batch}

        return item

class nsd_dataloader:
    
    """
    A custom dataloader-style class to load batches of NSD images. 
    Capable of loading several images at a time, which is faster than looping through individual images.
    This should behave similarly to PyTorch 'DataLoader' class. 
    """
    
    def __init__(self, nsd_dataset, batch_size=100, shuffle=False):
        
        self.dataset = nsd_dataset        
        self.batch_size = int(batch_size)
        self.n_total_images = int(len(self.dataset))
        self.n_batches = int(np.ceil(self.n_total_images/self.batch_size))
        self.shuffle = shuffle
        
        self.initialize_sequence()
            
    def __len__(self):     
        
        return self.n_batches
    
    def initialize_sequence(self):
        
        print('initializing iterator at first batch for nsd dataset')
        self.sample_order = np.arange(0,self.n_total_images)
        if self.shuffle:
            np.random.shuffle(self.sample_order)
        self.batch_num = -1
            
    def __iter__(self):
        
        self.initialize_sequence()       
        return self
    
    def __next__(self):
        
        self.batch_num += 1
        if self.batch_num<self.n_batches:            
            batch_inds = np.arange(self.batch_size*self.batch_num, np.min([self.batch_size*self.batch_num  + self.batch_size, self.n_total_images]), 1)
#             print('batch inds:')
#             print(batch_inds)
#             print(len(batch_inds))
            samples_to_use = self.sample_order[batch_inds]
#             print('sample inds:')
#             print(samples_to_use)
#             print(len(samples_to_use))
            return self.dataset[samples_to_use]
        else:
            raise StopIteration
            
            
def get_transform(input_height=224):

    """ 
    Transform to be used before sending NSD images into a pre-trained neural network.
    the normalization values were copied from https://pytorch.org/hub/pytorch_vision_alexnet/
    Should match pre-training data for these models.
    """
    
    # create a transformation that will normalize the image intensity (z-score)
    transform = transforms.Compose([transforms.Resize(input_height), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    return transform

            

def get_nsd_info(nsd_meta_file = None):

    """
    Load file containing metadata for all NSD images.
    """
    
    if nsd_meta_file is None:
        nsd_meta_file = os.path.join(default_paths.nsd_path,'nsddata','experiments','nsd','nsd_stim_info_merged.pkl')
        
    with open(nsd_meta_file,'rb') as f:
        stim_info = pickle.load(f,encoding="latin1")
        
    return stim_info

def load_nsd_ims(nsd_indices=None, nsd_brick_file = None):
    
    """
    Load a specified subset of the NSD images.
    """
    
    if nsd_brick_file is None:
        nsd_brick_file = os.path.join(default_paths.nsd_path,'nsddata_stimuli','stimuli','nsd','nsd_stimuli.hdf5')
      
    if nsd_indices is not None and not isinstance(nsd_indices, np.ndarray) and not isinstance(nsd_indices, list):
        nsd_indices = [nsd_indices]
  
    print('\nLoading images from brick file at %s'%nsd_brick_file)
    t = time.time()
    with h5py.File(nsd_brick_file, "r") as f:
        if nsd_indices is not None and np.all(np.diff(nsd_indices)==1) or len(nsd_indices)==1:
            nsd_ims = np.array(f['imgBrick'][nsd_indices,:,:,:])
        elif nsd_indices is not None and len(nsd_indices)<1000:
            nsd_ims = np.zeros([len(nsd_indices),425,425,3])
            for ii, nsd_ind in enumerate(nsd_indices):
                nsd_ims[ii,:,:,:] = np.array(f['imgBrick'][nsd_ind,:,:,:])       
        elif nsd_indices is not None:
            nsd_ims = np.array(f['imgBrick'])
            nsd_ims = nsd_ims[nsd_indices,:,:,:]
        else:
            nsd_ims = np.array(f['imgBrick'])
    elapsed = time.time() - t;
    print('elapsed time (loading ims): %.2f'%elapsed)
    
    print ('block size:', nsd_ims.shape, ', dtype:', nsd_ims.dtype, ', value range:',\
    np.min(nsd_ims[0]), np.max(nsd_ims[0]))
    
    return nsd_ims
