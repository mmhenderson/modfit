import numpy as np
import os, sys
import pandas as pd
import time, h5py
import PIL.Image

from utils import default_paths, nsd_utils

# code to work with stimuli from the fLoc category localizer (from Grill-Spector lab)
# access raw stimuli here:
# http://vpnl.stanford.edu/fLoc/

categories = ['word','number',
              'body','limb',
              'adult','child',
              'corridor','house',
              'car','instrument']

domains = ['characters','bodies','faces','places','objects']

n_each_categ = 144;

floc_image_root = default_paths.floc_image_root


def prep_images(newsize=(240,240)):
    
    n_images = n_each_categ * len(categories)
    ims_brick = np.zeros((n_images, 1, newsize[0], newsize[1]))

    categ_list = []
    domain_list = []
    instance_list = []
    filename_list = []

    count = -1;

    for cc, categ in enumerate(categories):

        print('processing %s'%categ)
        dd = int(np.floor(cc/2))
        
        for ii in np.arange(1,n_each_categ+1):

            count += 1

            filename = os.path.join(floc_image_root,'%s-%d.jpg'%(categories[cc], ii))
            if ii==1:       
                print('loading from %s'%filename)
            im = PIL.Image.open(filename)
            im_resized = im.resize(newsize, resample=PIL.Image.ANTIALIAS)

            ims_brick[count,0,:,:] = im_resized

            categ_list += [categ]
            domain_list += [domains[dd]]
            instance_list += [ii]
            filename_list += [filename]
            
    
    floc_labels = pd.DataFrame({'category': categ_list, \
                                'domain': domain_list, \
                                'instance': instance_list, \
                                'filename': filename_list})

    csv_filename =  os.path.join(floc_image_root,'floc_image_labels.csv')

    print('Writing csv labels to %s'%csv_filename)
    floc_labels.to_csv(csv_filename, index=False)
        
    
    fn2save = os.path.join(floc_image_root,'floc_stimuli_%d.h5py'%newsize[0])

    print('Writing resized images to %s\n'%fn2save)

    t = time.time()
    with h5py.File(fn2save, 'w') as data_set:
        dset = data_set.create_dataset("stimuli", np.shape(ims_brick), dtype=np.uint8)
        data_set['/stimuli'][:,:,:] = ims_brick
        data_set.close()  
    elapsed = time.time() - t

    print('Took %.5f sec to write file'%elapsed)

    return

def load_floc_images(npix=240):
    
    image_filename = os.path.join(floc_image_root, 'floc_stimuli_%d.h5py'%npix)
    ims = nsd_utils.load_from_hdf5(image_filename)

    ims = ims.astype(np.float32) / 255
    
    return ims