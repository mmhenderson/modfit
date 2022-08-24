import argparse
import numpy as np
import sys, os
import time
import h5py
import torch

#import custom modules
from utils import color_utils, nsd_utils, prf_utils, default_paths, floc_utils, torch_utils
from model_fitting import initialize_fitting

try:
    device = initialize_fitting.init_cuda()
except:
    device = 'cpu:0'

def extract_color_features(image_data, batch_size=100, \
                             which_prf_grid=5, debug=False):
    
    n_pix = image_data.shape[2]
    n_images = image_data.shape[0]
    
    n_batches = int(np.ceil(n_images/batch_size))
    
    # Params for the spatial aspect of the model (possible pRFs)
    models = initialize_fitting.get_prf_models(which_grid = which_prf_grid)    
    n_prfs = models.shape[0]
    
    n_features = 4 # [L*, a*, b*, saturation]
    
    print('number of features total: %d'%n_features)
    
    features_each_prf = np.zeros((n_images, n_features, n_prfs), dtype=np.float32)

    for bb in range(n_batches):
        
        if debug and bb>1:
            continue
            
        print('processing batch %d of %d'%(bb, n_batches))
        
        sys.stdout.flush()
       
        batch_inds = np.arange(bb*batch_size, np.minimum((bb+1)*batch_size, n_images))
    
        st = time.time()
        # color channels will be last dimension of this array
        image_batch = np.moveaxis(image_data[batch_inds,:,:,:], [0,1],[3,2])

        st_cielab = time.time()
        
        image_lab = color_utils.rgb_to_CIELAB(image_batch, device=device)
        
        elapsed_cielab = time.time() - st_cielab
        print('took %.5f sec to get cielab features'%elapsed_cielab)
        
        image_sat = color_utils.get_saturation(image_batch)[:,:,None,:]
              
        # 4 color feature channels concatenated here
        fmaps_batch = np.moveaxis(np.concatenate([image_lab, image_sat], axis=2), [3], [0])

        print('size of fmaps_batch')
        print(fmaps_batch.shape)
        
        elapsed = time.time() - st
        print('took %.5f s to gather color feature maps'%elapsed)
        
        fmaps_batch = torch_utils._to_torch(fmaps_batch, device=device)
        
        for mm in range(n_prfs):

            x,y,sigma = models[mm,:]
            
            prf = torch_utils._to_torch(prf_utils.gauss_2d(center=[x,y], sd=sigma, \
                                   patch_size=n_pix, aperture=1.0, dtype=np.float32), device=device)
            
            # weighted sum of color feature values in the pRF
            features_batch = torch.tensordot(fmaps_batch, prf, dims=[[1,2],[0,1]])
            
            features_each_prf[batch_inds,:,mm] = torch_utils.get_value(features_batch)

            
        elapsed = time.time() - st
        print('took %.5f s to multiply maps by pRFs'%elapsed)
        sys.stdout.flush()
            
    return features_each_prf
      
    
def proc_one_subject(subject, args):

    color_feat_path = default_paths.color_feat_path
       
    if args.debug:
        color_feat_path = os.path.join(color_feat_path,'DEBUG')
       
    if not os.path.exists(color_feat_path):
        os.makedirs(color_feat_path)
     
    # Load and prepare the image set to work with 
    if subject==999:
        # 999 is a code i am using to indicate the independent set of coco images, which were
        # not actually shown to any NSD participants
        from utils import coco_utils
        image_data = coco_utils.load_indep_coco_images(n_pix=240)
    else: 
        # load all images for the current subject, 10,000 ims
        image_data = nsd_utils.get_image_data(subject)  

    filename_save = os.path.join(color_feat_path, \
                               'S%d_cielab_plus_sat_grid%d.h5py'%(subject, args.which_prf_grid))
     
    features_each_prf = extract_color_features(image_data, batch_size=args.batch_size, \
                                         which_prf_grid=args.which_prf_grid, debug=args.debug)
    
    save_features(features_each_prf, filename_save)

    
def proc_other_image_set(image_set, args):
       
    color_feat_path = default_paths.color_feat_path
    
    if args.debug:
        color_feat_path = os.path.join(color_feat_path,'DEBUG')
        
    if not os.path.exists(color_feat_path):
        os.makedirs(color_feat_path)
        
    if image_set=='floc':
        image_data = floc_utils.load_floc_images(npix=240)
    else:
        raise ValueError('image set %s not recognized'%image_set)
        
    if image_data.shape[1]==1:
        image_data = np.tile(image_data, [1,3,1,1])
        
    filename_save = os.path.join(color_feat_path, \
                               '%s_cielab_plus_sat_grid%d.h5py'%(image_set, args.which_prf_grid))
        
    features_each_prf = extract_color_features(image_data, batch_size=args.batch_size, \
                                         which_prf_grid=args.which_prf_grid, debug=args.debug)
    
    save_features(features_each_prf, filename_save)
                                    
    
def save_features(features_each_prf, filename_save):
    
    print('Writing prf features to %s\n'%filename_save)
    
    t = time.time()
    with h5py.File(filename_save, 'w') as data_set:
        dset = data_set.create_dataset("features", np.shape(features_each_prf), dtype=np.float32)
        data_set['/features'][:,:,:] = features_each_prf
        data_set.close()  
    elapsed = time.time() - t
    
    print('Took %.5f sec to write file'%elapsed)
    

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=0,
                    help="number of the subject, 1-8")
    parser.add_argument("--image_set", type=str,default='none',
                    help="name of the image set to use (if not an NSD subject)")
    parser.add_argument("--batch_size", type=int,default=100,
                    help="batch size")
    
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which version of prf grid to use")

    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    
    args = parser.parse_args()
    
    if args.subject==0:
        args.subject=None
    if args.image_set=='none':
        args.image_set=None
        
    args.debug = (args.debug==1)     
    
    if args.subject is not None:
        
        proc_one_subject(subject = args.subject, args=args)
        
    elif args.image_set is not None:
        
        proc_other_image_set(image_set=args.image_set, args=args)
        
    