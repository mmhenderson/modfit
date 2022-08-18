import argparse
import numpy as np
import sys, os
import time
import h5py
import torch

#import custom modules
from utils import color_utils, nsd_utils, prf_utils, default_paths, floc_utils, torch_utils
from model_fitting import initialize_fitting
from feature_extraction import pca_feats

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def extract_color_features(image_data, batch_size, prf_batch_inds, save_batch_filenames,\
                           which_prf_grid=5, debug=False):
    
    models = initialize_fitting.get_prf_models(which_grid = which_prf_grid)    
    n_prfs = models.shape[0]
    
    n_pix = image_data.shape[2]
    n_images = image_data.shape[0]
    
    # create the binary masks for each pRF, +/- 1.5 SD
    prf_masks = np.zeros((n_prfs, n_pix, n_pix),dtype=bool)
    
    n_prf_sd_out = 1.5
    
    for prf_ind in range(n_prfs):    
        prf_params = models[prf_ind,:] 
        x,y,sigma = prf_params
        aperture=1.0
        prf_mask = prf_utils.get_prf_mask(center=[x,y], sd=sigma, \
                               patch_size=n_pix, zscore_plusminus=n_prf_sd_out)
        prf_masks[prf_ind,:,:] = prf_mask==1
    
    mask_sums = np.sum(np.sum(prf_masks, axis=1), axis=1)
    assert(not np.any(mask_sums==0))
    
    n_batches = int(np.ceil(n_images/batch_size))
    
    n_features_color = 4 # [L*, a*, b*, saturation]

    n_prf_batches = len(prf_batch_inds)
    
    # separate the prfs into batches to save memory.  
    for pb in range(n_prf_batches):

        prfs_this_batch = prf_batch_inds[pb]
        
        max_features_spat = np.max(mask_sums[prfs_this_batch])
    
        n_features_total = int(max_features_spat * n_features_color)
        
        print('pRF batch %d of %d, number of features max: %d'%(pb, n_prf_batches, n_features_total))
   
        features_each_prf = np.zeros((n_images, n_features_total,len(prfs_this_batch)),dtype=np.float32)
    

        for bb in range(n_batches):

            if debug and bb>1:
                continue

            print('proc batch %d of %d'%(bb, n_batches))
            sys.stdout.flush()

            batch_inds = np.arange(bb*batch_size, np.minimum((bb+1)*batch_size, n_images))

            st = time.time()
            fmaps_batch = np.zeros((len(batch_inds), n_pix, n_pix, n_features_color))

            for ii, image_ind in enumerate(batch_inds):

                # color channels will be last dimension of this array
                image = np.moveaxis(image_data[image_ind,:,:,:], [0],[2])

                image_lab = color_utils.rgb_to_CIELAB(image)
                image_sat = color_utils.get_saturation(image)

                # 4 color feature channels concatenated here
                fmaps_batch[ii,:,:,:] = np.dstack([image_lab, image_sat])

            elapsed = time.time() - st
            print('took %.5f s to gather color feature maps'%elapsed)

            st = time.time()

            for mi, mm in enumerate(prfs_this_batch):

                x,y,sigma = models[mm,:]

                prf_mask = prf_masks[mm,:,:]
                print('size of prf mask is:')
                print(prf_mask.shape)
                print('sum of prf mask is:')
                print(np.sum(prf_mask))
                # apply the prf mask here.                 
                # the final feature dimension includes both color channels and spatial pixels.
                features_batch = fmaps_batch[:,prf_mask]
                print('shape of features batch after masking is:')
                print(features_batch.shape)
                
                features_batch = np.reshape(features_batch, [len(batch_inds),-1])
                print('shape of features batch after reshape is:')
                print(features_batch.shape)
                
                n_feat_actual = features_batch.shape[1]
                
                # fill into my big array. leave zeros in the empty spots.
                # the zero columns will be fixed after pca.
                features_each_prf[batch_inds,0:n_feat_actual,mi] = features_batch                

            elapsed = time.time() - st
            print('took %.5f s to apply pRF masks'%elapsed)
            sys.stdout.flush()
         
        # Now save the results
        save_features(features_each_prf, save_batch_filenames[pb])
    
    
def proc_one_subject(subject, args):

    feat_path = default_paths.spatcolor_feat_path
       
    if args.debug:
        feat_path = os.path.join(feat_path,'DEBUG')
       
    if not os.path.exists(feat_path):
        os.makedirs(feat_path)
     
    # Load and prepare the image set to work with 
    if subject==999:
        # 999 is a code i am using to indicate the independent set of coco images, which were
        # not actually shown to any NSD participants
        from utils import coco_utils
        image_data = coco_utils.load_indep_coco_images(n_pix=args.map_res_pix)
    else: 
        # load all images for the current subject, 10,000 ims
        image_data = nsd_utils.get_image_data(subject,npix=args.map_res_pix)  

    prf_models = initialize_fitting.get_prf_models(which_grid=args.which_prf_grid)    
    n_prfs = len(prf_models)
    
    prf_batch_size = 100; # just keeping this par fixed, seems to work ok
    n_prf_batches = int(np.ceil(n_prfs/prf_batch_size))
    
    prf_batch_inds = [np.arange(pb*prf_batch_size, np.min([(pb+1)*prf_batch_size, n_prfs])) \
                      for pb in range(n_prf_batches)]
    
    save_batch_filenames = [os.path.join(feat_path, \
                               'S%d_spatcolor_res%dpix_grid%d_prfbatch%d.h5py'%(subject, \
                                                                                args.map_res_pix, \
                                                                                args.which_prf_grid, \
                                                                                pb)) \
                     for pb in range(n_prf_batches)]
    
    extract_color_features(image_data,
                           prf_batch_inds=prf_batch_inds,\
                           save_batch_filenames=save_batch_filenames,\
                           batch_size=args.batch_size, \
                           which_prf_grid=args.which_prf_grid, \
                           debug=args.debug)
    sys.stdout.flush()
    
    pca_feats.run_pca(subject=subject, \
                          feature_type='spatcolor',\
                          which_prf_grid=args.which_prf_grid,\
                          min_pct_var=args.min_pct_var,\
                          max_pc_to_retain=args.max_pc_to_retain, \
                          save_weights = args.save_pca_weights==1, \
                          use_saved_ncomp = args.use_saved_ncomp==1, \
                          map_res_pix=args.map_res_pix, \
                          debug=args.debug)
        
    # now removing the large intermediate files, leaving only the pca versions
    for big_fn in save_batch_filenames:

        print('removing big activations file: %s'%big_fn)
        sys.stdout.flush()
        os.remove(big_fn)

        print('big file removed.')
    
def proc_other_image_set(image_set, args):
       
    feat_path = default_paths.spatcolor_feat_path
    
    if args.debug:
        feat_path = os.path.join(feat_path,'DEBUG')
        
    if not os.path.exists(feat_path):
        os.makedirs(feat_path)
        
    if image_set=='floc':
        image_data = floc_utils.load_floc_images(npix=args.map_res_pix)
    else:
        raise ValueError('image set %s not recognized'%image_set)
        
    if image_data.shape[1]==1:
        image_data = np.tile(image_data, [1,3,1,1])
        
        
    prf_models = initialize_fitting.get_prf_models(which_grid=args.which_prf_grid)    
    n_prfs = len(prf_models)
    
    prf_batch_size = 100; # just keeping this par fixed, seems to work ok
    n_prf_batches = int(np.ceil(n_prfs/prf_batch_size))
    
    prf_batch_inds = [np.arange(pb*prf_batch_size, np.min([(pb+1)*prf_batch_size, n_prfs])) \
                      for pb in range(n_prf_batches)]
    
    save_batch_filenames = [os.path.join(feat_path, \
                               '%s_spatcolor_res%dpix_grid%d_prfbatch%d.h5py'%(image_set, \
                                                                               args.map_res_pix, \
                                                                               args.which_prf_grid, \
                                                                               pb)) \
                     for pb in range(n_prf_batches)]
    
        
    extract_color_features(image_data, 
                           prf_batch_inds=prf_batch_inds,\
                           save_batch_filenames=save_batch_filenames,\
                           batch_size=args.batch_size, \
                           which_prf_grid=args.which_prf_grid, \
                           debug=args.debug)
    sys.stdout.flush()
    
    subjects_pca = np.arange(1,9)
    for ss in subjects_pca:
        pca_feats.run_pca(subject=ss, 
                          image_set=image_set, \
                          feature_type='spatcolor',\
                          which_prf_grid=args.which_prf_grid,\
                          min_pct_var=args.min_pct_var,\
                          max_pc_to_retain=args.max_pc_to_retain, \
                          save_weights = args.save_pca_weights==1, \
                          use_saved_ncomp = args.use_saved_ncomp==1, \
                          map_res_pix=args.map_res_pix, \
                          debug=args.debug)

    # now removing the large intermediate files, leaving only the pca versions
    for big_fn in save_batch_filenames:

        print('removing big activations file: %s'%big_fn)
        sys.stdout.flush()
        os.remove(big_fn)

        print('big file removed.')
        
                                    
    
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
    parser.add_argument("--map_res_pix", type=int,default=240,
                    help="batch size")
    
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which version of prf grid to use")

    parser.add_argument("--max_pc_to_retain", type=int,default=0,
                    help="max pc to retain? enter 0 for None")
    parser.add_argument("--min_pct_var", type=int,default=95,
                    help="min pct var to explain? default 95")
    parser.add_argument("--save_pca_weights", type=int,default=0,
                    help="want to save the weights to reproduce the pca later? 1 for yes, 0 for no")
    parser.add_argument("--use_saved_ncomp", type=int,default=0,
                    help="want to use a previously saved number of components? 1 for yes, 0 for no")


    
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
        
    