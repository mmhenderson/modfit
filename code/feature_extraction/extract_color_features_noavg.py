import argparse
import numpy as np
import sys, os
import time

#import custom modules
from utils import color_utils, nsd_utils, prf_utils, default_paths, floc_utils, torch_utils
from model_fitting import initialize_fitting
from feature_extraction import pca_feats

try:
    device = initialize_fitting.init_cuda()
except:
    device = 'cpu:0'

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def extract_color_features(image_data,
                           prf_mask, 
                           batch_size=100, 
                           device=None,
                           debug=False):
    
    n_pix = image_data.shape[2]
    n_images = image_data.shape[0]
    
    n_batches = int(np.ceil(n_images/batch_size))
    
    n_features_color = 4 # [L*, a*, b*, saturation]   
    n_features_spat = np.sum(prf_mask)
    n_features_total = n_features_color * n_features_spat
    
    print('number of [images, features] total: %d, %d'%(n_images,n_features_total))
    
    features = np.zeros((n_images, n_features_total), dtype=np.float32)

    st = time.time()
    
    for bb in range(n_batches):
       
        if debug and bb>1:
            continue
        
        batch_inds = np.arange(bb*batch_size, np.minimum((bb+1)*batch_size, n_images))
        
        print('processing batch %d of %d'%(bb, n_batches))
        
        sys.stdout.flush()

        # color channels will be last dimension of this array
        image_batch = np.moveaxis(image_data[batch_inds,:,:,:], [0,1],[3,2])

        st_cielab = time.time()
        
        image_lab = color_utils.rgb_to_CIELAB(image_batch, device=device)
        
        elapsed_cielab = time.time() - st_cielab
        print('took %.5f sec to get cielab features'%elapsed_cielab)
        
        image_sat = color_utils.get_saturation(image_batch)[:,:,None,:]
              
        # 4 color feature channels concatenated here
        fmaps = np.concatenate([image_lab, image_sat], axis=2)

        # apply the prf mask      
        fmaps_masked = fmaps[prf_mask]
        
        if bb==0:
            
            print('size of prf mask is:')
            print(prf_mask.shape)
            print('sum of prf mask is:')
            print(np.sum(prf_mask))   
            print('size of fmaps_masked is:')
            print(fmaps_masked.shape)
        
        fmaps_masked_reshaped = np.moveaxis(fmaps_masked, [2],[0])
        
        # all the color and spatial channels going into one big dimension.
        features[batch_inds,:] = np.reshape(fmaps_masked_reshaped, [len(batch_inds),-1])

    elapsed = time.time() - st
    print('took %.5f s to gather color feature maps'%elapsed)

            
    return features
      
    
def proc_one_subject(subject, args):

    feat_path = default_paths.color_feat_path
       
    if args.debug:
        feat_path = os.path.join(feat_path,'DEBUG')
    
    path_to_save = os.path.join(feat_path, 'PCA')
        
    weights_folder = os.path.join(path_to_save, 'pca_weights_grid%d'%args.which_prf_grid)
    
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    if not os.path.exists(weights_folder):
        os.makedirs(weights_folder)
     
    # Load and prepare the image set to work with 
    if subject==999:
        # 999 is a code i am using to indicate the independent set of coco images, which were
        # not actually shown to any NSD participants
        from utils import coco_utils
        image_data = coco_utils.load_indep_coco_images(n_pix=args.map_res_pix)
        fit_inds = None
    else: 
        # load all images for the current subject, 10,000 ims
        image_data = nsd_utils.get_image_data(subject,npix=args.map_res_pix) 
        subject_df = nsd_utils.get_subj_df(subject)
        fit_inds = np.array(subject_df['shared1000']==False)

    # Params for the spatial aspect of the model (possible pRFs)
    models = initialize_fitting.get_prf_models(which_grid = args.which_prf_grid)    
    n_prfs = models.shape[0]
      
    n_pix = image_data.shape[2]
     
    n_prf_sd_out = 1.5
   
    for mm in range(n_prfs):

        if mm>1 and args.debug:
            continue
            
        print('proc pRF %d of %d'%(mm, n_prfs))
        
        # create the binary masks for each pRF, +/- 1.5 SD   
        x,y,sigma = models[mm,:] 
        if sigma is None:
            assert(args.which_prf_grid==0)
            prf_mask = np.ones((n_pix, n_pix),dtype=int)
        else:
            prf_mask = prf_utils.get_prf_mask(center=[x,y], sd=sigma, \
                               patch_size=n_pix, zscore_plusminus=n_prf_sd_out)
        prf_mask = prf_mask==1

        assert(np.sum(prf_mask)>0)
        
        # first extract features for all pixels/feature channels 
        features_raw = extract_color_features(image_data, \
                                              prf_mask = prf_mask,
                                              batch_size=args.batch_size,
                                              debug=args.debug, 
                                              device=device)

        # then do pca to make these less huge
        filename_save_pca = os.path.join(path_to_save, \
                                   'S%d_cielab_plus_sat_noavg_PCA_grid%d.h5py'%\
                                    (subject, args.which_prf_grid))
        
        if args.save_weights:
            save_weights_filename = os.path.join(weights_folder, \
                                    'S%d_cielab_plus_sat_noavg_PCA_weights_prf%d.npy'%\
                                    (subject, mm))
        else:
            save_weights_filename = None

        if args.use_saved_ncomp:
            ncomp_filename = os.path.join(weights_folder, \
                                    'S%d_cielab_plus_sat_noavg_PCA_ncomp_prf%d.npy'%\
                                    (subject, mm))
        else:
            ncomp_filename = None


        pca_feats.run_pca_oneprf(features_raw, 
                                    filename_save_pca, 
                                    prf_save_ind=mm, \
                                    n_prfs_total=n_prfs, 
                                    fit_inds=fit_inds, 
                                    min_pct_var=args.min_pct_var, 
                                    max_pc_to_retain=args.max_pc_to_retain,
                                    save_weights=args.save_weights, 
                                    save_weights_filename=save_weights_filename, 
                                    use_saved_ncomp=args.use_saved_ncomp, 
                                    ncomp_filename=ncomp_filename,\
                                    save_dtype=np.float32, compress=True, \
                                    debug=args.debug)



    
def proc_other_image_set(image_set, args):
       
    feat_path = default_paths.color_feat_path
    
    if args.debug:
        feat_path = os.path.join(feat_path,'DEBUG')
        
    path_to_save = os.path.join(feat_path, 'PCA')
      
    weights_folder = os.path.join(path_to_save, 'pca_weights_grid%d'%args.which_prf_grid)
    
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
       
    if image_set=='floc':
        image_data = floc_utils.load_floc_images(npix=args.map_res_pix)
        fit_inds=None
    else:
        raise ValueError('image set %s not recognized'%image_set)
        
    if image_data.shape[1]==1:
        image_data = np.tile(image_data, [1,3,1,1])
 
    print('shape of image_data:')
    print(image_data.shape)
    
    # Params for the spatial aspect of the model (possible pRFs)
    models = initialize_fitting.get_prf_models(which_grid = args.which_prf_grid)    
    n_prfs = models.shape[0]
      
    n_pix = image_data.shape[2]
    
    n_prf_sd_out = 1.5
        
    
    for mm in range(n_prfs):
        
        if mm>1 and args.debug:
            continue

        print('proc pRF %d of %d'%(mm, n_prfs))
        
        # create the binary masks for each pRF, +/- 1.5 SD   
        x,y,sigma = models[mm,:] 
        if sigma is None:
            assert(args.which_prf_grid==0)
            prf_mask = np.ones((n_pix, n_pix),dtype=int)
        else:
            prf_mask = prf_utils.get_prf_mask(center=[x,y], sd=sigma, \
                               patch_size=n_pix, zscore_plusminus=n_prf_sd_out)
        prf_mask = prf_mask==1

        assert(np.sum(prf_mask)>0)
        
        features_raw = extract_color_features(image_data, \
                                              prf_mask = prf_mask,
                                              batch_size=args.batch_size,
                                              debug=args.debug, 
                                              device=device)

        subjects_pca=np.arange(1,9)

        for ss in subjects_pca:

            load_weights_filename = os.path.join(weights_folder, 'S%d_cielab_plus_sat_noavg_PCA_weights_prf%d.npy'%\
                                        (ss, mm))
            
            filename_save_pca = os.path.join(path_to_save, \
                                   '%s_cielab_plus_sat_noavg_PCA_wtsfromS%d_grid%d.h5py'%\
                                    (image_set, ss, args.which_prf_grid))

            pca_feats.apply_pca_oneprf(features_raw, 
                                        filename_save_pca, 
                                        load_weights_filename=load_weights_filename, 
                                        prf_save_ind=mm, \
                                        n_prfs_total=n_prfs, 
                                        max_pc_to_retain = args.max_pc_to_retain, 
                                        save_dtype=np.float32, compress=True, \
                                        debug=args.debug)



    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=0,
                    help="number of the subject, 1-8")
    parser.add_argument("--image_set", type=str,default='none',
                    help="name of the image set to use (if not an NSD subject)")
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which version of prf grid to use")

    
    parser.add_argument("--map_res_pix", type=int,default=100,
                    help="resolution of feature maps before pca")
    parser.add_argument("--batch_size", type=int,default=100,
                    help="batch size for color feature extraction")
    
    parser.add_argument("--max_pc_to_retain", type=int,default=0,
                    help="max pc to retain? enter 0 for None")
    parser.add_argument("--min_pct_var", type=int,default=95,
                    help="min pct var to explain? default 95")
    parser.add_argument("--save_weights", type=int,default=0,
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
        
    