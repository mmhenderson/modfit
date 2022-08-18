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
                           debug=False):
    
    n_pix = image_data.shape[2]
    n_images = image_data.shape[0]
    
    n_features_color = 4 # [L*, a*, b*, saturation]   
    n_features_spat = n_pix*n_pix
    n_features_total = n_features_color * n_features_spat
    
    print('number of features total: %d'%n_features_total)
    
    features = np.zeros((n_images, n_features_total), dtype=np.float32)

    st = time.time()
    
    for ii in range(n_images):
        
        if debug and ii>1:
            continue
        
        if np.mod(ii,500)==0:
            print('processing image %d of %d'%(ii, n_images))
            
        sys.stdout.flush()
        
        # color channels will be last dimension of this array
        image = np.moveaxis(image_data[ii,:,:,:], [0],[2])

        image_lab = color_utils.rgb_to_CIELAB(image)
        image_sat = color_utils.get_saturation(image)

        # 4 color feature channels concatenated here
        fmaps = np.dstack([image_lab, image_sat])

        features[ii,:] = fmaps.ravel()

    elapsed = time.time() - st
    print('took %.5f s to gather color feature maps'%elapsed)

            
    return features
      
    
def proc_one_subject(subject, args):

    feat_path = default_paths.color_feat_path
       
    if args.debug:
        feat_path = os.path.join(feat_path,'DEBUG')
    
    path_to_save = os.path.join(feat_path, 'PCA')
        
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
     
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

    # first extract features for all pixels/feature channels 
    features_raw = extract_color_features(image_data, \
                                          debug=args.debug)
    
    # then do pca to make these less huge
    
    filename_save_pca = os.path.join(path_to_save, \
                               'S%d_cielab_plus_sat_res%dpix_PCA_grid0.h5py'%\
                                (subject, args.map_res_pix))
    if args.save_weights:
        save_weights_filename = os.path.join(path_to_save, \
                                'S%d_cielab_plus_sat_res%dpix_PCA_weights_grid0.npy'%\
                                (subject, args.map_res_pix))
    else:
        save_weights_filename = None
            
    if args.use_saved_ncomp:
        ncomp_filename = os.path.join(path_to_save, \
                                'S%d_cielab_plus_sat_res%dpix_PCA_ncomp_grid0.npy'%\
                                (subject, args.map_res_pix))
    else:
        ncomp_filename = None
     
    
    pca_feats.run_pca_fullfield(features_raw, 
                                filename_save_pca, 
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
        
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
       
    if image_set=='floc':
        image_data = floc_utils.load_floc_images(npix=args.map_res_pix)
        fit_inds=None
    else:
        raise ValueError('image set %s not recognized'%image_set)
        
    if image_data.shape[1]==1:
        image_data = np.tile(image_data, [1,3,1,1])
 

    features_raw = extract_color_features(image_data, \
                                          debug=args.debug)
    
    subjects_pca=np.arange(1,9)
    
    for ss in subjects_pca:
        
        load_weights_filename = os.path.join(path_to_save, 'S%d_cielab_plus_sat_res%dpix_PCA_weights_grid0.npy'%\
                                    (ss, args.map_res_pix))
        filename_save_pca = os.path.join(path_to_save, \
                               '%s_cielab_plus_sat_res%dpix_PCA_wtsfromS%d_grid0.h5py'%\
                                (image_set, args.map_res_pix, ss))
    
        pca_feats.apply_pca_fullfield(features_raw, 
                                    filename_save_pca, 
                                    load_weights_filename=load_weights_filename, 
                                    save_dtype=np.float32, compress=True, \
                                    debug=args.debug)
    


    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=0,
                    help="number of the subject, 1-8")
    parser.add_argument("--image_set", type=str,default='none',
                    help="name of the image set to use (if not an NSD subject)")

    parser.add_argument("--map_res_pix", type=int,default=100,
                    help="resolution of feature maps before pca")
    
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
        
    