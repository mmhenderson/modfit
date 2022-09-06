import argparse
import numpy as np
import sys, os
import time
import torch
import gc

#import custom modules
from utils import nsd_utils, prf_utils, default_paths, floc_utils, torch_utils
from model_fitting import initialize_fitting
from feature_extraction import pca_feats
from feature_extraction import gabor_feature_extractor

try:
    device = initialize_fitting.init_cuda()
except:
    device = 'cpu:0'

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def extract_gabor_features(image_data,
                           _gabor_ext_complex,
                           pooling_op_list, 
                           prf_mask_each_res, 
                           batch_size=100, 
                           device=None,
                           debug=False):
    
    n_pix = image_data.shape[2]
    n_images = image_data.shape[0]
    
    n_batches = int(np.ceil(n_images/batch_size))
    
    n_ori = _gabor_ext_complex.n_ori
    n_sf = _gabor_ext_complex.n_sf
    n_features_spat_each_sf = [np.sum(prf_mask) for prf_mask in prf_mask_each_res]   
    n_features_total = np.sum([n_features_spat_each_sf[sf]*n_ori for sf in range(n_sf)])
    
    print('number of features total: %d'%n_features_total)
    
    features = np.zeros((n_images, n_features_total), dtype=np.float32)

    st = time.time()
    
    for bb in range(n_batches):
       
        if debug and bb>1:
            continue
            
        batch_inds = np.arange(bb*batch_size, np.minimum((bb+1)*batch_size, n_images))      
        print('processing batch %d of %d'%(bb, n_batches))
        sys.stdout.flush()

        fmaps = _gabor_ext_complex(torch_utils._to_torch(image_data[batch_inds,:,:,:], device))
        
        # decrease the size of maps with maxpooling
        fmaps_pooled = [pooling_op(fm) for [pooling_op, fm] in zip(pooling_op_list, fmaps)]
        
        if bb==0:
            print('size of fmaps this batch')
            for fm in fmaps:
                print(fm.shape)
            print('size of fmaps_pooled')
            for fm in fmaps_pooled:
                print(fm.shape)
             
        # apply prf mask to gather desired portion of map
        fmaps_masked = [fmap[:,:,mask] for [fmap, mask] in zip(fmaps_pooled, prf_mask_each_res)]
   
        fmaps = None; fmaps_pooled=None;
        gc.collect()
        torch.cuda.empty_cache() 
        
        fmaps_concat = torch.cat([torch.reshape(fmap, [len(batch_inds),-1]) for fmap in fmaps_masked], axis=1)
      
        if bb==0:
            print('size of prf mask is:')
            print([prf_mask.shape for prf_mask in prf_mask_each_res])
            print('sum of prf mask is:')
            print([np.sum(prf_mask) for prf_mask in prf_mask_each_res])
            print('size of masked fmaps')
            for fm in fmaps_masked:
                print(fm.shape)
            print('size of fmaps_concat:')
            print(fmaps_concat.shape)
      
        # all the feature and spatial channels going into one big dimension.
        features[batch_inds,:] = fmaps_concat.detach().cpu().numpy()

        fmaps_masked=None; fmaps_concat=None;
        gc.collect()
        torch.cuda.empty_cache() 
        
    elapsed = time.time() - st
    print('took %.5f s to gather gabor feature maps'%elapsed)
    sys.stdout.flush()
    
    return features
      
    
def proc_one_subject(subject, args):

    feat_path = default_paths.gabor_texture_feat_path
       
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
        image_data = coco_utils.load_indep_coco_images(n_pix=240)
        image_data = nsd_utils.image_uncolorize_fn(image_data)
        fit_inds = None
    else: 
        # load all images for the current subject, 10,000 ims
        image_data = nsd_utils.get_image_data(subject,npix=240) 
        image_data = nsd_utils.image_uncolorize_fn(image_data)
        subject_df = nsd_utils.get_subj_df(subject)
        fit_inds = np.array(subject_df['shared1000']==False)

    n_pix = image_data.shape[2]
    nonlin = lambda x: torch.log(1+torch.sqrt(x))
    
    _gabor_ext_complex = gabor_feature_extractor.gabor_extractor_multi_scale(n_ori=args.n_ori, n_sf=args.n_sf, \
                             sf_range_cyc_per_stim = (3, 72), log_spacing = True, \
                             pix_per_cycle=4.13, cycles_per_radius=0.7, radii_per_filter=4, \
                             complex_cell=True, padding_mode = 'circular', nonlin_fn=nonlin, \
                             RGB=False, device = device)

    
    _gabor_ext_complex.get_fmaps_sizes([n_pix, n_pix])   
    fmaps_res = _gabor_ext_complex.resolutions_each_sf
    
    # to keep the features from getting huge, going to apply maxpooling to some of the larger maps 
    # (in this case, the higher spatial frequencies).
    # define some pooling operations here
    if args.n_sf==8:
        kernel_sizes = [1,1,1,1,2,2,3,4]
    else:
        kernel_sizes = [1 if sf<n_sf/2 else 2 for sf in range(n_sf)] 
    pooling_op_list = [torch.nn.MaxPool2d(kernel_size=ks, stride=ks, padding=0) for ks in kernel_sizes]
    
    # determine what the size of each map will be, after pooling
    pooled_res = []
    for nn, res in enumerate(fmaps_res):
        template = torch.Tensor(np.ones((1,1,res,res)))
        pooled_template = pooling_op_list[nn](template)    
        pooled_res.append(pooled_template.shape[2])

    print('including spat frequencies:')
    print(np.unique(_gabor_ext_complex.feature_table['SF: cycles per stim']))
    print('original map resolutions:')
    print(fmaps_res)
    print('pooled map resolutions:')
    print(pooled_res)
    
    # Params for the spatial aspect of the model (possible pRFs)
    models = initialize_fitting.get_prf_models(which_grid = args.which_prf_grid)    
    n_prfs = models.shape[0]
      
    # masks will be +/- 1.5 SD 
    n_prf_sd_out = 1.5
       
    for mm in range(n_prfs):

        if mm>1 and args.debug:
            continue
            
        print('proc pRF %d of %d'%(mm, n_prfs))
        
        # create the binary masks for each pRF
        prf_mask_each_res = []
        
        x,y,sigma = models[mm,:] 
        # make the pRF at same resolution as each pooled feature map 
        for nn, res in enumerate(pooled_res):
            if sigma is None:
                assert(args.which_prf_grid==0)
                prf_mask = np.ones((res,res),dtype=int)
            else:
                prf_mask = prf_utils.get_prf_mask(center=[x,y], sd=sigma, \
                                   patch_size=res, zscore_plusminus=n_prf_sd_out)
            assert(np.sum(prf_mask)>0)
            prf_mask_each_res.append(prf_mask==1)
    
        # first extract features for all pixels/feature channels 
        features_raw = extract_gabor_features(image_data, \
                                              _gabor_ext_complex, 
                                              pooling_op_list, 
                                              prf_mask_each_res = prf_mask_each_res,
                                              batch_size=args.batch_size,
                                              debug=args.debug, 
                                              device=device)

        
        print('done getting raw features')
        sys.stdout.flush()
        
        # then do pca to make these less huge
        filename_save_pca = os.path.join(path_to_save, \
                                   'S%d_gabor_noavg_%dori_%dsf_PCA_grid%d.h5py'%\
                                    (subject, args.n_ori, args.n_sf, args.which_prf_grid))
        
        if args.save_weights:
            save_weights_filename = os.path.join(weights_folder, \
                                    'S%d_gabor_noavg_%dori_%dsf_PCA_weights_prf%d.npy'%\
                                    (subject, args.n_ori, args.n_sf, mm))
        else:
            save_weights_filename = None

        if args.use_saved_ncomp:
            ncomp_filename = os.path.join(weights_folder, \
                                    'S%d_gabor_noavg_%dori_%dsf_PCA_ncomp_prf%d.npy'%\
                                    (subject, args.n_ori, args.n_sf, mm))
        else:
            ncomp_filename = None

            
        print('abt to run pca')
        sys.stdout.flush()
        
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

        print('done with pca')
        sys.stdout.flush()
        

    
def proc_other_image_set(image_set, args):
       
    feat_path = default_paths.gabor_texture_feat_path
    
    if args.debug:
        feat_path = os.path.join(feat_path,'DEBUG')
        
    path_to_save = os.path.join(feat_path, 'PCA')
      
    weights_folder = os.path.join(path_to_save, 'pca_weights_grid%d'%args.which_prf_grid)
    
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
       
    if image_set=='floc':
        image_data = floc_utils.load_floc_images(npix=240)
        fit_inds=None
    else:
        raise ValueError('image set %s not recognized'%image_set)
        
    n_pix = image_data.shape[2]
    nonlin = lambda x: torch.log(1+torch.sqrt(x))
    
    _gabor_ext_complex = gabor_feature_extractor.gabor_extractor_multi_scale(n_ori=args.n_ori, n_sf=args.n_sf, \
                             sf_range_cyc_per_stim = (3, 72), log_spacing = True, \
                             pix_per_cycle=4.13, cycles_per_radius=0.7, radii_per_filter=4, \
                             complex_cell=True, padding_mode = 'circular', nonlin_fn=nonlin, \
                             RGB=False, device = device)

    
    _gabor_ext_complex.get_fmaps_sizes([n_pix, n_pix])   
    fmaps_res = _gabor_ext_complex.resolutions_each_sf
    
    # to keep the features from getting huge, going to apply maxpooling to some of the larger maps 
    # (in this case, the higher spatial frequencies).
    # define some pooling operations here
    if args.n_sf==8:
        kernel_sizes = [1,1,1,1,2,2,3,4]
    else:
        kernel_sizes = [1 if sf<n_sf/2 else 2 for sf in range(n_sf)] 
    pooling_op_list = [torch.nn.MaxPool2d(kernel_size=ks, stride=ks, padding=0) for ks in kernel_sizes]
    
    # determine what the size of each map will be, after pooling
    pooled_res = []
    for nn, res in enumerate(fmaps_res):
        template = torch.Tensor(np.ones((1,1,res,res)))
        pooled_template = pooling_op_list[nn](template)    
        pooled_res.append(pooled_template.shape[2])

    print('including spat frequencies:')
    print(np.unique(_gabor_ext_complex.feature_table['SF: cycles per stim']))
    print('original map resolutions:')
    print(fmaps_res)
    print('pooled map resolutions:')
    print(pooled_res)
    
    # Params for the spatial aspect of the model (possible pRFs)
    models = initialize_fitting.get_prf_models(which_grid = args.which_prf_grid)    
    n_prfs = models.shape[0]
      
    # masks will be +/- 1.5 SD 
    n_prf_sd_out = 1.5
     
    for mm in range(n_prfs):
        
        if mm>1 and args.debug:
            continue

        print('proc pRF %d of %d'%(mm, n_prfs))
        
        # create the binary masks for each pRF
        prf_mask_each_res = []
        
        x,y,sigma = models[mm,:] 
        # make the pRF at same resolution as each pooled feature map 
        for nn, res in enumerate(pooled_res):
            if sigma is None:
                assert(args.which_prf_grid==0)
                prf_mask = np.ones((res,res),dtype=int)
            else:
                prf_mask = prf_utils.get_prf_mask(center=[x,y], sd=sigma, \
                                   patch_size=res, zscore_plusminus=n_prf_sd_out)
            assert(np.sum(prf_mask)>0)
            prf_mask_each_res.append(prf_mask==1)
    
        # first extract features for all pixels/feature channels 
        features_raw = extract_gabor_features(image_data, \
                                              _gabor_ext_complex, 
                                              pooling_op_list, 
                                              prf_mask_each_res = prf_mask_each_res,
                                              batch_size=args.batch_size,
                                              debug=args.debug, 
                                              device=device)

        
        print('done getting raw features')
        sys.stdout.flush()
        
        subjects_pca=np.arange(1,9)

        for ss in subjects_pca:

            load_weights_filename = os.path.join(weights_folder, \
                                     'S%d_gabor_noavg_%dori_%dsf_PCA_weights_prf%d.npy'%\
                                    (ss, args.n_ori, args.n_sf, mm))
            filename_save_pca = os.path.join(path_to_save, \
                                     '%s_gabor_noavg_%dori_%dsf_PCA_wtsfromS%d_grid%d.h5py'%\
                                    (image_set, args.n_ori, args.n_sf, ss, args.which_prf_grid))
                                   
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

    
    parser.add_argument("--n_ori", type=int,default=12,
                    help="num orientations in gabor bank")
    parser.add_argument("--n_sf", type=int,default=8,
                    help="num spat freq in gabor bank")
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
        
    