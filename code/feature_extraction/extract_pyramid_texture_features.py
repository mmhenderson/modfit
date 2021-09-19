import argparse
import numpy as np
import sys, os
import torch
import time
import h5py
import torch.nn

#import custom modules
code_dir = '/user_data/mmhender/imStat/code/'
sys.path.append(code_dir)
from utils import torch_utils, nsd_utils, default_paths
from model_fitting import initialize_fitting
from feature_extraction import texture_statistics_pyramid

device = initialize_fitting.init_cuda()

def extract_features(subject, n_ori=4, n_sf=4, batch_size=100, use_node_storage=False, debug=False):
    
    if use_node_storage:
        pyramid_texture_feat_path = default_paths.pyramid_texture_feat_path_localnode
    else:
        pyramid_texture_feat_path = default_paths.pyramid_texture_feat_path
               
    # Load and prepare the image set to work with (all images for the current subject, 10,000 ims)
    stim_root = default_paths.stim_root
    image_data = nsd_utils.get_image_data(subject)  
    image_data = nsd_utils.image_uncolorize_fn(image_data)

    # Need to have size a multiple of 8, for the pyramid to work right
    process_at_size=240
    resample_fn = torch.nn.Upsample((process_at_size, process_at_size), mode="bilinear", align_corners=True)

    # Params for the spatial aspect of the model (possible pRFs)
    aperture = 1.0
    aperture_rf_range = 1.1
    aperture, models = initialize_fitting.get_prf_models(aperture_rf_range=aperture_rf_range)    

    # Set up the pyramid
    feature_types_exclude = []
    n_prf_sd_out = 2
    do_varpart=False # this doesn't do anything here
    group_all_hl_feats = False # this doesn't do anything here

    compute_features = True
    _fmaps_fn = texture_statistics_pyramid.steerable_pyramid_extractor(pyr_height = n_sf, n_ori = n_ori)
    _feature_extractor = texture_statistics_pyramid.texture_feature_extractor(_fmaps_fn,sample_batch_size=batch_size, \
                                                                             feature_types_exclude=feature_types_exclude, \
                                                   n_prf_sd_out=n_prf_sd_out, aperture=aperture, do_varpart = do_varpart, \
                                  compute_features=compute_features, group_all_hl_feats = group_all_hl_feats, device=device)

    n_features = _feature_extractor.n_features_total
    n_images = image_data.shape[0]
    n_prfs = models.shape[0]
    n_batches = int(np.ceil(n_images/batch_size))

    features_each_prf = np.zeros((n_images, n_features, n_prfs), dtype=np.float32)

    for bb in range(n_batches):

        if debug and bb>1:
            continue

        batch_inds = np.arange(batch_size * bb, np.min([batch_size * (bb+1), n_images]))

        print('Extracting features for images [%d - %d]'%(batch_inds[0], batch_inds[-1]))

        image_batch = torch_utils._to_torch(image_data[batch_inds,:,:,:], device)

        image_batch = resample_fn(image_batch)

        _feature_extractor.clear_big_features()

        for mm in range(n_prfs):

            if debug and mm>1:
                continue

            x,y,sigma = models[mm,:]
            print('Getting features for pRF [x,y,sigma]:')
            print([x,y,sigma])

            features_batch, _ = _feature_extractor(image_batch, models[mm],mm)

            print('model %d, min/max of features in batch: [%s, %s]'%(mm, torch.min(features_batch), torch.max(features_batch))) 

            features_each_prf[batch_inds,:,mm] = torch_utils.get_value(features_batch)

            sys.stdout.flush()
                
    fn2save = os.path.join(pyramid_texture_feat_path, 'S%d_features_each_prf_%dori_%dsf.h5py'%(subject, n_ori, n_sf))

    print('Writing prf features to %s\n'%fn2save)
    
    t = time.time()
    with h5py.File(fn2save, 'w') as data_set:
        dset = data_set.create_dataset("features", np.shape(features_each_prf), dtype=np.float64)
        data_set['/features'][:,:,:] = features_each_prf
        data_set.close()  
    elapsed = time.time() - t
    
    print('Took %.5f sec to write file'%elapsed)
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=1,
                    help="number of the subject, 1-8")
    parser.add_argument("--n_ori", type=int,default=4,
                    help="how many orientation channels?")
    parser.add_argument("--n_sf", type=int,default=4,
                    help="how many frequency channels (pyramid levels)?")
    parser.add_argument("--batch_size", type=int,default=100,
                    help="batch size to use for feature extraction")
    
    parser.add_argument("--use_node_storage", type=int,default=0,
                    help="want to save and load from scratch dir on current node? 1 for yes, 0 for no")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    
    args = parser.parse_args()
    
    extract_features(subject = args.subject, n_ori = args.n_ori, n_sf = args.n_sf, batch_size = args.batch_size, use_node_storage = args.use_node_storage, debug = args.debug==1)
