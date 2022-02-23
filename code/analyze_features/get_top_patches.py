from feature_extraction import fwrf_features
from utils import texture_utils, default_paths, nsd_utils
from model_fitting import initialize_fitting

import os, sys
import numpy as np
from PIL import Image

import argparse

def get_top_patches_sketch_tokens(top_n_images=96,which_prf_grid=5,debug=False):

    """
    For each feature in sketch tokens feature space, identify the top n images for each pRF, 
    crop around each pRF and save the patches. 
    """

    subjects = np.arange(1,9)
    path_to_load = default_paths.sketch_token_feat_path
    feat_loaders = [fwrf_features.fwrf_feature_loader(subject=ss,\
                                which_prf_grid=which_prf_grid, \
                                feature_type='sketch_tokens',\
                                use_pca_feats = False) for ss in subjects]
    n_features = feat_loaders[0].max_features
    
    # putting all the patch images into this folder, can delete once we've analyzed them
    folder2save = os.path.join(default_paths.sketch_token_feat_path, 'top_im_patches')
    if not os.path.exists(folder2save):
        os.mkdir(folder2save)

    # how many images to look at per subj? will later combine across all subs.
    top_n_each_subj = int(np.ceil(top_n_images/len(subjects)))
    
    prf_models = initialize_fitting.get_prf_models(which_prf_grid)
    n_prfs = prf_models.shape[0]
    
    for ss, feat_loader in zip(subjects, feat_loaders):

        val_inds = np.array(nsd_utils.get_subj_df(subject=ss)['shared1000'])
        trninds = np.where(val_inds==False)[0]
        
        images = nsd_utils.get_image_data(subject=ss)
        images = images[trninds,:,:,:]
        image_size = images.shape[2:4]
        images = nsd_utils.image_uncolorize_fn(images)

        for mm in range(n_prfs):

            print('proc subj %d, prf %d of %d'%(ss, mm, n_prfs))
            if debug and mm>1:
                continue

            bbox = texture_utils.get_bbox_from_prf(prf_models[mm,:], \
                                       image_size, n_prf_sd_out=2, \
                                       min_pix=None, verbose=False, \
                                       force_square=False)

            feat, _ = feat_loader.load(trninds, prf_model_index=mm)
            assert(feat.shape[0]==images.shape[0])
            
            for ff in range(n_features):

                # sort in descending order, to get top n images
                sorted_order = np.flip(np.argsort(feat[:,ff]))
            
                top_images = images[sorted_order[0:top_n_each_subj],:,:,:]

                # taking just the patch around this pRF, because this is the region that 
                # contributed to computing the sketch tokens feature
                top_images_cropped = top_images[:,:,bbox[0]:bbox[1], bbox[2]:bbox[3]]

                for ii in range(top_n_each_subj):

                    image2save = (top_images_cropped[ii,0,:,:]*255).astype(np.uint8)
                    fn2save = os.path.join(folder2save, 'S%d_prf%d_feature%d_ranked%d.jpg'%\
                                           (ss, mm, ff, ii))
                    print('saving to %s'%fn2save)
                    Image.fromarray(image2save).save(fn2save)
                    
                    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which prf grid to use")
    parser.add_argument("--top_n_images", type=int,default=96,
                    help="how many top images to use?")
   
   
    args = parser.parse_args()
    
    get_top_patches_sketch_tokens(top_n_images=args.top_n_images,\
                                  which_prf_grid=args.which_prf_grid,\
                                  debug=args.debug==1)
    