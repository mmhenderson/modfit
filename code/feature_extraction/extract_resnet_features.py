import numpy as np
import sys, os
import argparse
import gc
import torch
import time
import h5py
import copy
from collections import OrderedDict
import torch.nn as nn
import torchvision.models as models


#import custom modules
from utils import prf_utils, torch_utils, texture_utils, default_paths, nsd_utils, floc_utils
from model_fitting import initialize_fitting
from feature_extraction import pca_feats

# clip implemented in this package, from:
# https://github.com/openai/CLIP
import clip

# Define stuff about the resnet layers here
n_features_each_resnet_block = [256,256,256, 512,512,512,512, 1024,1024,1024,1024,1024,1024, 2048,2048,2048]
resnet_block_names = ['block%d'%nn for nn in range(len(n_features_each_resnet_block))]

try:
    device = initialize_fitting.init_cuda()
except:
    device = 'cpu:0'

def extract_features(image_data, \
                    block_inds, \
                    prf_batch_inds, \
                    save_batch_filenames, \
                    which_prf_grid=5, \
                    save_dtype=np.float32,\
                    training_type='clip', \
                    debug=False):
    """ 
    Extract the portion of CNN feature maps corresponding to each pRF in the grid.
    Save values of the features in each pRF, for each layer of interest.
    """

    prf_models = initialize_fitting.get_prf_models(which_grid=args.which_prf_grid)  
    
    assert(len(block_inds)==1)
    ll = block_inds[0]
    
    n_images = image_data.shape[0]
    
    # Keep these params fixed
    n_prf_sd_out = 2
    batch_size = 100 # batches in image dimension
    mult_patch_by_prf = True
    do_avg_pool = True
    model_architecture='RN50'

    n_prf_batches = len(prf_batch_inds)
    
    n_batches = int(np.ceil(n_images/batch_size))

    with torch.no_grad():

        # separate the prfs into batches to save memory.  
        for pb in range(n_prf_batches):

            prfs_this_batch = prf_batch_inds[pb]
            # doing the whole procedure of feature extraction one layer at a time
            # otherwise will run out of memory bc huge arrays.
            features_each_prf = np.zeros((n_images, n_features_each_resnet_block[ll], \
                                          len(prfs_this_batch)),dtype=save_dtype)

            for bb in range(n_batches):

                if debug and bb>1:
                    continue
                print('Processing images for batch %d of %d'%(bb, n_batches))

                batch_inds = np.arange(batch_size * bb, np.min([batch_size * (bb+1), n_images]))

                # using grayscale images for better comparison w my other models.
                # need to tile to 3 so model weights will be right size
                image_batch = np.tile(image_data[batch_inds,:,:,:], [1,3,1,1])

                gc.collect()
                torch.cuda.empty_cache()

                activ_batch = get_resnet_activations_batch(image_batch, block_inds, \
                                                     model_architecture, training_type, device=device)


                print('Getting prf-specific activations for %s'%resnet_block_names[ll])

                maps_full_field = torch.moveaxis(activ_batch[0], [0,1,2,3], [0,3,1,2])

                if bb==0:
                    print('size of maps stack for first batch is:')
                    print(maps_full_field.shape)

                for mi, mm in enumerate(prfs_this_batch):

                    if debug and mi>1:
                        continue

                    prf_params = prf_models[mm,:]
                    x,y,sigma = prf_params
                    print('Getting features for pRF [x,y,sigma]:')
                    print([x,y,sigma])
                    n_pix = maps_full_field.shape[1]

                    # Define the RF for this "model" version
                    prf = torch_utils._to_torch(prf_utils.gauss_2d(center=[x,y], sd=sigma, \
                               patch_size=n_pix, aperture=1.0, dtype=np.float32), device=device)
                    minval = torch.min(prf)
                    maxval = torch.max(prf-minval)
                    prf_scaled = (prf - minval)/maxval

                    if mult_patch_by_prf:
                        # This effectively restricts the spatial location, so no need to crop
                        maps = maps_full_field * prf_scaled.view([1,n_pix, n_pix,1])
                    else:
                        # This is a coarser way of choosing which spatial region to look at
                        # Crop the patch +/- n SD away from center
                        bbox = texture_utils.get_bbox_from_prf(prf_params, prf.shape, n_prf_sd_out, \
                                                       min_pix=None, verbose=False, force_square=False)
                        print('bbox to crop is:')
                        print(bbox)
                        maps = maps_full_field[:,bbox[0]:bbox[1], bbox[2]:bbox[3],:]

                    if do_avg_pool:
                        features_batch = torch.mean(maps, dim=(1,2))
                    else:
                        features_batch = torch.max(maps, dim=(1,2))

                    print('model %d, min/max of features in batch: [%s, %s]'%(mm, \
                                                  torch.min(features_batch), torch.max(features_batch))) 

                    features_each_prf[batch_inds,:,mi] = torch_utils.get_value(features_batch)

            # Now save the results
            save_features(features_each_prf, save_batch_filenames[pb], save_dtype)
          
        
def get_resnet_activations_batch(image_batch, \
                               block_inds, \
                               model_architecture, \
                               training_type, \
                               device=None):

    """
    Get activations for images in NSD, passed through pretrained resnet model.
    Specify which NSD images to look at, and which layers to return.
    """

    if device is None:
        device = torch.device('cpu:0')
       
    if training_type=='clip':        
        
        print('Using CLIP model')
        model, preprocess = clip.load(model_architecture, device=device)
        model = model.visual
        
    elif training_type=='blurface':
        
        model = models.resnet50().float().to(device)
        
        # use model trained on face-blurred imagenet ims
        model_filename = os.path.join(default_paths.resnet50_blurface_feat_path, 'resnet50_blurred_ILSVRC.pth')
        print('Loading saved model from %s'%model_filename)
        saved = torch.load(model_filename, map_location=device)     
        sd = saved['state_dict']
        # need to fix keys in the statedict to have expected names
        new_sd = OrderedDict([])
        keynames = list(sd.keys())
        for kn in keynames:
            new_kn = kn.split('module.')[1]
            new_sd[new_kn] = sd[kn]
        model.load_state_dict(new_sd)
        
    elif training_type=='imgnet':
        
        # normal pre-trained model from pytorch
        print('Using pretrained Resnet50 model')
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).float().to(device)
        
    else:
        raise ValueError('training type %s not recognized'%training_type)
        
    model.eval()
    
    # The 16 residual blocks are segmented into 4 groups here, which have different numbers of features.
    blocks_each= [len(model.layer1), len(model.layer2), len(model.layer3),len(model.layer4)]
    which_group = np.repeat(np.arange(4), blocks_each)

    activ = [[] for ll in block_inds]
    hooks = [[] for ll in block_inds]
    
    # first making this subfunction that is needed to get the activation on a forward pass
    def get_activ_fwd_hook(ii,ll):
        def hook(self, input, output):
            # the relu operation is used multiple times per block, but we only 
            # want to save its output when it has this specific size.
            if output.shape[1]==n_features_each_resnet_block[ll]:
                print('executing hook for %s'%resnet_block_names[ll])  
                activ[ii] = output
                print(output.shape)
        return hook

    image_tensors = torch.Tensor(image_batch).to(device)
    with torch.no_grad():

        # adding a "hook" to the module corresponding to each layer, so we'll save activations at each layer.
        # For resnet, going to save output of each residual block following last relu operation.
        for ii, ll in enumerate(block_inds):
            if which_group[ll]==0:            
                h = model.layer1[ll].relu.register_forward_hook(get_activ_fwd_hook(ii,ll))
            elif which_group[ll]==1:            
                h = model.layer2[ll-blocks_each[0]].relu.register_forward_hook(get_activ_fwd_hook(ii,ll))
            elif which_group[ll]==2:            
                h = model.layer3[ll-sum(blocks_each[0:2])].relu.register_forward_hook(get_activ_fwd_hook(ii,ll))
            elif which_group[ll]==3:            
                h = model.layer4[ll-sum(blocks_each[0:3])].relu.register_forward_hook(get_activ_fwd_hook(ii,ll))
            else:
                h=None
            hooks[ii] = h

        # Pass images though the model (hooks get run now)
        image_features = model(image_tensors)

        # Now remove all the hooks
        for ii, ll in enumerate(block_inds):
            print(activ[ii].shape)
            hooks[ii].remove

    # Sanity check that we grabbed the right activations - check their sizes against expected
    # output size of each block
    exp_size = np.array(n_features_each_resnet_block)[block_inds]
    actual_size = [activ[bb].shape[1] for bb in range(len(activ))]
    assert(np.all(np.array(actual_size)==np.array(exp_size)))

    return activ


def proc_one_subject(subject, args):
    
    if args.training_type=='clip':
        feat_path = default_paths.clip_feat_path
    elif args.training_type=='blurface':
        feat_path = default_paths.resnet50_blurface_feat_path
    elif args.training_type=='imgnet':
        feat_path = default_paths.resnet50_feat_path
        
    if args.debug: 
        feat_path = os.path.join(feat_path, 'DEBUG') 
    if not os.path.exists(feat_path):
        os.makedirs(feat_path)
        
    # Load and prepare the image set to work with 
    if subject==999:
        # 999 is a code i am using to indicate the independent set of coco images, which were
        # not actually shown to any NSD participants
        from utils import coco_utils
        image_data = coco_utils.load_indep_coco_images(n_pix=240)
        image_data = nsd_utils.image_uncolorize_fn(image_data)
    else: 
        # load all images for the current subject, 10,000 ims
        image_data = nsd_utils.get_image_data(subject)  
        image_data = nsd_utils.image_uncolorize_fn(image_data)

   
    if args.n_layers_save==16:
        blocks_to_do = np.arange(16)
    elif args.n_layers_save==8:
        blocks_to_do = np.arange(0,16,2)+1
    elif args.n_layers_save==4:
        blocks_to_do = [2,6,12,15]
        
    if args.start_layer>0:
        blocks_to_do = blocks_to_do[args.start_layer:]
    
    prf_models = initialize_fitting.get_prf_models(which_grid=args.which_prf_grid)    
    n_prfs = len(prf_models)
    
    prf_batch_size = 100; # just keeping this par fixed, seems to work ok
    n_prf_batches = int(np.ceil(n_prfs/prf_batch_size))
    
    prf_batch_inds = [np.arange(pb*prf_batch_size, np.min([(pb+1)*prf_batch_size, n_prfs])) \
                      for pb in range(n_prf_batches)]
    
    for ll in blocks_to_do:

        # each batch will be in a separate file, since they're big features
        model_architecture='RN50'
        save_batch_filenames = [os.path.join(feat_path, \
               'S%d_%s_%s_features_each_prf_grid%d_prfbatch%d.h5py'%\
                (subject, model_architecture, resnet_block_names[ll], args.which_prf_grid, pb)) \
                                for pb in range(n_prf_batches)]
        
        block_inds = [ll]
       
        extract_features(image_data,\
                          block_inds,\
                          prf_batch_inds,\
                          save_batch_filenames,\
                          which_prf_grid=args.which_prf_grid, \
                          save_dtype=np.float32,\
                          training_type=args.training_type, \
                          debug=args.debug)

        sys.stdout.flush()
            
        layer = 'block%d'%(ll)
        pca_feats.run_pca(subject=subject, \
                          feature_type='resnet_%s'%args.training_type, \
                          layer_name = layer, \
                          which_prf_grid=args.which_prf_grid,\
                          min_pct_var=args.min_pct_var,\
                          max_pc_to_retain=args.max_pc_to_retain, \
                          save_weights = args.save_pca_weights==1, \
                          use_saved_ncomp = args.use_saved_ncomp==1, \
                          debug=args.debug)
        
        # now removing the large intermediate files, leaving only the pca versions
        for big_fn in save_batch_filenames:
            
            print('removing big activations file: %s'%big_fn)
            sys.stdout.flush()
            os.remove(big_fn)

            print('big file removed.')

            
def proc_other_image_set(image_set, args):
    
    if args.training_type=='clip':
        feat_path = default_paths.clip_feat_path
    elif args.training_type=='blurface':
        feat_path = default_paths.resnet50_blurface_feat_path
    elif args.training_type=='imgnet':
        feat_path = default_paths.resnet50_feat_path
       
    
    if args.debug: 
        feat_path = os.path.join(feat_path, 'DEBUG') 
    if not os.path.exists(feat_path):
        os.makedirs(feat_path)
         
    if image_set=='floc':
        image_data = floc_utils.load_floc_images(npix=240)
    else:
        raise ValueError('image set %s not recognized'%image_set)
       
    if args.n_layers_save==16:
        blocks_to_do = np.arange(16)
    elif args.n_layers_save==8:
        blocks_to_do = np.arange(0,16,2)+1
    elif args.n_layers_save==4:
        blocks_to_do = [2,6,12,15]
        
    if args.start_layer>0:
        blocks_to_do = blocks_to_do[args.start_layer:]
    
    prf_models = initialize_fitting.get_prf_models(which_grid=args.which_prf_grid)    
    n_prfs = len(prf_models)
    
    prf_batch_size = 100; # just keeping this par fixed, seems to work ok
    n_prf_batches = int(np.ceil(n_prfs/prf_batch_size))
    
    prf_batch_inds = [np.arange(pb*prf_batch_size, np.min([(pb+1)*prf_batch_size, n_prfs])) \
                      for pb in range(n_prf_batches)]
    
    subjects_pca = np.arange(1,9)
       
    for ll in blocks_to_do:

        # each batch will be in a separate file, since they're big features
        model_architecture='RN50'
        save_batch_filenames = [os.path.join(feat_path, \
               '%s_%s_%s_features_each_prf_grid%d_prfbatch%d.h5py'%\
                (image_set, model_architecture, resnet_block_names[ll], args.which_prf_grid, pb)) \
                                for pb in range(n_prf_batches)]
        
        block_inds = [ll]
        
        extract_features(image_data,\
                          block_inds,\
                          prf_batch_inds,\
                          save_batch_filenames,\
                          which_prf_grid=args.which_prf_grid,\
                          save_dtype=np.float32,\
                          training_type=args.training_type, \
                          debug=args.debug)

        sys.stdout.flush()
            
        layer = 'block%d'%(ll)
        for ss in subjects_pca:
            
            pca_feats.run_pca(subject=ss, \
                              image_set=image_set, \
                              feature_type='resnet_%s'%args.training_type, \
                              layer_name = layer, \
                              which_prf_grid=args.which_prf_grid,\
                              min_pct_var=args.min_pct_var,\
                              max_pc_to_retain=args.max_pc_to_retain, \
                              save_weights = False,
                              use_saved_ncomp = args.use_saved_ncomp==1, \
                              debug=args.debug)

        # now removing the large intermediate files, leaving only the pca versions
        for big_fn in save_batch_filenames:
            
            print('removing big activations file: %s'%big_fn)
            sys.stdout.flush()
            os.remove(big_fn)

            print('big file removed.')
  
            
def save_features(features_each_prf, filename_save, save_dtype):
    
    print('Writing prf features to %s\n'%filename_save)
    
    t = time.time()
    with h5py.File(filename_save, 'w') as data_set:
        dset = data_set.create_dataset("features", np.shape(features_each_prf), dtype=save_dtype)
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
    parser.add_argument("--training_type", type=str,default='clip',
                    help="what kind of model training was used?")
    
    parser.add_argument("--start_layer", type=int,default=0,
                    help="which network layer to start from?")
    parser.add_argument("--n_layers_save", type=int,default=16,
                    help="how many layers to save?")
    
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
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
        

        
        