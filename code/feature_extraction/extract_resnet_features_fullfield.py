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
                    pooling_op = None,
                    save_dtype=np.float32,\
                    training_type='clip', \
                    debug=False):
    """ 
    Extract the portion of CNN feature maps corresponding to each pRF in the grid.
    Save values of the features in each pRF, for each layer of interest.
    """

    assert(len(block_inds)==1)
    ll = block_inds[0]
    
    n_images = image_data.shape[0]
    
    # Keep these params fixed
    batch_size = 100 # batches in image dimension
    model_architecture='RN50'
    
    if 'startingblurry' in training_type:   
        do_tile=False
        if 'ecoset_noblur_bw' in training_type:
            trial_num = training_type.split('ecoset_noblur_bw_')[1]
            model_filename = os.path.join(default_paths.startingblurry_root,
                                          'BW50/%s/BWNoBlurModel.pt'%trial_num)
        elif 'ecoset_linblur_bw' in training_type:
            trial_num = training_type.split('ecoset_linblur_bw_')[1]
            model_filename = os.path.join(default_paths.startingblurry_root,
                                          'BW50/%s/50BWLinearBlurModel.pt'%trial_num)
        elif 'ecoset_nonlinblur_bw' in training_type:
            trial_num = training_type.split('ecoset_nonlinblur_bw_')[1]
            model_filename = os.path.join(default_paths.startingblurry_root,
                                          'BW50/%s/50BWNonLinearBlurModel2.pt'%trial_num)
        else:        
            raise ValueError('training type %s not recognized'%(training_type))
    else:
        do_tile=True
        model_filename=None
   
    n_batches = int(np.ceil(n_images/batch_size))

    # figure out how big features will be, by passing a test image through
    if do_tile:
        image_template = np.tile(image_data[0:1,:,:,:], [1,3,1,1])
    else:
        image_template = image_data[0:1,:,:,:]
    activ_template = get_resnet_activations_batch(image_template, block_inds, \
                                                 model_architecture, training_type, \
                                                  model_filename=model_filename, device=device)
    if pooling_op is not None:
        activ_template = [pooling_op(activ_template[0])]
    n_features_total = np.prod(activ_template[0].shape[1:])  
    print('number of features total: %d'%n_features_total)
    
    features = np.zeros((n_images, n_features_total),dtype=save_dtype)

    with torch.no_grad():

        
        for bb in range(n_batches):

            if debug and bb>1:
                continue
            print('Processing images for batch %d of %d'%(bb, n_batches))
            sys.stdout.flush()
            
            batch_inds = np.arange(batch_size * bb, np.min([batch_size * (bb+1), n_images]))

            # using grayscale images for better comparison w my other models.
            # need to tile to 3 so model weights will be right size
            if do_tile:
                image_batch = np.tile(image_data[batch_inds,:,:,:], [1,3,1,1])
            else:
                image_batch = image_data[batch_inds,:,:,:]

            gc.collect()
            torch.cuda.empty_cache()

            activ_batch = get_resnet_activations_batch(image_batch, block_inds, \
                                                 model_architecture, training_type, \
                                                 model_filename=model_filename, device=device)
            if bb==0:
                print('size of activ this batch raw:')
                print(activ_batch[0].shape)
              
            if pooling_op is not None:
                
                activ_batch = [pooling_op(activ_batch[0])]
                
                if bb==0:
                    print('size of activ pooled:')
                    print(activ_batch[0].shape)
              
            activ_batch_reshaped = torch.reshape(activ_batch[0], [len(batch_inds), -1])
           
            if bb==0:
                print('size of activ reshaped:')
                print(activ_batch_reshaped.shape)
                
            features[batch_inds,:] = activ_batch_reshaped.detach().cpu().numpy()
        
    return features

        
def get_resnet_activations_batch(image_batch, \
                               block_inds, \
                               model_architecture, \
                               training_type, \
                               model_filename = None, \
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
        
    elif 'startingblurry' in training_type:
        
        model = models.resnet50().float().to(device)
        print('Loading saved model from %s'%model_filename)
        loaded_state_dict = torch.load(model_filename, map_location=device)

        # fix the key names
        correct_state_dict = OrderedDict([k[7:],v] for k,v in loaded_state_dict.items())
        
        # if model was trained w grayscale images, need to reduce size of first weights
        in_channels = correct_state_dict['conv1.weight'].shape[1]
        if in_channels<3:
            model.conv1 = nn.Conv2d(in_channels, model.conv1.out_channels, \
                                    kernel_size=model.conv1.kernel_size, \
                                    stride=model.conv1.stride, \
                                    padding=model.conv1.padding, \
                                    bias=model.conv1.bias, device=device)
        
        # if model was trained with ecoset, need to fix dim of output
        n_out_feat_saved = correct_state_dict['fc.bias'].shape[0]
        model.fc = nn.Linear(model.fc.in_features, n_out_feat_saved, device=device)

        print(model.conv1.weight.shape)
        print(model.fc.weight.shape)
        # now load the saved weights into our model
        model.load_state_dict(correct_state_dict)

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
            hooks[ii].remove

    # Sanity check that we grabbed the right activations - check their sizes against expected
    # output size of each block
    exp_size = np.array(n_features_each_resnet_block)[block_inds]
    actual_size = [activ[bb].shape[1] for bb in range(len(activ))]
    assert(np.all(np.array(actual_size)==np.array(exp_size)))

    return activ


def proc_one_subject(subject, args):
    
    model_architecture_str='RN50'
    
    if args.training_type=='clip':
        feat_path = default_paths.clip_feat_path
    elif args.training_type=='blurface':
        feat_path = default_paths.resnet50_blurface_feat_path
    elif args.training_type=='imgnet':
        feat_path = default_paths.resnet50_feat_path
    elif 'startingblurry' in args.training_type:
        feat_path = default_paths.resnet50_startingblurry_feat_path
        model_architecture_str += '_%s'%args.training_type.split('startingblurry_')[1]
       
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
        image_data = coco_utils.load_indep_coco_images(n_pix=240)
        image_data = nsd_utils.image_uncolorize_fn(image_data)
        fit_inds = np.ones((10000,),dtype=bool)
    else: 
        # load all images for the current subject, 10,000 ims
        image_data = nsd_utils.get_image_data(subject)  
        image_data = nsd_utils.image_uncolorize_fn(image_data)
        subject_df = nsd_utils.get_subj_df(subject)
        fit_inds = np.array(subject_df['shared1000']==False)
   
    if args.n_layers_save==16:
        blocks_to_do = np.arange(16)
    elif args.n_layers_save==8:
        blocks_to_do = np.arange(0,16,2)+1
    elif args.n_layers_save==4:
        blocks_to_do = [2,6,12,15]
        
    if args.start_layer>0:
        blocks_to_do = blocks_to_do[args.start_layer:]
   
    
    for ll in blocks_to_do:

        # each batch will be in a separate file, since they're big features
        
        block_inds = [ll]
        
        if ll<=2:
            # reduce size of larger feature maps
            pooling_op = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        else:
            pooling_op = None
            
        # first extract features for all pixels/feature channels 
        features_raw = extract_features(image_data,\
                                        block_inds,\
                                        pooling_op = pooling_op,
                                        save_dtype=np.float32,\
                                        training_type=args.training_type, \
                                        debug=args.debug)

        # then do pca to make these less huge
        layer_name = 'block%d'%(ll)
        filename_save_pca = os.path.join(path_to_save, \
                                          'S%d_%s_%s_noavg_PCA_grid0.h5py'%\
                                    (subject, model_architecture_str, layer_name))
        
        if args.save_pca_weights:
            save_weights_filename = os.path.join(path_to_save, \
                                    'S%d_%s_%s_noavg_PCA_weights_grid0.npy'%\
                                    (subject, model_architecture_str, layer_name))
        else:
            save_weights_filename = None

        if args.use_saved_ncomp:
            ncomp_filename = os.path.join(path_to_save, \
                                    'S%d_%s_%s_noavg_PCA_grid0_ncomp.npy'%\
                                    (subject, model_architecture_str, layer_name))
        else:
            ncomp_filename = None


        pca_feats.run_pca_oneprf(features_raw, 
                                    filename_save_pca, 
                                    fit_inds=fit_inds, 
                                    min_pct_var=args.min_pct_var, 
                                    max_pc_to_retain=args.max_pc_to_retain,
                                    save_weights=args.save_pca_weights==1, 
                                    save_weights_filename=save_weights_filename, 
                                    use_saved_ncomp=args.use_saved_ncomp, 
                                    ncomp_filename=ncomp_filename,\
                                    save_dtype=np.float32, compress=True, \
                                    debug=args.debug)


            
def proc_other_image_set(image_set, args):
    
    model_architecture_str='RN50'
    
    if args.training_type=='clip':
        feat_path = default_paths.clip_feat_path
    elif args.training_type=='blurface':
        feat_path = default_paths.resnet50_blurface_feat_path
    elif args.training_type=='imgnet':
        feat_path = default_paths.resnet50_feat_path
    elif 'startingblurry' in args.training_type:
        feat_path = default_paths.resnet50_startingblurry_feat_path
        model_architecture_str += '_%s'%args.training_type.split('startingblurry_')[1]
       
    if args.debug:
        feat_path = os.path.join(feat_path,'DEBUG')
    
    path_to_save = os.path.join(feat_path, 'PCA')
        
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
     
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
    
    
    subjects_pca = np.arange(1,9)
    
    for ll in blocks_to_do:

        block_inds = [ll]
        
        if ll<=2:
            # reduce size of larger feature maps
            pooling_op = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        else:
            pooling_op = None
            
        # first extract features for all pixels/feature channels 
        features_raw = extract_features(image_data,\
                                        block_inds,\
                                        pooling_op = pooling_op, 
                                        save_dtype=np.float32,\
                                        training_type=args.training_type, \
                                        debug=args.debug)

        layer_name = 'block%d'%(ll)
        
        for ss in subjects_pca:

            load_weights_filename = os.path.join(path_to_save, 'S%d_%s_%s_noavg_PCA_weights_grid0.npy'%\
                                    (ss, model_architecture_str, layer_name))
            filename_save_pca = os.path.join(path_to_save, \
                                    '%s_%s_%s_noavg_PCA_wtsfromS%d_grid0.h5py'%\
                                    (image_set, model_architecture_str, layer_name, ss))

            pca_feats.apply_pca_oneprf(features_raw, 
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
    parser.add_argument("--training_type", type=str,default='clip',
                    help="what kind of model training was used?")
    
    parser.add_argument("--start_layer", type=int,default=0,
                    help="which network layer to start from?")
    parser.add_argument("--n_layers_save", type=int,default=16,
                    help="how many layers to save?")
    
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
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
        

        
        