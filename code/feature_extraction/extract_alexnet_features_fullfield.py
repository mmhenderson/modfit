import numpy as np
import sys, os
import argparse
import gc
import torch
import time
import h5py
import copy
from collections import OrderedDict
import torchvision.models as models
import torch.nn as nn

#import custom modules
from utils import prf_utils, torch_utils, texture_utils, default_paths, nsd_utils, floc_utils
from model_fitting import initialize_fitting
from feature_extraction import pca_feats

dtype=np.float32

# Define sets of alexnet layers
alexnet_conv_layer_names = ['Conv1','Conv1_ReLU','Conv1_MaxPool', \
                       'Conv2','Conv2_ReLU','Conv2_MaxPool', \
                       'Conv3','Conv3_ReLU', \
                       'Conv4','Conv4_ReLU', \
                       'Conv5','Conv5_ReLU','Conv5_MaxPool']

alexnet_fc_layer_names = ['Dropout6','FC6','FC6_ReLU','Dropout7','FC7','FC7_ReLU','FC8']

n_feature_layers = len(alexnet_conv_layer_names)
n_classif_layers = len(alexnet_fc_layer_names)
n_total_layers = n_feature_layers + n_classif_layers
alexnet_layer_names = copy.deepcopy(alexnet_conv_layer_names)
alexnet_layer_names.extend(alexnet_fc_layer_names)

n_features_each_layer = [64,64,64, 192,192,192, 384,384, 256,256, 256,256]

try:
    device = initialize_fitting.init_cuda()
except:
    device = 'cpu:0'


def extract_features(image_data, layer_inds,\
                          padding_mode=None,\
                          pooling_op=None, 
                          batch_size=50, \
                          blurface=False, \
                          debug=False):

    """
    Extract the portion of CNN feature maps corresponding to pRF defined in "models"
    Return list of the features in each pRF, for each layer of interest.
    """

    assert(len(layer_inds)==1)
    ll = layer_inds[0]
    
    if padding_mode=='':
        padding_mode = None
     
    n_images = image_data.shape[0]
    n_batches = int(np.ceil(n_images/batch_size))

    # figure out how big features will be, by passing a test image through
    image_template = np.tile(image_data[0:1,:,:,:], [1,3,1,1])
    activ_template = get_alexnet_activations_batch(image_template, layer_inds, \
                                                        device=device, \
                                                        padding_mode=padding_mode, \
                                                        blurface=blurface)
    if pooling_op is not None:
        activ_template = [pooling_op(activ_template[0])]
        
    n_features_total = np.prod(activ_template[0].shape[1:])  
    print('number of features total: %d'%n_features_total)
    
    
    features = np.zeros((n_images, n_features_total),dtype=dtype)

    with torch.no_grad():

        for bb in range(n_batches):

            if debug and bb>1:
                continue
            print('Processing images for batch %d of %d'%(bb, n_batches))

            batch_inds = np.arange(batch_size * bb, np.min([batch_size * (bb+1), n_images]))

            # using grayscale images for better comparison w my other models.
            # need to tile to 3 so alexnet weights will be right size
            image_batch = np.tile(image_data[batch_inds,:,:,:], [1,3,1,1])

            gc.collect()
            torch.cuda.empty_cache()
            
            activ_batch = get_alexnet_activations_batch(image_batch, layer_inds, \
                                                        device=device, \
                                                        padding_mode=padding_mode, \
                                                        blurface=blurface)
            
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
    

def get_alexnet_activations_batch(image_batch, layer_inds, device=None, padding_mode=None, blurface=False):

    """
    Get activations for images in NSD, passed through pretrained AlexNet.
    Specify which NSD images to look at, and which layers to return.
    """

    if device is None:
        device = torch.device('cpu:0')
       
    if blurface:
        # use model trained on face-blurred imagenet ims
        model_filename = os.path.join(default_paths.alexnet_blurface_feat_path, 'alexnet_blurred_ILSVRC.pth')
        print('Loading saved model from %s'%model_filename)
        saved = torch.load(model_filename, map_location=device)     
        sd = saved['state_dict']
        
        # need to fix keys in the statedict to have expected names
        new_sd = OrderedDict([])
        keynames = list(sd.keys())
        for kn in keynames:
            if 'features' in kn:
                new_kn = 'features.' + kn.split('features.module.')[1]
            else:
                new_kn = kn
            new_sd[new_kn] = sd[kn]
            
        model = models.alexnet().float().to(device)
        model.load_state_dict(new_sd)

    else:       
        # normal pre-trained model from torch model zoo
        print('Loading alexnet pre-trained on imagenet')
        model = models.alexnet(pretrained=True).float().to(device)
        
        
    if padding_mode is not None:
        # change padding type for all convolutional layers, "reflect" is a
        # good way to minimize edge artifacts.
        for ff in model.features:
            if hasattr(ff, 'padding_mode'):
                ff.padding_mode=padding_mode
                print('changing padding mode to %s'%padding_mode)
                print(ff)
                
                
    model.eval()
    
    is_fc = [('FC' in alexnet_layer_names[ll] or 'fc' in alexnet_layer_names[ll]) for ll in layer_inds]
    
    if len(layer_inds)==0:
        raise ValueError('your layer names do not match any of those specified in alexnet_features.py')

    
    # first making this subfunction that is needed to get the activation on a forward pass
    def get_activ_fwd_hook(ii,ll):
        def hook(self, input, output):            
            print('hook for %s'%alexnet_layer_names[ll])           
            activ[ii] = output
            print(output.shape)
        return hook
   
    # get image and labels for this batch
    # image_tensors is [batch_size x 3 x 224 x 224]
    image_tensors =  torch_utils._to_torch(image_batch, device=device).float()
    activ = [[] for ll in layer_inds]
    hook = [[] for ll in layer_inds]
    
    model.eval()

    # adding this "hook" to the module corresponding to each layer, so we'll save activations at each layer
    # this only modifies the "graph" e.g. what the model code does when run, but doesn't actually run it yet.
    for ii, ll in enumerate(layer_inds):
        if not is_fc[ii]:
            h = model.features[ll].register_forward_hook(get_activ_fwd_hook(ii,ll))
        else:
            h = model.classifier[ll-n_feature_layers].register_forward_hook(get_activ_fwd_hook(ii,ll))
        hook[ii] = h

    # do the forward pass of model, which now includes the forward hooks
    # now the "activ" variable will get modified, because it gets altered during the hook function
    model(image_tensors)
    
    # Now remove all the hooks
    for ii, ll in enumerate(layer_inds):
        print(activ[ii].shape)
        hook[ii].remove

    return activ




def proc_one_subject(subject, args):

    if args.blurface==1:
        feat_path = default_paths.alexnet_blurface_feat_path
    else:
        feat_path = default_paths.alexnet_feat_path
        
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
    
    layers_to_return = ['Conv1_ReLU', 'Conv2_ReLU','Conv3_ReLU','Conv4_ReLU','Conv5_ReLU','FC6_ReLU','FC7_ReLU']
    layer_inds = [ll for ll in range(len(alexnet_layer_names)) \
                      if alexnet_layer_names[ll] in layers_to_return]
    if args.start_layer>0:
        layer_inds = layer_inds[args.start_layer:]

    for ll in layer_inds:

        pooling_op = None
            
        features_raw = extract_features(image_data, \
                                             layer_inds = [ll],
                                             padding_mode=args.padding_mode, \
                                             batch_size=args.batch_size,
                                             blurface=args.blurface==1, 
                                             debug=args.debug)
        # then do pca to make these less huge
        layer_name = alexnet_layer_names[ll]
        filename_save_pca = os.path.join(path_to_save, \
                                          'S%d_%s_%s_noavg_PCA_grid0.h5py'%\
                                    (subject, layer_name, args.padding_mode))
        
        if args.save_pca_weights:
            save_weights_filename = os.path.join(path_to_save, \
                                    'S%d_%s_%s_noavg_PCA_weights_grid0.npy'%\
                                    (subject,layer_name, args.padding_mode))
        else:
            save_weights_filename = None

        if args.use_saved_ncomp:
            ncomp_filename = os.path.join(path_to_save, \
                                    'S%d_%s_%s_noavg_PCA_grid0_ncomp.npy'%\
                                    (subject, layer_name, args.padding_mode))
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
       
    if args.blurface==1:
        feat_path = default_paths.alexnet_blurface_feat_path
    else:
        feat_path = default_paths.alexnet_feat_path
      
    if args.debug:
        feat_path = os.path.join(feat_path,'DEBUG')
    
    path_to_save = os.path.join(feat_path, 'PCA')
        
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
      
    if image_set=='floc':
        image_data = floc_utils.load_floc_images(npix=240)
    else:
        raise ValueError('image set %s not recognized'%image_set)
       
    layers_to_return = ['Conv1_ReLU', 'Conv2_ReLU','Conv3_ReLU','Conv4_ReLU','Conv5_ReLU','FC6_ReLU','FC7_ReLU']
    layer_inds = [ll for ll in range(len(alexnet_layer_names)) \
                      if alexnet_layer_names[ll] in layers_to_return]
    if args.start_layer>0:
        layer_inds = layer_inds[args.start_layer:]

    for ll in layer_inds:

        pooling_op = None
        
        features_raw = extract_features(image_data, \
                                             layer_inds = [ll],
                                             padding_mode=args.padding_mode, \
                                             batch_size=args.batch_size,
                                             blurface=args.blurface==1, 
                                             debug=args.debug)
    
        layer_name = alexnet_layer_names[ll]
        
        subjects_pca = np.arange(1,9)
        
        for ss in subjects_pca:

            filename_save_pca = os.path.join(path_to_save, \
                                              '%s_%s_%s_noavg_PCA_wtsfromS%d_grid0.h5py'%\
                                        (image_set, layer_name, args.padding_mode, ss))
            load_weights_filename = os.path.join(path_to_save, \
                                        'S%d_%s_%s_noavg_PCA_weights_grid0.npy'%\
                                        (ss,layer_name, args.padding_mode))
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
    parser.add_argument("--start_layer", type=int,default=0,
                    help="which network layer to start from?")
    parser.add_argument("--batch_size", type=int,default=50,
                    help="batch size")
    
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")

    parser.add_argument("--padding_mode", type=str,default='',
                    help="padding mode for alexnet convolutional layers")
    parser.add_argument("--blurface", type=int, default=0, 
                    help="use model trained with faces blurred? 1 for yes, 0 for no")
    
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
        
