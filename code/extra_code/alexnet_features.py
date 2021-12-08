import numpy as np
import copy
import torch
import torchvision.models as models

from utils import torch_utils, nsd_utils
from model_fitting import initialize_fitting

# some OLD code that i am not using, snippets may be useful in future

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


def get_alexnet_nsd_activations(layer_name, nsd_indices, batch_size=100, debug=False, device=None):

    """
    Get activations for images in NSD, passed through pretrained AlexNet.
    Specify which NSD images to look at, and which layer to return.
    """

    if device is None:
        device = torch.device('cpu:0')
       
    # first loading pre-trained model from torch model zoo
    model = models.alexnet(pretrained=True).double().to(device)
    model.eval()
    model_name='AlexNet'
    
    dset = nsd_utils.nsd_dataset(transform=nsd_utils.get_transform(224), device=device, nsd_inds_include=nsd_indices)
    dloader = nsd_utils.nsd_dataloader(dset, batch_size=batch_size, shuffle=False)
    dgenerator = iter(dloader)
    n_batches = dloader.n_batches
    n_total_images = len(dset)
    
    layer_index = np.where(np.array(alexnet_layer_names)==layer_name)[0]
    if len(layer_index)==0:
        raise ValueError('your layer name does not match one of those specified in alexnet_features.py')
    layer_index = layer_index[0]
    
    is_fc = 'FC' in layer_name or 'fc' in layer_name
    
    # first making this subfunction that is needed to get the activation on a forward pass
    def get_activ_fwd_hook(ii):
        def hook(self, input, output):            
            print('hook for %s'%layer_name)           
            activ[ii] = torch_utils.get_value(output)
            print(output.shape)
        return hook
    
    for bb in range(n_batches):
        if bb>1 and debug:
            break

        print('\nGetting activs for batch %d of %d'%(bb, n_batches))
        curr_batch = next(dgenerator)

        # get image and labels for this batch
        # image_tensors is [batch_size x 3 x 224 x 224]
        image_tensors =  curr_batch['image']
        activ = list([[]])
        hook = None

        model.eval()

        # adding this "hook" to the module corresponding to each layer, so we'll save activations at each layer
        # this only modifies the "graph" e.g. what the model code does when run, but doesn't actually run it yet.
        ii=0
        if not is_fc:
            hook = model.features[layer_index].register_forward_hook(get_activ_fwd_hook(ii))
        else:
            hook = model.classifier[layer_index-n_feature_layers].register_forward_hook(get_activ_fwd_hook(ii))

        # do the forward pass of model, which now includes the forward hooks
        # now the "activ" variable will get modified, because it gets altered during the hook function
        model(image_tensors)
        print(activ[ii].shape)
        hook.remove()
        
        if bb==0:
            activ_all = np.zeros([n_total_images, np.prod(activ[ii].shape[1:])])
            if not is_fc:
                coords = get_NCHW_coords(activ[ii].shape, order='F')
            else:
                coords = np.expand_dims(np.arange(0,np.prod(activ[ii].shape[1:])), axis=1)

        # full activ here is NCHW format: Batch Size x Channels x Height (top to bottom) x Width (left to right)
        # new activ will be nIms x nFeatures
        batch_size_actual = activ[ii].shape[0]
        activ_all[bb*batch_size:(bb+1)*batch_size,:] = np.reshape(activ[ii], [batch_size_actual, -1], order='F')
  
    return activ_all, coords


def get_NCHW_coords(shape, order='F'):
    
    """
    Return coordinates (channel index, position in width dimension, position in height dimension) of feature map units. 
    For feature maps that originally had NCHW shape specified in 'shape', and have been reshaped into a long list of all features.
    Can specify which way the array was reshaped (i.e. when combining all features into a single list) as 'F' or 'C' 
    (see np.reshape for meaning of these)
    """
    
    N = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]
    if order=='F':
        clabs = np.expand_dims(np.tile(np.arange(0,C), W*H), axis=1)
        wlabs = np.expand_dims(np.repeat(np.arange(0,W),C*H),axis=1)  
        hlabs = np.expand_dims(np.repeat(np.tile(np.expand_dims(np.arange(0,H),axis=1),[W,1]), C),axis=1)        
    elif order=='C':
        clabs = np.expand_dims(np.repeat(np.arange(0,C), W*H), axis=1)        
        wlabs = np.expand_dims(np.tile(np.arange(0,W),C*H),axis=1)  
        hlabs = np.expand_dims(np.repeat(np.tile(np.expand_dims(np.arange(0,H),axis=1),[C,1]), W),axis=1)       
       
    # the coords matrix goes [nUnits x 3] where columns are [H,W,C]
    coords = np.concatenate((clabs,hlabs,wlabs),axis=1) 
    print(len(np.unique(coords, axis=0)))
    assert(len(np.unique(coords, axis=0))==C*H*W)

    return coords