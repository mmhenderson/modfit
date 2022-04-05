
import os

# Path to the full NSD data repository (includes data and stimuli)
# http://naturalscenesdataset.org/
root = '/user_data/mmhender/'
nsd_path = '/lab_data/tarrlab/common/datasets/NSD'   
    
nsd_root = nsd_path
beta_root = os.path.join(nsd_root,'nsddata_betas','ppdata')
stim_root = os.path.join(root, 'nsd/stimuli/')    
stim_labels_root = os.path.join(root, 'nsd/labels/')    
nsd_data_concat_root = os.path.join(root, 'nsd/data/')

# Where to save model fits
save_fits_path = os.path.join(root,'imStat/model_fits/')

# Path where gabor model features will be
gabor_texture_feat_path = os.path.join(root, 'features/gabor_texture/')
gabor_texture_feat_path_localnode = '/scratch/mmhender/features/gabor_texture/'

# Path where pyramid texture model features will be
pyramid_texture_feat_path = os.path.join(root, 'features/pyramid_texture/')
pyramid_texture_feat_path_localnode = '/scratch/mmhender/features/pyramid_texture/'

# Path where sketch token features will be
sketch_token_feat_path = os.path.join(root, 'features/sketch_tokens/')
sketch_token_feat_path_localnode = '/scratch/mmhender/features/sketch_tokens/'

# Path where AlexNet features will be saved
alexnet_feat_path = os.path.join(root, 'features/alexnet/')
alexnet_feat_path_localnode = '/scratch/mmhender/features/alexnet/'

# Path where CLIP features will be saved
clip_feat_path = os.path.join(root, 'features/CLIP/')
clip_feat_path_localnode = '/scratch/mmhender/features/CLIP/'

# Path to the COCO API toolbox
# https://github.com/cocodataset/cocoapi
coco_api_path = '/user_data/mmhender/toolboxes/coco_annot'
coco_ims_path = '/lab_data/tarrlab/common/datasets/COCO'