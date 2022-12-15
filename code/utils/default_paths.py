# this file sets relative paths to folders of interest for our project
# see path_defs.py to change the root directory/absolute paths

import os
import path_defs

nsd_path = path_defs.nsd_path
nsd_root = nsd_path 
root = path_defs.root
root_localnode = path_defs.root_localnode
project_name = path_defs.project_name

# Where we are keeping the preprocessed NSD stimuli/labeling data
stim_root = os.path.join(root, 'nsd','stimuli')    
stim_labels_root = os.path.join(root, 'nsd','labels')    
nsd_rois_root = os.path.join(root, 'nsd', 'rois')

# Where to save model fits
save_fits_path = os.path.join(root, project_name, 'model_fits')

# Where to save any figures we make
fig_path = os.path.join(root, project_name, 'figures')

# Path where gabor model features will be saved
gabor_texture_feat_path = os.path.join(root, 'features','gabor_texture')

# Path where texture model features will be saved
pyramid_texture_feat_path = os.path.join(root, 'features', 'pyramid_texture')

# Path where sketch token features will be saved
sketch_token_feat_path = os.path.join(root, 'features', 'sketch_tokens')
sketch_token_feat_path_localnode = os.path.join(root_localnode, 'features', 'sketch_tokens')

# Path where AlexNet features will be saved
alexnet_feat_path = os.path.join(root, 'features','alexnet')
alexnet_blurface_feat_path = os.path.join(root, 'features','alexnet_blurface')

# Path where CLIP features will be saved
clip_feat_path = os.path.join(root, 'features', 'CLIP')

# Path where ResNet50 features will be saved
resnet50_feat_path = os.path.join(root, 'features', 'resnet50')
resnet50_startingblurry_feat_path = os.path.join(root, 'features', 'resnet50_startingblurry')
resnet50_blurface_feat_path = os.path.join(root, 'features', 'resnet50_blurface')

# Path where color model features will be saved
color_feat_path = os.path.join(root, 'features','color')

# Path where gist model features will be saved
gist_feat_path = os.path.join(root, 'features','gist')

# Path where spatial-color model features will be saved
spatcolor_feat_path = os.path.join(root, 'features','spatcolor')

# Where the raw NSD beta weights are located
beta_root = os.path.join(nsd_root,'nsddata_betas','ppdata')

coco_api_path = path_defs.coco_api_path
coco_ims_path = path_defs.coco_ims_path

floc_image_root = path_defs.floc_image_root
food_image_root = path_defs.food_image_root

startingblurry_root = path_defs.startingblurry_root

# needed for face labeling code
retinaface_path = path_defs.retinaface_path