
import os
# Path to the full NSD data repository (includes data and stimuli)
# http://naturalscenesdataset.org/
nsd_path = '/lab_data/tarrlab/common/datasets/NSD'
nsd_root = nsd_path
stim_root = '/user_data/mmhender/nsd_stimuli/stimuli/nsd/'     
beta_root = os.path.join(nsd_root,'nsddata_betas','ppdata')

# Where to save model fits
save_fits_path = '/user_data/mmhender/imStat/model_fits/'

# Path to the COCO API toolbox
# https://github.com/cocodataset/cocoapi
coco_api_path = '/user_data/mmhender/toolboxes/coco_annot'

# Path where gabor model features will be
gabor_texture_feat_path = '/user_data/mmhender/features/gabor_texture/'
gabor_texture_feat_path_localnode = '/scratch/mmhender/features/gabor_texture/'

# Path where pyramid texture model features will be
pyramid_texture_feat_path = '/user_data/mmhender/features/pyramid_texture/'
pyramid_texture_feat_path_localnode = '/scratch/mmhender/features/pyramid_texture/'

# Path where sketch token features will be
sketch_token_feat_path = '/user_data/mmhender/features/sketch_tokens/'
sketch_token_feat_path_localnode = '/scratch/mmhender/features/sketch_tokens/'

# Path where all the BDCN code lives
bdcn_path = '/user_data/mmhender/toolboxes/BDCN/'

# Path where AlexNet features will be saved
alexnet_activ_path = '/lab_data/tarrlab/common/datasets/features/NSD/alexnet_full'
