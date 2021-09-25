
import os, platform
if platform.system()=='Darwin':
    testing_on_mac = True
else:
    testing_on_mac = False
    
# Path to the full NSD data repository (includes data and stimuli)
# http://naturalscenesdataset.org/
if testing_on_mac:
    root = '/Users/margarethenderson/Box Sync/'
    nsd_path = os.path.join(root, 'nsd_betas_for_testing/')     
else:
    root = '/user_data/mmhender/'
    nsd_path = '/lab_data/tarrlab/common/datasets/NSD'   
    
nsd_root = nsd_path
beta_root = os.path.join(nsd_root,'nsddata_betas','ppdata')
stim_root = os.path.join(root, 'nsd_stimuli/stimuli/nsd/')    

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

# Path where all the BDCN code lives
bdcn_path = '/user_data/mmhender/toolboxes/BDCN/'

# Path where AlexNet features will be saved
alexnet_activ_path = '/lab_data/tarrlab/common/datasets/features/NSD/alexnet_full'

# Path to the COCO API toolbox
# https://github.com/cocodataset/cocoapi
coco_api_path = '/user_data/mmhender/toolboxes/coco_annot'
