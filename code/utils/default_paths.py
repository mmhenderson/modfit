
import os
# Path to the full NSD data repository (includes data and stimuli)
# http://naturalscenesdataset.org/
nsd_path = '/lab_data/tarrlab/common/datasets/NSD'
nsd_root = nsd_path
stim_root = '/user_data/mmhender/nsd_stimuli/stimuli/nsd/'     
beta_root = os.path.join(nsd_root,'nsddata_betas','ppdata')

# Path to the COCO API toolbox
# https://github.com/cocodataset/cocoapi
coco_api_path = '/user_data/mmhender/toolboxes/coco_annot'

# Path where AlexNet features will be saved
alexnet_activ_path = '/lab_data/tarrlab/common/datasets/features/NSD/alexnet_full'