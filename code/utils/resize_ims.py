import sys
code_dir = '/user_data/mmhender/imStat/code/'
sys.path.append(code_dir)
from utils import nsd_utils
from utils import default_paths
nsd_root = default_paths.nsd_root
path_to_save = default_paths.stim_root

if __name__ == '__main__':
    
    nsd_utils.get_subject_specific_images(nsd_root, path_to_save, npix=240)

