import sys
from utils import nsd_utils

if __name__ == '__main__':
    
    for ss in range(8):
        
        nsd_utils.get_concat_betas(ss+1, debug=False)

