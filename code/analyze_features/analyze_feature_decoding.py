import os
import numpy as np

import scipy.stats
import pandas as pd
import statsmodels
import statsmodels.stats.multitest

from utils import default_paths, prf_utils
from model_fitting import initialize_fitting

def load_floc_decoding(feature_type, which_prf_grid=5, verbose=False):

    path_to_load = get_path(feature_type)
    path_to_load = os.path.join(path_to_load, 'feature_decoding_floc')
    
    image_set='floc'
    fn1 = os.path.join(path_to_load, '%s_%s_LDA_all_grid%d.npy'%(image_set, feature_type, which_prf_grid))
   
    if verbose:
        print('loading from %s'%fn1)
    decoding = np.load(fn1,allow_pickle=True).item()
    
    names = decoding['discrim_type_list']
    acc = decoding['acc']
    dprime = decoding['dprime']
    pairwise_acc = decoding['pairwise_acc']
    pairwise_dprime = decoding['pairwise_dprime']
    
    return acc, dprime, pairwise_acc, pairwise_dprime, names

