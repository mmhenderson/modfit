"""
Run the model fitting for FWRF model. 
There are a few different versions of fitting in this script, the input arguments tell which kind of fitting to do.
"""

# import basic modules
import sys
import os
import time
import numpy as np
from tqdm import tqdm
import gc
import torch
import argparse
import skimage.transform

# import custom modules
from utils import nsd_utils
from model_fitting import initialize_fitting

fpX = np.float32
device = initialize_fitting.init_cuda()

if __name__ == '__main__':
    
    tst = nsd_utils.get_image_data(subject=1, shuffle_images=False, random_images=False)
