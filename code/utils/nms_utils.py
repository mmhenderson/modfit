import os, sys
import cv2
import numpy as np
import torch

path_to_toolbox = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))), 'toolboxes','StructuredForests')
sys.path.append(path_to_toolbox)
import pyximport
pyximport.install(build_dir=".pyxbld",
                  setup_args={"include_dirs": np.get_include()})
from _StructuredForests import non_maximum_supr

"""
Code is from the repository at:
https://github.com/ArtanisCV/StructuredForests
In this file are several utility functions from that repo that have been modified to work with python 3. 
Uses their 'non-maximum suppression' implementation, located in a .pyx file stored in the StructuredForests folder.
"""   

def apply_nms(edge_map_batch):
    
    """ 
    Apply non-maximum suppression to a batch of images (can use tensor or numpy array).
    Expected size is [batch_size x channels x height x width] 
    """
    
    if not isinstance(edge_map_batch, np.ndarray):
        device = edge_map_batch.device
        edge_map_batch = edge_map_batch.detach().cpu().numpy()
        was_tensor = True
    else:
        was_tensor = False
        
    if not len(edge_map_batch.shape)==4:
        raise ValueError('input should be [batch_size x nchannels x height x width]')

    edge_map_batch = edge_map_batch.astype('float64')
    edge_map_batch_supr = np.zeros(shape=edge_map_batch.shape, dtype='float64')
    for bb in range(edge_map_batch.shape[0]):
        for cc in range(edge_map_batch.shape[1]):

            E = edge_map_batch[bb,cc,:,:]
            [Ox, Oy] = gradient(conv_tri(E, 4))
            [Oxx, _] = gradient(Ox)
            [Oxy, Oyy] = gradient(Oy)
            O = np.mod(np.arctan(Oyy * np.sign(-Oxy)/(Oxx+1e-5)), np.pi)
            E_supr = non_maximum_supr(E, O, 1, 5, 1.01)
            
            edge_map_batch_supr[bb,cc,:,:] = E_supr

    if was_tensor:
        edge_map_batch_supr = torch.tensor(edge_map_batch_supr, device=device);

    return edge_map_batch_supr

def conv_tri(src, radius):
    """
    Image convolution with a triangle filter.
    :param src: input image
    :param radius: gradient normalization radius
    :return: convolution result
    """

    if radius == 0:
        return src
    elif radius <= 1:
        p = 12.0 / radius / (radius + 2) - 2
        kernel = np.asarray([1, p, 1], dtype=np.float64) / (p + 2)
        return cv2.sepFilter2D(src, ddepth=-1, kernelX=kernel, kernelY=kernel,
                               borderType=cv2.BORDER_REFLECT)
    else:
        radius = int(radius)
        kernel = list(range(1, radius + 1)) + [radius + 1] + list(range(radius, 0, -1))
        kernel = np.asarray(kernel, dtype=np.float64) / (radius + 1) ** 2
        return cv2.sepFilter2D(src, ddepth=-1, kernelX=kernel, kernelY=kernel,
                               borderType=cv2.BORDER_REFLECT)


def gradient(src, norm_radius=0, norm_const=0.01):
    """
    Compute gradient magnitude and orientation at each image location.
    :param src: input image
    :param norm_radius: normalization radius (no normalization if 0)
    :param norm_const: normalization constant
    :return: gradient magnitude and orientation (0 ~ pi)
    """

    if src.ndim == 2:
        src = src[:, :, None]

    dx = np.zeros(src.shape, dtype=src.dtype)
    dy = np.zeros(src.shape, dtype=src.dtype)
    for i in np.arange(0,src.shape[2]):
        dy[:, :, i], dx[:, :, i] = np.gradient(src[:, :, i])

    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    idx_2 = np.argmax(magnitude, axis=2)
    idx_0, idx_1 = np.indices(magnitude.shape[:2])
    magnitude = magnitude[idx_0, idx_1, idx_2]
    if norm_radius != 0:
        magnitude /= conv_tri(magnitude, norm_radius) + norm_const
    magnitude = magnitude.astype(src.dtype, copy=False)

    dx = dx[idx_0, idx_1, idx_2]
    dy = dy[idx_0, idx_1, idx_2]
    orientation = np.arctan2(dy, dx)
    orientation[orientation < 0] += np.pi
    orientation[np.abs(dx) + np.abs(dy) < 1e-5] = 0.5 * np.pi
    orientation = orientation.astype(src.dtype, copy=False)

    return magnitude, orientation