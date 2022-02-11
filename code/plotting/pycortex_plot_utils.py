import numpy as np
import copy
import cortex

def plot_maps_pycortex(subject, port, maps, names, mins=None, maxes=None, cmaps=None, \
                        title=None, vox2plot = None, roi_def=None,  \
                        volume_space=None, nii_shape=None, voxel_mask=None):

    """
    Plot a set of maps in pycortex surface space, using cortex.webshow()
    
    subject: nsd subject number, 1-8    
    port: port number to use for cortex.webshow()
    maps: list of arrays, each size [n_voxels,] to plot on surface
    names: list of strings, names of the maps
    mins: optional list of minimum values for each map's colorbar
    maxes: optional list of max values for each map's colorbar
    cmaps: optional list of colormap names, i.e. 'RdBu'
    title: optional string describing the analysis
    voxel_mask: boolean mask of size prod(nii_shape), indicates which voxels in the
        whole brain the n_voxels in each map correspond to.
    roi_def: optional, an roi definition object from roi_utils.nsd_roi_def()
        roi_def includes fields for voxel_mask, nii_shape, and volume_space, 
        so if roi_def is specified, no need to specify those other arguments.
        If roi_def is not specified, must specify voxel_mask and nii_shape.
        If it is is specified, this code will also plot ROI masks as additional
        maps in the webviewer.
    vox2plot: optional mask for which voxels to include, size [n_voxels,]
    volume_space: is the data in volume space (from a 3D nifti file?)
        otherwise assume it is in surface space (i.e. each voxel is a mesh vertex)
    nii_shape: shape of the original nifti file 
    
    """
    
    substr = 'subj%02d'%subject
  
    if roi_def is not None:
        dat2plot = get_roi_maps_for_pycortex(subject, roi_def)
        volume_space = roi_def.volume_space
        voxel_mask = roi_def.voxel_mask
        nii_shape = roi_def.nii_shape
    else:
        if voxel_mask is None:
            raise ValueError('must specify either voxel_mask or roi_def.')   
        if volume_space is None:
            volume_space = nii_shape is not None
        else:
            if nii_shape is None:
                raise ValueError('must specify nii_shape if volume_space=True.')   
        dat2plot = {}
     
    if volume_space:
        xfmname = 'func1pt8_to_anat0pt8_autoFSbbr'
        mask_3d = np.reshape(voxel_mask, nii_shape, order='C')

    if mins is None:
        mins = [None for ll in range(len(names))]
    elif len(mins)==1:
        mins = [mins[0] for ll in range(len(names))]
    if maxes is None:
        maxes = [None for ll in range(len(names))]
    elif len(maxes)==1:
        maxes = [maxes[0] for ll in range(len(names))]
    if cmaps is None:
        cmaps = ['PuBu' for ll in range(len(names))]
    elif len(cmaps)==1:
        cmaps = [cmaps[0] for ll in range(len(names))]  
    
    for ni, name in enumerate(names):
        map_thresh = copy.deepcopy(maps[ni])
        if vox2plot is not None:          
            map_thresh[~vox2plot] = np.nan
        if volume_space:
            dat2plot[name] = cortex.Volume(data = get_full_volume(map_thresh, voxel_mask, nii_shape), \
                                           cmap=cmaps[ni], subject=substr, \
                                           vmin=mins[ni], vmax=maxes[ni],\
                                           xfmname=xfmname, mask=mask_3d)
        else:
            dat2plot[name] = cortex.Vertex(data = get_full_surface(map_thresh, voxel_mask),\
                                           cmap=cmaps[ni], subject=substr, \
                                           vmin=mins[ni], vmax=maxes[ni])
        
    # Open the webviewer
    print('navigate browser to: 127.0.0.1:%s'%port)
    cortex.webshow(dat2plot, open_browser=True, port=port, \
                   title = 'S%02d, %s'%(subject, title))

def get_roi_maps_for_pycortex(subject, roi_def):    
    """
    Create a dictionary of cortex.Vertex labels for ROIs, to be plotted in PyCortex.
    
    subject: nsd subject number, 1-8
    roi_def: an roi definition object from roi_utils.nsd_roi_def()
     
    """ 
    substr = 'subj%02d'%subject
    
    retlabs = roi_def.retlabs
    facelabs = roi_def.facelabs
    placelabs = roi_def.placelabs
    bodylabs = roi_def.bodylabs
    retlabs[retlabs==-1] = np.nan
    facelabs[facelabs==-1] = np.nan
    placelabs[placelabs==-1] = np.nan
    bodylabs[bodylabs==-1] = np.nan

    volume_space = roi_def.volume_space
    voxel_mask = roi_def.voxel_mask
    nii_shape = roi_def.nii_shape

    if volume_space:
        print('Data is in 3d volume space')
        xfmname = 'func1pt8_to_anat0pt8_autoFSbbr'
        mask_3d = np.reshape(voxel_mask, nii_shape, order='C')

        dat2plot = {'ROI labels (retinotopic)': \
                    cortex.Volume(data=get_full_volume(retlabs, voxel_mask, nii_shape),\
                    subject=substr, cmap='Accent',\
                    vmin = 0, vmax = np.max(retlabs[~np.isnan(retlabs)])+1,\
                    xfmname=xfmname, mask=mask_3d), \
                'ROI labels (face-selective)': \
                    cortex.Volume(data=get_full_volume(facelabs, voxel_mask, nii_shape),\
                    subject=substr, cmap='Accent',\
                    vmin = 0, vmax = np.max(facelabs[~np.isnan(facelabs)])+1, \
                    xfmname=xfmname, mask=mask_3d), \
                'ROI labels (place-selective)': \
                    cortex.Volume(data=get_full_volume(placelabs, voxel_mask, nii_shape),\
                    subject=substr, cmap='Accent',\
                    vmin = 0, vmax = np.max(placelabs[~np.isnan(placelabs)])+1, \
                    xfmname=xfmname, mask=mask_3d), \
                'ROI labels (body-selective)': \
                    cortex.Volume(data=get_full_volume(bodylabs, voxel_mask, nii_shape),\
                    subject=substr, cmap='Accent',\
                    vmin = 0, vmax = np.max(bodylabs[~np.isnan(bodylabs)])+1, \
                     xfmname=xfmname, mask=mask_3d)}

    else:
        print('Data is in nativesurface space')

        dat2plot = {'ROI labels (retinotopic)': \
                    cortex.Vertex(data = get_full_surface(retlabs, voxel_mask), \
                    subject=substr, cmap='Accent',\
                    vmin = 0, vmax = np.max(retlabs)+1), \
                'ROI labels (face-selective)': \
                    cortex.Vertex(data = get_full_surface(facelabs, voxel_mask), \
                    subject=substr, cmap='Accent',\
                    vmin = 0, vmax = np.max(facelabs)+1), \
                'ROI labels (place-selective)': \
                    cortex.Vertex(data = get_full_surface(placelabs, voxel_mask), \
                    subject=substr, cmap='Accent',\
                    vmin = 0, vmax = np.max(placelabs)+1), \
                'ROI labels (body-selective)': \
                    cortex.Vertex(data = get_full_surface(bodylabs, voxel_mask), \
                    subject=substr, cmap='Accent',\
                    vmin = 0, vmax = np.max(bodylabs)+1)}
        
    return dat2plot


def get_full_surface(values, voxel_mask):
    """
    For PyCortex: Put values for voxels that were analyzed back into their 
    correct coordinates in full surface space matrix.
    """
    full_vals = copy.deepcopy(voxel_mask).astype('float64')
    full_vals[voxel_mask==0] = np.nan
    full_vals[voxel_mask==1] = values
    
    return full_vals

def get_full_volume(values, voxel_mask, shape):
    """
    For PyCortex: Put values for voxels that were analyzed back into their 
    correct coordinates in full volume space matrix.
    """
    voxel_mask_3d = np.reshape(voxel_mask, shape)
    full_vals = copy.deepcopy(voxel_mask_3d).astype('float64')
    full_vals[voxel_mask_3d==0] = np.nan
    full_vals[voxel_mask_3d==1] = values
    
    full_vals = np.moveaxis(full_vals, [0,1,2], [2,1,0])
    
    return full_vals
    