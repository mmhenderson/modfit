import numpy as np
import copy
import cortex

def plot_maps_pycortex(subject, port, maps, names, subject_map_inds=None, \
                        mins=None, maxes=None, cmaps=None, \
                        title=None, vox2plot = None, roi_def=None,  \
                        volume_space=None, nii_shape=None, voxel_mask=None, 
                        simplest_roi_maps=False):

    """
    Plot a set of maps in pycortex surface space, using cortex.webshow()
    
    subject: nsd subject numbers, 1-8  (can be list or single number)  
    port: port number to use for cortex.webshow()
    maps: list of arrays, each size [n_voxels,] to plot on surface
    names: list of strings, names of the maps
    subject_map_inds: list of which maps in "maps" correspond to each subject 
        (gives index into "subject"). only required if >1 subject.
    mins: optional list of minimum values for each map's colorbar
    maxes: optional list of max values for each map's colorbar
    cmaps: optional list of colormap names, i.e. 'RdBu'
    title: optional string describing the analysis, goes in browser tab header
    voxel_mask: boolean mask of size prod(nii_shape), indicates which voxels in the
        whole brain the n_voxels in each map correspond to.
        (if >1 subject, should be a list n subjects long)
    roi_def: optional, an roi definition object from roi_utils.nsd_roi_def()
        Note roi_def includes fields for voxel_mask, nii_shape, and volume_space, 
        so if roi_def is specified, no need to specify those other arguments.
        If roi_def is not specified, must specify voxel_mask and nii_shape.
        If it is is specified, this code will also plot ROI masks as additional
        maps in the webviewer.
    vox2plot: optional mask for which voxels to include, size [n_voxels,]
        (if >1 subject, should be a list n subjects long)   
    nii_shape: shape of the original nifti file 
        (if >1 subject, should be a list n subjects long)
    volume_space: is the data in volume space (from a 3D nifti file?)
        otherwise assume it is in surface space (i.e. each voxel is a mesh vertex)
    simplest_roi_maps: want to draw all ROIs on one map, color coded only by "type"?
        (i.e. V1-V2 are same color, PPA/RSC are same color) 
        Only used if you have specified roi_def.
    """
    
    if not hasattr(subject, '__len__'):
        subject = [subject]
        voxel_mask = [voxel_mask]
        nii_shape = [nii_shape]
    if len(subject)==1:
        subject_map_inds = np.zeros((len(maps),),dtype=int)
        n_subjects = 1
    else:
        n_subjects = len(subject)
        if len(maps)>0:
            assert(subject_map_inds is not None)
            assert(len(np.unique(subject_map_inds))==n_subjects)
        
    
    if roi_def is not None:
        if n_subjects>1:
            assert(len(roi_def.ss_roi_defs)==len(subject))
            dat2plot = {}
            for si, ss in enumerate(subject):
                dat2plot.update(get_roi_maps_for_pycortex(ss, roi_def.ss_roi_defs[si],\
                                                          simplest_maps=simplest_roi_maps))          
            voxel_mask = roi_def.voxel_mask
            nii_shape = roi_def.nii_shape
        else:
            if hasattr(roi_def, 'subjects'):
                rdef = roi_def.ss_roi_defs[0]
                voxel_mask = roi_def.voxel_mask
                nii_shape = roi_def.nii_shape
            else:
                rdef = roi_def
                voxel_mask = [roi_def.voxel_mask]
                nii_shape = [roi_def.nii_shape]
            dat2plot = get_roi_maps_for_pycortex(subject[0], rdef, simplest_maps=simplest_roi_maps)
            
            
        volume_space = roi_def.volume_space
        
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
        assert(len(voxel_mask)==n_subjects)
        assert(len(nii_shape)==n_subjects)
        mask_3d = [np.reshape(voxel_mask[si], nii_shape[si], order='C') \
                    for si in range(n_subjects)]

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
        
        si = subject_map_inds[ni]
        ss = subject[si]
        substr = 'subj%02d'%ss
  
        map_thresh = copy.deepcopy(maps[ni])
    
        if vox2plot is not None:          
            map_thresh[~vox2plot[si]] = np.nan
        if volume_space:
            dat2plot[name] = cortex.Volume(data = get_full_volume(map_thresh, voxel_mask[si], nii_shape[si]), \
                                           cmap=cmaps[ni], subject=substr, \
                                           vmin=mins[ni], vmax=maxes[ni],\
                                           xfmname=xfmname, mask=mask_3d[si])
        else:
            dat2plot[name] = cortex.Vertex(data = get_full_surface(map_thresh, voxel_mask[si]),\
                                           cmap=cmaps[ni], subject=substr, \
                                           vmin=mins[ni], vmax=maxes[ni])
        
    # Open the webviewer
    print('navigate browser to: 127.0.0.1:%s'%port)
    viewer = cortex.webshow(dat2plot, open_browser=True, port=port, \
                   title = title)
    
#     viewer.get_view('subj01','default_flat')
    
    return viewer

def get_roi_maps_for_pycortex(subject, roi_def, simplest_maps=False):    
    """
    Create a dictionary of cortex.Vertex labels for ROIs, to be plotted in PyCortex.
    
    subject: nsd subject number, 1-8
    roi_def: an roi definition object from roi_utils.nsd_roi_def()
     
    """ 
    substr = 'subj%02d'%subject
    
    retlabs = roi_def.retlabs 
    has_ret = retlabs>-1
    facelabs = roi_def.facelabs 
    has_face = facelabs>-1
    placelabs = roi_def.placelabs
    has_place=placelabs>-1
    bodylabs = roi_def.bodylabs 
    has_body = bodylabs>-1
    
    retlabs[~has_ret] = np.nan
    facelabs[~has_face] = np.nan
    placelabs[~has_place] = np.nan
    bodylabs[~has_body] = np.nan

    volume_space = roi_def.volume_space
    voxel_mask = roi_def.voxel_mask
    nii_shape = roi_def.nii_shape

    if volume_space:
        
        xfmname = 'func1pt8_to_anat0pt8_autoFSbbr'
        mask_3d = np.reshape(voxel_mask, nii_shape, order='C')
        
        if simplest_maps:
            roi_labs = np.nan*np.ones(np.shape(retlabs))
            roi_labs[has_ret] = 0
            roi_labs[has_face] = 1
            roi_labs[has_body] = 2
            roi_labs[has_place] = 3
            overlap = (has_ret.astype(int)+has_face.astype(int)+\
                       has_body.astype(int)+has_place.astype(int))>1
            roi_labs[overlap] = 4;
            dat2plot = {'S%d ROI labels (ret/face/body/place/overlap)'%subject: \
                    cortex.Volume(data=get_full_volume(roi_labs, voxel_mask, nii_shape),\
                    subject=substr, cmap='BROYG',\
                    vmin = 0, vmax = 4,\
                    xfmname=xfmname, mask=mask_3d)}
        else:
            dat2plot = {'S%d ROI labels (retinotopic)'%subject: \
                    cortex.Volume(data=get_full_volume(retlabs, voxel_mask, nii_shape),\
                    subject=substr, cmap='Accent',\
                    vmin = 0, vmax = np.max(retlabs[~np.isnan(retlabs)])+1,\
                    xfmname=xfmname, mask=mask_3d), \
                'S%d ROI labels (face-selective)'%subject: \
                    cortex.Volume(data=get_full_volume(facelabs, voxel_mask, nii_shape),\
                    subject=substr, cmap='Accent',\
                    vmin = 0, vmax = np.max(facelabs[~np.isnan(facelabs)])+1, \
                    xfmname=xfmname, mask=mask_3d), \
                'S%d ROI labels (place-selective)'%subject: \
                    cortex.Volume(data=get_full_volume(placelabs, voxel_mask, nii_shape),\
                    subject=substr, cmap='Accent',\
                    vmin = 0, vmax = np.max(placelabs[~np.isnan(placelabs)])+1, \
                    xfmname=xfmname, mask=mask_3d), \
                'S%d ROI labels (body-selective)'%subject: \
                    cortex.Volume(data=get_full_volume(bodylabs, voxel_mask, nii_shape),\
                    subject=substr, cmap='Accent',\
                    vmin = 0, vmax = np.max(bodylabs[~np.isnan(bodylabs)])+1, \
                     xfmname=xfmname, mask=mask_3d)}

    else:

        dat2plot = {'S%d ROI labels (retinotopic)'%subject: \
                    cortex.Vertex(data = get_full_surface(retlabs, voxel_mask), \
                    subject=substr, cmap='Accent',\
                    vmin = 0, vmax = np.max(retlabs)+1), \
                'S%d ROI labels (face-selective)'%subject: \
                    cortex.Vertex(data = get_full_surface(facelabs, voxel_mask), \
                    subject=substr, cmap='Accent',\
                    vmin = 0, vmax = np.max(facelabs)+1), \
                'S%d ROI labels (place-selective)'%subject: \
                    cortex.Vertex(data = get_full_surface(placelabs, voxel_mask), \
                    subject=substr, cmap='Accent',\
                    vmin = 0, vmax = np.max(placelabs)+1), \
                'S%d ROI labels (body-selective)'%subject: \
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
    