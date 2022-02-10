import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import copy
import cortex
from utils import roi_utils

def set_all_font_sizes(fs):
    
    plt.rc('font', size=fs)          # controls default text sizes
    plt.rc('axes', titlesize=fs)     # fontsize of the axes title
    plt.rc('axes', labelsize=fs)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=fs)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=fs)    # fontsize of the tick labels
    plt.rc('legend', fontsize=fs)    # legend fontsize
    plt.rc('figure', titlesize=fs)  # fontsize of the figure title

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

def get_roi_maps_for_pycortex(subject, roi_def):    
    """
    Create a dictionary of cortex.Vertex labels for ROIs, to be plotted in PyCortex.
    roi_def is from roi_utils.nsd_roi_def()
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

    
def plot_maps_pycortex(subject, title, port, maps, names, roi_def=None, \
                       mins=None, maxes=None, cmaps=None, vox2plot = None, \
                      volume_space=None, nii_shape=None, voxel_mask=None):

    """
    Plot a set of maps in PyCortex surface space.
    Need to either specify a voxel_mask (which voxels in whole brain do your maps 
    correspond to?) or include roi_def (an object from roi_utils.nsd_roi_def that includes
    info about all the voxels/rois).
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
    

    
def create_roi_subplots(data, inds2use, single_plot_object, subject, out, roi_def=None, skip_inds=None, \
                        suptitle=None, label_just_corner=True, figsize=None):

    """
    Create a grid of subplots for each ROI in the dataset.
    Takes in a 'single plot object' which is either scatter plot, violin plot, or bar plot, with some parameters.
    """

    if roi_def is None:
        roi_def = roi_utils.get_combined_rois(subject,verbose=False)    
    retlabs, facelabs, placelabs, bodylabs, ret_names, face_names, place_names, body_names = roi_def    

    if skip_inds is None:
        skip_inds = []
    nret = len(ret_names)
    nface = len(face_names)
    nplace = len(place_names)
    nbody = len(body_names)    
    n_rois = len(ret_names) + len(face_names) + len(place_names) + len(body_names)
    
    is_ret = np.arange(0, n_rois)<nret
    is_face = (np.arange(0, n_rois)>=nret) & (np.arange(0, n_rois)<nret+nface)
    is_place = (np.arange(0, n_rois)>=nret+nface) & (np.arange(0, n_rois)<nret+nface+nplace)
    is_body = np.arange(0, n_rois)>=nret+nface+nplace
    
    # Preferred feature type, based on unique var explained. Separate plot each ROI.
    if figsize==None:
        figsize=(24,20)
    plt.figure(figsize=figsize)
    npy = int(np.ceil(np.sqrt(n_rois-len(skip_inds))))
    npx = int(np.ceil((n_rois-len(skip_inds))/npy))

    if label_just_corner:
        pi2label = [(npx-1)*npy+1]
    else:
        pi2label = np.arange(npx*npy)
        
    pi = 0
    for rr in range(n_rois):

        if rr not in skip_inds:
            
            if is_ret[rr]:
                inds_this_roi = retlabs==rr
                rname = ret_names[rr]
            elif is_face[rr]:
                inds_this_roi = facelabs==(rr-nret)
                rname = face_names[rr-nret]
            elif is_place[rr]:
                inds_this_roi = placelabs==(rr-nret-nface)
                rname = place_names[rr-nret-nface]
            elif is_body[rr]:
                inds_this_roi = bodylabs==(rr-nret-nface-nplace)
                rname = body_names[rr-nret-nface-nplace]
            
            data_this_roi = data[inds2use & inds_this_roi,:]

            pi = pi+1
            plt.subplot(npx, npy, pi)

            single_plot_object.title = '%s (%d vox)'%(rname, data_this_roi.shape[0])
            if pi in pi2label:
                minimal_labels=False
            else:
                minimal_labels=True
            single_plot_object.create(data_this_roi, new_fig=False, minimal_labels=minimal_labels)

    if suptitle is not None:
        plt.suptitle(suptitle)
        
        
class bar_plot:
       
    def __init__(self, colors=None, column_labels=None, plot_errorbars=True, ylabel=None, yticks=None, title=None, horizontal_line_pos=None, ylims=None, plot_counts=False, groups=None):

        self.colors = colors
        self.column_labels = column_labels
        self.plot_errorbars=plot_errorbars
        self.ylabel = ylabel
        self.title = title
        self.horizontal_line_pos = horizontal_line_pos
        self.ylims = ylims
        self.yticks = yticks
        self.plot_counts = plot_counts
        self.groups = groups
        
    def create(self, data, new_fig=True, figsize=None, minimal_labels=False):
    
        if new_fig:
            if figsize is None:
                figsize=(16,8)
            plt.figure(figsize=figsize)
  
        if self.plot_counts:
            data = np.squeeze(data)
            assert(len(data.shape)==1)
            if self.groups is None:
                self.groups = np.unique(data)                
            counts = np.array([np.sum(data==gg) for gg in self.groups])
            counts = counts[np.newaxis, :]
            data = counts
            self.plot_errorbars = False
            
        n_columns = data.shape[1]
          
        if self.colors is None:
            colors = cm.plasma(np.linspace(0,1,n_columns))
            colors = np.flipud(colors)
        else:
            colors = self.colors
            
        if data.shape[0]>0:
        
            for cc in range(n_columns):               
                mean = np.mean(data[:,cc])
                plt.bar(cc, mean, color=colors[cc,:])
                if self.plot_errorbars:
                    sem = np.std(data[:,cc])/np.sqrt(data.shape[0])
                    plt.errorbar(cc, mean, sem, color = colors[cc,:], ecolor='k', zorder=10)

        if not minimal_labels:
            if self.column_labels is not None:
                plt.xticks(ticks=np.arange(0,n_columns),labels=self.column_labels,rotation=45, ha='right',rotation_mode='anchor')
            if self.ylabel is not None:
                plt.ylabel(self.ylabel)
            if self.yticks is not None:
                plt.yticks(self.yticks)
        else:
            plt.yticks([])
            plt.xticks([])
            
        if self.title is not None:
            plt.title(self.title)
        if self.horizontal_line_pos is not None:
            plt.axhline(self.horizontal_line_pos, color=[0.8, 0.8, 0.8])
        if self.ylims is not None:
            plt.ylim(self.ylims)
            
            
class scatter_plot:
       
    def __init__(self, color=None, xlabel=None, ylabel=None, xlims=None, ylims=None, xticks=None, yticks=None, \
                 title=None, show_diagonal=True, show_axes=True, square=True):

        self.color = color
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlims = xlims
        self.ylims = ylims
        self.xticks = xticks
        self.yticks = yticks
        self.title = title
        self.show_diagonal = show_diagonal
        self.show_axes = show_axes   
        self.square = square
        
    def create(self, data, new_fig=True, figsize=None, minimal_labels=False):
    
        if new_fig:
            if figsize is None:
                figsize=(10,10)
            plt.figure(figsize=figsize)

        if self.color is None:
            color = [0.29803922, 0.44705882, 0.69019608, 1]
        else:
            color = self.color
            
        plt.plot(data[:,0], data[:,1],'.',color=color)
        if self.square==True:
            plt.axis('square')
        
        if not minimal_labels:
            if self.xlabel is not None:
                plt.xlabel(self.xlabel)
            if self.ylabel is not None:
                plt.ylabel(self.ylabel)
            if self.xticks is not None:
                plt.xticks(self.xticks)
            if self.yticks is not None:
                plt.yticks(self.yticks)
        else:
            plt.yticks([])
            plt.xticks([])
            
        if self.title is not None:
            plt.title(self.title)
        if self.xlims is not None:
            plt.xlim(self.xlims)
        if self.ylims is not None:
            plt.ylim(self.ylims)
            
        if self.show_diagonal:
            if self.xlims is None:
                xlim = plt.gca().get_xlim()
            else:
                xlim = self.xlims
            if self.ylims is None:
                ylim = plt.gca().get_ylim()
            else:
                ylim = self.ylims
            
            plt.plot(xlim,ylim, color=[0.8, 0.8, 0.8])
        if self.show_axes:
            plt.axvline(color=[0.8, 0.8, 0.8])
            plt.axhline(color=[0.8, 0.8, 0.8])
            
class violin_plot:
       
    def __init__(self, colors=None, column_labels=None, ylabel=None, yticks=None, title=None, horizontal_line_pos=None, ylims=None):
        
        self.colors = colors
        self.column_labels = column_labels
        self.ylabel = ylabel
        self.title = title
        self.horizontal_line_pos = horizontal_line_pos
        self.ylims = ylims
        self.yticks = yticks
        
    def create(self, data, new_fig=True, figsize=None, minimal_labels=False):
    
        if new_fig:
            if figsize is None:
                figsize=(16,8)
            plt.figure(figsize=figsize)

        n_columns = data.shape[1]
        if self.colors is None:
            colors = cm.plasma(np.linspace(0,1,n_columns))
            colors = np.flipud(colors)
        else:
            colors = self.colors
            
        if data.shape[0]>0:
            for cc in range(n_columns):

                parts = plt.violinplot(data[:,cc], [cc])
                for pc in parts['bodies']:
                    pc.set_color(self.colors[cc,:])
                parts['cbars'].set_color(self.colors[cc,:])
                parts['cmins'].set_color(self.colors[cc,:])
                parts['cmaxes'].set_color(self.colors[cc,:])

        if not minimal_labels:
            if self.column_labels is not None:
                plt.xticks(ticks=np.arange(0,n_columns),labels=self.column_labels,rotation=45, ha='right',rotation_mode='anchor')
            if self.ylabel is not None:
                plt.ylabel(self.ylabel)
            if self.yticks is not None:
                plt.yticks(self.yticks)
        else:
            plt.xticks([])
            plt.yticks([])
            
        if self.title is not None:
            plt.title(self.title)
        if self.horizontal_line_pos is not None:
            plt.axhline(self.horizontal_line_pos, color=[0.8, 0.8, 0.8])
        if self.ylims is not None:
            plt.ylim(self.ylims)