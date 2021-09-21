import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import numpy as np
import copy
import sys
import cortex

from plotting_and_analysis.analysis_utils import get_combined_rois

def set_all_font_sizes(fs):
    
    plt.rc('font', size=fs)          # controls default text sizes
    plt.rc('axes', titlesize=fs)     # fontsize of the axes title
    plt.rc('axes', labelsize=fs)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=fs)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=fs)    # fontsize of the tick labels
    plt.rc('legend', fontsize=fs)    # legend fontsize
    plt.rc('figure', titlesize=fs)  # fontsize of the figure title


def set_plotting_defaults():
    sns.axes_style()
    sns.set_style("white")
    sns.set_context("notebook", rc={'axes.labelsize': 14.0, 'axes.titlesize': 16.0, 'legend.fontsize': 14.0, 'xtick.labelsize': 14.0, 'ytick.labelsize': 14.0})
    sns.set_palette("deep")
    plt.rcParams['image.cmap'] = 'viridis'

def get_full_surface(values, voxel_mask):
    """
    For PyCortex: Put values for voxels that were analyzed back into their correct coordinates in full surface space matrix.
    """
    full_vals = copy.deepcopy(voxel_mask).astype('float64')
    full_vals[voxel_mask==0] = np.nan
    full_vals[voxel_mask==1] = values
    
    return full_vals

def get_full_volume(values, voxel_mask, shape):
    """
    For PyCortex: Put values for voxels that were analyzed back into their correct coordinates in full surface space matrix.
    """
    voxel_mask_3d = np.reshape(voxel_mask, shape)
    full_vals = copy.deepcopy(voxel_mask_3d).astype('float64')
    full_vals[voxel_mask_3d==0] = np.nan
    full_vals[voxel_mask_3d==1] = values
    
    full_vals = np.moveaxis(full_vals, [0,1,2], [2,1,0])
    
    return full_vals


def plot_maps_pycortex(maps, names, subject, out, fitting_type, port, mins=None, maxes=None, cmaps=None):

    """
    Plot some maps in pycortex with desired specifications. Also add roi plots as additional layers.
    """
    
    retlabs, catlabs, ret_group_names, categ_group_names = get_combined_rois(subject, out)
    substr = 'subj%02d'%subject

    voxel_mask = out['voxel_mask']
    
    if out['brain_nii_shape'] is not None:
        print('Data is in 3d volume space')
        xfmname = 'func1pt8_to_anat0pt8_autoFSbbr'
        nii_shape = out['brain_nii_shape']
        mask_3d = np.reshape(voxel_mask, nii_shape, order='C')
        
        dat2plot = {'ROI labels (retinotopic)': cortex.Volume(data=get_full_volume(retlabs, voxel_mask, \
                                                                                   nii_shape),\
                                              subject=substr, cmap='Accent',vmin = 0, vmax = np.max(retlabs),\
                                                              xfmname=xfmname, mask=mask_3d), \
            'ROI labels (category-selective)': cortex.Volume(data=get_full_volume(catlabs, voxel_mask, nii_shape),\
                                             subject=substr, cmap='Accent',vmin = 0, vmax = np.max(catlabs), \
                                                             xfmname=xfmname, mask=mask_3d)}
       
    else:
        print('Data is in nativesurface space')

        dat2plot = {'ROI labels (retinotopic)': cortex.Vertex(data = get_full_surface(retlabs, voxel_mask), \
                                                    subject=substr, cmap='Accent',vmin = 0, vmax = np.max(retlabs)), \
                'ROI labels (category-selective)': cortex.Vertex(data = get_full_surface(catlabs, voxel_mask), \
                                                    subject=substr, cmap='Accent',vmin = 0, vmax = np.max(catlabs))}
                
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
        if out['brain_nii_shape'] is not None:
            dat2plot[name] = cortex.Volume(data = get_full_volume(maps[ni], voxel_mask, nii_shape), \
                                                    cmap=cmaps[ni], subject=substr, vmin=mins[ni], vmax=maxes[ni],\
                                           xfmname=xfmname, mask=mask_3d)
        else:
            dat2plot[name] = cortex.Vertex(data = get_full_surface(maps[ni], voxel_mask), \
                                                    cmap=cmaps[ni], subject=substr, vmin=mins[ni], vmax=maxes[ni])
            
    # Open the webviewer
    print('navigate browser to: 127.0.0.1:%s'%port)
    overlay_file = 'overlays.svg'
    cortex.webshow(dat2plot, open_browser=True, port=port, title = 'S%02d, %s'%(subject, fitting_type), overlay_file=overlay_file, \
                  overlays_visible=('rois', 'sulci'))
    

    
def create_roi_subplots(data, inds2use, single_plot_object, subject, out, suptitle=None, label_just_corner=True, figsize=None):

    """
    Create a grid of subplots for each ROI in the dataset.
    Takes in a 'single plot object' which is either scatter plot, violin plot, or bar plot, with some parameters.
    """
    
    retlabs, catlabs, ret_group_names, categ_group_names = get_combined_rois(subject, out)
    n_rois_ret = len(ret_group_names)
    n_rois = len(ret_group_names) + len(categ_group_names)
    
    # Preferred feature type, based on unique var explained. Separate plot each ROI.
    if figsize==None:
        figsize=(24,20)
    plt.figure(figsize=figsize)
    npy = int(np.ceil(np.sqrt(n_rois)))
    npx = int(np.ceil(n_rois/npy))

    if label_just_corner:
        rr2label = [n_rois-npx-1]
    else:
        rr2label = np.arange(n_rois)
        
    for rr in range(n_rois):

        if rr<n_rois_ret:
            inds_this_roi = retlabs==(rr+1)
            rname = ret_group_names[rr]
        else:
            inds_this_roi = catlabs==(rr+1-n_rois_ret)
            rname = categ_group_names[rr-n_rois_ret]

        data_this_roi = data[inds2use & inds_this_roi,:]

        plt.subplot(npx, npy, rr+1)
        
        single_plot_object.title = '%s (%d vox)'%(rname, data_this_roi.shape[0])
        if rr in rr2label:
            minimal_labels=False
        else:
            minimal_labels=True
        single_plot_object.create(data_this_roi, new_fig=False, minimal_labels=minimal_labels)
            
    if suptitle is not None:
        plt.suptitle(suptitle)
        
        
class bar_plot:
       
    def __init__(self, colors=None, column_labels=None, plot_errorbars=True, ylabel=None, yticks=None, title=None, horizontal_line_pos=None, ylims=None):

        self.colors = colors
        self.column_labels = column_labels
        self.plot_errorbars=plot_errorbars
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
                 title=None, show_diagonal=True, show_axes=True):

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
        plt.axis('square')
        
        if self.show_diagonal:
            plt.plot(self.xlims, self.ylims, color=[0.8, 0.8, 0.8])
        if self.show_axes:
            plt.axvline(color=[0.8, 0.8, 0.8])
            plt.axhline(color=[0.8, 0.8, 0.8])

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