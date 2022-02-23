import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import copy

def set_all_font_sizes(fs):
    
    plt.rc('font', size=fs)          # controls default text sizes
    plt.rc('axes', titlesize=fs)     # fontsize of the axes title
    plt.rc('axes', labelsize=fs)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=fs)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=fs)    # fontsize of the tick labels
    plt.rc('legend', fontsize=fs)    # legend fontsize
    plt.rc('figure', titlesize=fs)  # fontsize of the figure title

def create_roi_subplots(data, inds2use, single_plot_object, roi_def, \
                        group_color_inds=None, skip_inds=None, suptitle=None, \
                        label_just_corner=True, figsize=None):

    """
    Create a grid of subplots for each ROI in our dataset.
    
    data: array [n_voxels x n_features]
    inds2use: mask for which of n_voxels to use (based on e.g. R2 threshold)
    single_plot_object: either a scatter_plot, violin_plot, or bar_plot (see below)
    roi_def: roi definition object, from roi_utils.nsd_roi_def()
    group_color_inds: for scatter plots only, do you want to assign different colors to 
        subsets of the points? For example for different subjects.
    skip_inds: which of the n_rois in roi_def do you want to skip?
    suptitle: optional string for a sup-title
    label_just_corner: boolean, want to add axis labels just to the lower left plot?
    figsize: optional figure size for entire plot
    
    """

    n_rois = roi_def.n_rois
    roi_names = roi_def.roi_names
   
    if skip_inds is None:
        skip_inds = []
   
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
            
            inds_this_roi = roi_def.get_indices(rr)

            data_this_roi = data[inds2use & inds_this_roi,:]
            if group_color_inds is not None:
                group_color_inds_this_roi = group_color_inds[inds2use & inds_this_roi]
            else:   
                group_color_inds_this_roi = None
                
            pi = pi+1
            plt.subplot(npx, npy, pi)

            single_plot_object.title = '%s (%d vox)'%(roi_names[rr], data_this_roi.shape[0])
            if pi in pi2label:
                minimal_labels=False
            else:
                minimal_labels=True
            single_plot_object.create(data_this_roi, new_fig=False, minimal_labels=minimal_labels, \
                                      group_color_inds=group_color_inds_this_roi)

    if suptitle is not None:
        plt.suptitle(suptitle)
        
        
class bar_plot:
       
    """
    A bar plot object that plots the means of each column in a data matrix.
    
    Can be passed to "create_roi_subplots" to draw many barplots.
    
    colors: optional array of colors, [n_bars x 3]
    column_labels: optional list of strings, [n_bars]
    plot_errorbars: boolean, want to add SEM errorbars to your plot?
    ylabel: optional, string for yaxis label
    yticks: optional, list of y-axis ticks to use
    ylims: optional, tuple of y-lims
    title: optional title for the barplot
    horizontal_line_pos: optional, position to draw a horizontal line on the plot    
    plot_counts: boolean, are we going to plot a histogram? if yes, must specify "groups"
    groups: list of groups to do counts in, if plot_counts=True
    
    Use "create" method to pass in data and make the plot.
    data: [n_samples x n_bars], the bar heights will be the mean of each column. 
    new_fig: boolean, if True create a new figure, if False assume we already have a figure open.
    figsize: optional figure size
    minimal_labels: boolean, if True the plot will not have xticks/yticks/xlims/ylims
    
    """
    
    def __init__(self, colors=None, column_labels=None, plot_errorbars=True, \
                 ylabel=None, yticks=None, ylims=None, title=None, \
                 horizontal_line_pos=None, plot_counts=False, groups=None):

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
        
    def create(self, data, new_fig=True, figsize=None, minimal_labels=False, **kwargs):
    
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
                plt.xticks(ticks=np.arange(0,n_columns),labels=self.column_labels,\
                           rotation=45, ha='right',rotation_mode='anchor')
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
       
    """
    A scatter plot object that plots the relationship between two columns of data.
    
    Can be passed to "create_roi_subplots" to draw many scatterplots.
    
    colors: optional color for the dots
    xlabel/ylabel: optional, string for x-axis/y-axis label
    xticks/yticks: optional, list of x-axis/y-axis ticks to use
    xlims/ylims: optional, tuple of x-lims/y-lims   
    title: optional title for the plot
    show_diagonal: boolean, do you want to add a diagonal line with slope 1/intercept 0?
    show_axes: boolean, do you want to draw horizontal and vertical lines at x=0, y=0?
    square: boolean, do you want to make the axes square?
    
    Use "create" method to pass in data and make the plot.
    data: [n_samples x 2], or [xvals, yvals] 
    group_color_inds: [n_samples,], provides index into self.colors for each point.
    new_fig: boolean, if True create a new figure, if False assume we already have a figure open.
    figsize: optional figure size
    minimal_labels: boolean, if True the plot will not have xticks/yticks/xlims/ylims
    
    """
        
    def __init__(self, color=None, xlabel=None, ylabel=None, xlims=None, \
                 ylims=None, xticks=None, yticks=None, \
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
        
    def create(self, data, new_fig=True, \
               figsize=None, minimal_labels=False, **kwargs):
    
        if new_fig:
            if figsize is None:
                figsize=(10,10)
            plt.figure(figsize=figsize)
            
        group_color_inds = kwargs['group_color_inds'] if 'group_color_inds' in kwargs.keys() else None
        if group_color_inds is not None:
            group_color_inds = copy.deepcopy(group_color_inds).astype('int')
            assert(np.all(group_color_inds>=0))
            n_color_groups = np.max(group_color_inds)+1
        else:
            group_color_inds = np.zeros((data.shape[0],))
            n_color_groups = 1
            
        if self.color is None:
            if group_color_inds is not None:
                color = cm.tab10(np.linspace(0,1,n_color_groups))
            else:
                color = [0.29803922, 0.44705882, 0.69019608, 1]
        else:
            # if i specify n colors and there are fewer groups represented here, 
            # this is ok (will keep colors consistent even if a subj is missing)
            assert(self.color.shape[0]>=n_color_groups)
            color = self.color
            
        for gg in range(n_color_groups):
            inds = (group_color_inds==gg);
            plt.plot(data[inds,0], data[inds,1],'.',color=color[gg,:])
                                
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
       
    """
    A violin plot object that plots the distribution of each column in a data matrix.
    
    Can be passed to "create_roi_subplots" to draw many violin plots.
    
    colors: optional array of colors, [n_violins x 3]
    column_labels: optional list of strings, [n_violins]
    ylabel: optional, string for yaxis label
    yticks: optional, list of y-axis ticks to use
    ylims: optional, tuple of y-lims
    title: optional title for the violin plot
    horizontal_line_pos: optional, position to draw a horizontal line on the plot    
    
    Use "create" method to pass in data and make the plot.
    data: [n_samples x n_violins], each violin will be one column.
    new_fig: boolean, if True create a new figure, if False assume we already have a figure open.
    figsize: optional figure size
    minimal_labels: boolean, if True the plot will not have xticks/yticks/xlims/ylims
    
    """
        
    def __init__(self, colors=None, column_labels=None, ylabel=None, yticks=None, \
                 title=None, horizontal_line_pos=None, ylims=None):
        
        self.colors = colors
        self.column_labels = column_labels
        self.ylabel = ylabel
        self.title = title
        self.horizontal_line_pos = horizontal_line_pos
        self.ylims = ylims
        self.yticks = yticks
        
    def create(self, data, new_fig=True, figsize=None, minimal_labels=False, **kwargs):
    
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
                plt.xticks(ticks=np.arange(0,n_columns),labels=self.column_labels,\
                           rotation=45, ha='right',rotation_mode='anchor')
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
            
            
def plot_multi_bars(mean_data, err_data=None, colors=None, space=0.3, \
                    xticklabels=None, ylabel=None, ylim=None, \
                    horizontal_line_pos=None,
                    title=None, legend_labels=None, \
                    legend_overlaid=False, legend_separate=True, \
                    fig_size=(12,6)):
    
    """
    Create a bar plot with multiple series of data next to each other.
    Plots error bars if desired.
    
    mean_data: heights of bars to plot; shape [nlevels1 x nlevels2] 
        where nlevels1 is the number of clusters of bars, and nlevels2 
        is the number of bars per cluster.
    err_data: symmetrical error bar lengths, should be same size as mean_data.
    colors: list of colors, [nlevels2 x 3]
    space: how big of a space between each bar cluster? max is 0.45.
    xticklabels: name for each bar "cluster", should be [nlevels1] in length.
    ylabel: optional yaxis label
    horizontal_line_pos: optional position to draw a horizontal line on plot.
    ylim: optional yaxis limits
    title: optional title
    legend_labels: labels for each series in the plot, [nlevels2] in length
    legend_overlaid: want a legend drawn on top of the plot?
    legend_separate: want legend as a separate axis?
    
    """
    assert(space<0.45 and space>0)
    assert(len(mean_data.shape)==2)
    nlevels1, nlevels2 = mean_data.shape
    
    offsets = np.linspace(-0.5+space, 0.5-space, nlevels2)
    bw = np.min([(offsets[1] - offsets[0]), space*2])
    if colors is None:
        colors = cm.Dark2(np.linspace(0,1,nlevels2))
        
    plt.figure(figsize=fig_size)
    
    lh = []
    for ll in range(nlevels2):
        
        h = plt.bar(np.arange(nlevels1)+offsets[ll], mean_data[:,ll], \
                width=bw, color=colors[ll,:])
        lh.append(h)
        if err_data is not None:
            plt.errorbar(np.arange(nlevels1)+offsets[ll], \
                     mean_data[:,ll], err_data[:,ll], \
                     ecolor='k',zorder=10, ls='none')
    
    if xticklabels is not None:                                          
        plt.xticks(np.arange(nlevels1), xticklabels, \
               rotation=45,ha='right',rotation_mode='anchor')
    if ylim is not None:
        plt.ylim(ylim)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if horizontal_line_pos is not None:
        plt.axhline(horizontal_line_pos, color=[0.8, 0.8, 0.8])
    if title is not None:
        plt.title(title)
        
    if legend_overlaid and legend_labels is not None:
        ax = plt.gca()
        ax.legend(lh, legend_labels)
 
    if legend_separate and legend_labels is not None:
        plt.figure();
        for ll in range(nlevels2):
            plt.plot(0,ll,'-',color=colors[ll,:],linewidth=15)
        plt.legend(legend_labels)
    