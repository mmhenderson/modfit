import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import seaborn as sns
import numpy as np
import copy
import sys
import os
import cortex
import torch

"""
General use functions for plotting encoding model fit results.
Input to most of these functions is 'out', which is a dictionary containing 
fit results. Created by the model fitting code in model_fitting/fit_model.py
"""

sys.path.append('/user_data/mmhender/imStat/code/')
from utils import roi_utils, nsd_utils


def set_plotting_defaults():
    sns.axes_style()
    sns.set_style("white")
    sns.set_context("notebook", rc={'axes.labelsize': 12.0, 'axes.titlesize': 14.0, 'legend.fontsize': 12.0, 'xtick.labelsize': 12.0, 'ytick.labelsize': 12.0})
    sns.set_palette("deep")
    plt.rcParams['image.cmap'] = 'viridis'
      

def load_fit_results(subject, volume_space, fitting_type, n_from_end, root, verbose=True):
       
    if root is None:
        root = os.path.dirname(os.path.dirname(os.getcwd()))
    if volume_space:
        folder2load = os.path.join(root, 'model_fits','S%02d'%subject, fitting_type)
    else:
        folder2load = os.path.join(root, 'model_fits','S%02d_surface'%subject, fitting_type)
        
    # within this folder, assuming we want the most recent version that was saved
    files_in_dir = os.listdir(folder2load)
    from datetime import datetime
    my_dates = [f for f in files_in_dir if 'ipynb' not in f and 'DEBUG' not in f]
    my_dates.sort(key=lambda date: datetime.strptime(date, "%b-%d-%Y_%H%M_%S"))
    
    # if n from end is not zero, then going back further in time 
    most_recent_date = my_dates[-1-n_from_end]

    subfolder2load = os.path.join(folder2load, most_recent_date)
    file2load = os.listdir(subfolder2load)[0]
    fullfile2load = os.path.join(subfolder2load, file2load)

    if verbose:
        print('loading from %s\n'%fullfile2load)

    out = torch.load(fullfile2load)
    
    if verbose:
        print(out.keys())
        
    fig_save_folder = os.path.join(root,'figures','S%02d'%subject, fitting_type, most_recent_date)
    
    return out, fig_save_folder

def print_output_summary(out):
    """
    Print all the keys in the saved data file and a summary of each value.
    """
    for kk in out.keys():
        if out[kk] is not None:
            if np.isscalar(out[kk]):
                print('%s = %s'%(kk, out[kk]))
            elif isinstance(out[kk],tuple) or isinstance(out[kk],list):
                print('%s: len %s'%(kk, len(out[kk])))
            elif isinstance(out[kk],np.ndarray):
                print('%s: shape %s'%(kk, out[kk].shape))
            elif isinstance:
                print('%s: unknown'%kk)


def get_roi_info(subject, out, verbose=False):
    """
    Gather all information about roi definitions for the analyzed voxels.
    """
    voxel_roi = out['voxel_roi']
    voxel_idx = out['voxel_index'][0]
    
    assert(len(voxel_roi)==2)
    [roi_labels_retino, roi_labels_categ] = copy.deepcopy(voxel_roi)
    roi_labels_retino = roi_labels_retino[voxel_idx]
    roi_labels_categ = roi_labels_categ[voxel_idx]
    
    ret, face, place = roi_utils.load_roi_label_mapping(subject, verbose=verbose)
    
    max_ret_label = np.max(ret[0])
    face[0] = face[0]+max_ret_label
    max_face_label = np.max(face[0])
    place[0] = place[0]+max_face_label
    if verbose:
        print(face)
        print(place)
        print(np.unique(roi_labels_categ))

    ret_group_names = roi_utils.ret_group_names
    ret_group_inds =  roi_utils.ret_group_inds
    n_rois_ret = len(ret_group_names)

    categ_group_names = list(np.concatenate((face[1], place[1])))
    categ_group_inds =  list(np.concatenate((face[0], place[0])))
    n_rois_categ = len(categ_group_names)

    n_rois = n_rois_ret + n_rois_categ
    
    return roi_labels_retino, roi_labels_categ, ret_group_inds, categ_group_inds, ret_group_names, categ_group_names, \
                n_rois_ret, n_rois_categ, n_rois

def get_combined_rois(subject, out):
    
    roi_labels_retino, roi_labels_categ, ret_group_inds, categ_group_inds, ret_group_names, categ_group_names, \
        n_rois_ret, n_rois_categ, n_rois = get_roi_info(subject, out)

    retlabs = np.zeros(np.shape(roi_labels_retino))
    catlabs = np.zeros(np.shape(roi_labels_retino))

    for rr in range(n_rois_ret):   
        inds_this_roi = np.isin(roi_labels_retino, ret_group_inds[rr])
        retlabs[inds_this_roi] = rr+1

    for rr in range(n_rois_categ):   
        inds_this_roi = np.isin(roi_labels_categ, categ_group_inds[rr])
        catlabs[inds_this_roi] = rr+1

    return retlabs, catlabs, ret_group_names, categ_group_names


def get_prf_pars_deg(out, screen_eccen_deg=8.4):
    """
    Convert the saved estimates of prf position/sd into eccentricity, angle, etc in degrees.
    """
    if len(out['best_params'])==7:
        best_models, weights, bias, features_mt, features_st, best_model_inds, _ = out['best_params']
    else:
        best_models, weights, bias, features_mt, features_st, best_model_inds = out['best_params']
    best_models_deg = best_models * screen_eccen_deg
    if len(best_models_deg.shape)==2:
        best_models_deg = np.expand_dims(best_models_deg, axis=1)
    pp=0
    best_ecc_deg  = np.sqrt(np.square(best_models_deg[:,pp,0]) + np.square(best_models_deg[:,pp,1]))
    best_angle_deg  = np.arctan2(best_models_deg[:,pp,1], best_models_deg[:,pp,0])*180/np.pi + 180
    best_size_deg = best_models_deg[:,pp,2]
    
    return best_ecc_deg, best_angle_deg, best_size_deg

def get_r2(out):
    
    val_cc = out['val_cc']
    # Note i'm NOT using the thing that actually is in the field val_r2, 
    # bc that is coefficient of determination which gives poor results for ridge regression.
    # instead using the signed squared correlation coefficient for r2/var explained.
    val_r2 = np.sign(val_cc)*val_cc**2

    return val_r2

def plot_perf_summary(subject, fitting_type,out, fig_save_folder=None):
    """
    Plot some general metrics of fit performance, across all voxels.
    """
    cclims = [-1,1]
#     losslims = [350,850]
    
    best_losses = out['best_losses']
    if len(best_losses.shape)==2:
        best_losses = best_losses[:,0]
    val_cc = out['val_cc'][:,0]
    val_r2 = get_r2(out)[:,0]
    best_lambdas = out['best_lambdas']
    lambdas = out['lambdas']

    
    plt.figure(figsize=(16,8));

    plt.subplot(2,2,1)
    plt.hist(best_losses,100)
    # plt.xlim(losslims)
    # plt.xlim([500,2000])
    plt.xlabel('loss value/SSE (held-out training)');
    plt.ylabel('number of voxels');

    plt.subplot(2,2,2)
    plt.hist(val_cc,100)
    # plt.hist(val_cc,100)
    # plt.xlim([-0.2, 0.8])
    plt.xlim(cclims)
    plt.xlabel('correlation coefficient r (validation)');
    plt.ylabel('number of voxels');
    plt.axvline(0,color='k')

    plt.subplot(2,2,3)

    # plt.hist(np.sign(val_cc)*val_cc**2,100)
    plt.hist(val_r2,100)
    # plt.xlim([-0.2, 0.8])
    plt.xlabel('r2 (validation)');
    plt.ylabel('number of voxels');
    plt.axvline(0,color='k')

    plt.subplot(2,2,4)

    plt.plot(lambdas, [np.sum(best_lambdas==k) for k in range(len(lambdas))], lw=4, marker='o', ms=12)
    plt.xscale('log');
    plt.xlabel('lambda value for best fit');
    plt.ylabel('number of voxels');

    plt.suptitle('S%02d, %s'%(subject, fitting_type))
    if fig_save_folder is not None:
        plt.savefig(os.path.join(fig_save_folder,'fit_summary.png'))
        plt.savefig(os.path.join(fig_save_folder,'fit_summary.pdf'))

        

def get_full_surface(values, voxel_mask):
    """
    For PyCortex: Put values for voxels that were analyzed back into their correct coordinates in full surface space matrix.
    """
    full_vals = copy.deepcopy(voxel_mask).astype('float64')
    full_vals[voxel_mask==0] = np.nan
    full_vals[voxel_mask==1] = values
    
    return full_vals

def plot_summary_pycortex(subject, fitting_type, out, port):

    """
    Use pycortex webgl function to plot some summary statistics for encoding model fits, in surface space.
    Plots pRF spatial parameters, and the model's prediction performance on validation set.
    """
    
    if out['brain_nii_shape'] is not None:
        raise ValueError('Cannot use this function for data that is in volume space!')
       
    substr = 'subj%02d'%subject

    retlabs, catlabs, ret_group_names, categ_group_names = get_combined_rois(subject, out)
    best_ecc_deg, best_angle_deg, best_size_deg = get_prf_pars_deg(out, screen_eccen_deg=8.4)
    val_cc = out['val_cc'][:,0]
    val_r2 = get_r2(out)[:,0]
    voxel_mask = out['voxel_mask']
    
    cmin = 0.0
    cmax = 0.8
    rmin = 0.0
    rmax = 0.6
    vemin = -0.05
    vemax = 0.05

    dat2plot = {'ROI labels (retinotopic)': cortex.Vertex(data = get_full_surface(retlabs, voxel_mask), \
                                                subject=substr, cmap='Accent',vmin = 0, vmax = np.max(retlabs)), \
                'ROI labels (category-selective)': cortex.Vertex(data = get_full_surface(catlabs, voxel_mask), \
                                                subject=substr, cmap='Accent',vmin = 0, vmax = np.max(catlabs)), \
                'pRF eccentricity': cortex.Vertex(data = get_full_surface(best_ecc_deg, voxel_mask), subject=substr, \
                                                  cmap='PRGn', vmin=0, vmax=7), \
                'pRF angle': cortex.Vertex(data = get_full_surface(best_angle_deg, voxel_mask), subject=substr, \
                                           cmap='Retinotopy_RYBCR', vmin=0, vmax=360), \
                'pRF size': cortex.Vertex(data = get_full_surface(best_size_deg, voxel_mask), subject=substr, \
                                          cmap='PRGn', vmin=0, vmax=4), \
                'Correlation (validation set)': cortex.Vertex(data = get_full_surface(val_cc, voxel_mask), subject=substr, \
                                                              cmap='PuBu', vmin=cmin, vmax=cmax), \
                'R2 (validation set)': cortex.Vertex(data = get_full_surface(val_r2, voxel_mask), subject=substr, \
                                                     cmap='PuBu', vmin=rmin, vmax=rmax), \
               }

    dat2plot.keys()
    
    # Open the webviewer
    print('navigate browser to: 127.0.0.1:%s'%port)
    cortex.webshow(dat2plot, open_browser=True, port=port, title = 'S%02d, %s'%(subject, fitting_type))
    

def plot_texture_pars_pycortex(subject, fitting_type, out, port):
    """
    Plots fit parameters for texture model, in surface space using pycortex webgl
    """
    if 'texture' not in fitting_type:
        raise ValueError('this plot is just for texture model')        
      

    retlabs, catlabs, ret_group_names, categ_group_names = get_combined_rois(subject, out)

    if out['brain_nii_shape'] is not None:
        raise ValueError('Cannot use this function for data that is in volume space!')

    substr = 'subj%02d'%subject

    val_r2 = get_r2(out)
    # Compute variance explained by each feature type - how well does the model without that feature type
    # do, compared to the model with all features? 
    n_feature_types = len(out['feature_info'][1])
    var_expl = np.tile(np.expand_dims(val_r2[:,0], axis=1), [1,n_feature_types]) - val_r2[:,1:] 

    # which feature type explains most unique variance, for each voxel?
    max_ind = np.argmax(var_expl, axis=1)
    colors = cm.plasma(np.linspace(0,1,n_feature_types))
    colors = np.flipud(colors)

    voxel_mask = out['voxel_mask']

    vemin = -0.05
    vemax = 0.05

    dat2plot = {'ROI labels (retinotopic)': cortex.Vertex(data = get_full_surface(retlabs, voxel_mask), \
                                                    subject=substr, cmap='Accent',vmin = 0, vmax = np.max(retlabs)), \
                'ROI labels (category-selective)': cortex.Vertex(data = get_full_surface(catlabs, voxel_mask), \
                                                    subject=substr, cmap='Accent',vmin = 0, vmax = np.max(catlabs)), \
                'Preferred feature type (based on var expl)': \
                                    cortex.Vertex(data = get_full_surface(max_ind, voxel_mask), \
                                                    subject=substr, cmap='plasma_r', vmin=0, vmax=n_feature_types)}

    for fi, ff in enumerate(out['feature_info'][1]):
        dat2plot['Var expl: %s'%ff] = cortex.Vertex(data = get_full_surface(var_expl[:,fi], voxel_mask), \
                                                    subject=substr, vmin=vemin, vmax=vemax)

    # Open the webviewer
    print('navigate browser to: 127.0.0.1:%s'%port)
    cortex.webshow(dat2plot, open_browser=True, port=port, title = 'S%02d, %s'%(subject, fitting_type))

    
def plot_fit_summary_volume_space(subject, fitting_type, out, screen_eccen_deg = 8.4, fig_save_folder=None):

    """
    Visualize some basic properties of pRFs for each voxel, in volume space
    Should be sanity check for dorsal/visual distinctions, esp w/r/t RF angle estimates
    """ 
    
    if out['brain_nii_shape'] is None:
        raise ValueError('Cannot use this function for data that is in surface space, should use pycortex to visualize instead.')
        
   
    brain_nii_shape = out['brain_nii_shape']
    voxel_idx = out['voxel_index'][0]
    best_losses = out['best_losses']
    if len(best_losses.shape)==2:
        best_losses = best_losses[:,0]
    val_cc = out['val_cc'][:,0]

    best_ecc_deg, best_angle_deg, best_size_deg = get_prf_pars_deg(out, screen_eccen_deg)    

    roi_labels_retino, roi_labels_categ, ret_group_inds, categ_group_inds, ret_group_names, categ_group_names, \
        n_rois_ret, n_rois_categ, n_rois = get_roi_info(subject, out)

    volume_loss = roi_utils.view_data(brain_nii_shape, voxel_idx, best_losses)
    volume_cc   = roi_utils.view_data(brain_nii_shape, voxel_idx, val_cc)
    volume_ecc  = roi_utils.view_data(brain_nii_shape, voxel_idx, best_ecc_deg)
    volume_ang  = roi_utils.view_data(brain_nii_shape, voxel_idx, best_angle_deg)
    volume_size = roi_utils.view_data(brain_nii_shape, voxel_idx, best_size_deg)
    volume_roi = roi_utils.view_data(brain_nii_shape, voxel_idx, roi_labels_retino)

    slice_idx = 40
    fig = plt.figure(figsize=(16,8))
    plt.subplot(2,3,1)
    plt.title('Loss')
    plt.imshow(volume_loss[:,:,slice_idx], cmap='viridis', interpolation='None')
    plt.colorbar()
    plt.subplot(2,3,2)
    plt.title('Validation accuracy')
    plt.imshow(volume_cc[:,:,slice_idx], cmap='viridis', interpolation='None')
    plt.colorbar()
    plt.subplot(2,3,3)
    plt.title('RF Eccentricity')
    plt.imshow(volume_ecc[:,:,slice_idx], cmap='viridis', interpolation='None')
    plt.colorbar()
    plt.subplot(2,3,4)
    plt.title('RF Angle')
    plt.imshow(volume_ang[:,:,slice_idx], cmap='hsv', interpolation='None')
    plt.colorbar()
    plt.subplot(2,3,5)
    plt.title('RF Size')
    plt.imshow(volume_size[:,:,slice_idx], cmap='viridis', interpolation='None')
    plt.colorbar()
    plt.subplot(2,3,6)
    plt.title('ROI labels')
    plt.imshow(volume_roi[:,:,slice_idx], cmap='jet', interpolation='None')
    plt.colorbar()

    plt.suptitle('S%02d, %s'%(subject, fitting_type));
    if fig_save_folder is not None:
        plt.savefig(os.path.join(fig_save_folder,'fit_summary_volumespace.pdf'))
        plt.savefig(os.path.join(fig_save_folder,'fit_summary_volumespace.png'))
        
def plot_noise_ceilings(subject, fitting_type,out, fig_save_folder):

    """
    Plot distribution of noise ceilings and NCSNR across all voxels.
    """
    
    voxel_ncsnr = out['voxel_ncsnr'].ravel()[out['voxel_index'][0]]
    noise_ceiling = nsd_utils.ncsnr_to_nc(voxel_ncsnr)

    plt.figure(figsize=(16,4));

    plt.subplot(1,2,1)
    plt.hist(voxel_ncsnr,100)
    plt.xlabel('NCSNR');
    plt.ylabel('number of voxels');

    plt.subplot(1,2,2)
    plt.hist(noise_ceiling,100)
    plt.xlabel('Noise ceiling (percent) for single trial estimates');
    plt.ylabel('number of voxels');
    plt.axvline(0,color='k')

    plt.suptitle('S%02d, %s'%(subject, fitting_type))
    
    if fig_save_folder is not None:
        plt.savefig(os.path.join(fig_save_folder,'noise_ceiling_dist.png'))
        plt.savefig(os.path.join(fig_save_folder,'noise_ceiling_dist.pdf'))
 
def plot_cc_each_roi(subject, fitting_type,out, fig_save_folder=None):

    """
    Make a histogram for each ROI, showing distibution of validation set correlation coefficient for all voxels.
    """
    
    roi_labels_retino, roi_labels_categ, ret_group_inds, categ_group_inds, ret_group_names, categ_group_names, \
        n_rois_ret, n_rois_categ, n_rois = get_roi_info(subject, out)
    
    val_cc = out['val_cc'][:,0]
    
    plt.figure(figsize=(20,16))
    npx = int(np.ceil(np.sqrt(n_rois)))
    npy = int(np.ceil(n_rois/npx))

    for rr in range(n_rois):

        if rr<n_rois_ret:
            inds_this_roi = np.isin(roi_labels_retino, ret_group_inds[rr])
            rname = ret_group_names[rr]
        else:
            inds_this_roi = np.isin(roi_labels_categ, categ_group_inds[rr-n_rois_ret])
            rname = categ_group_names[rr-n_rois_ret]

        plt.subplot(npx,npy,rr+1)

        h = plt.hist(val_cc[inds_this_roi], bins=np.linspace(-0.2,1,100))

        if rr==n_rois-4:
            plt.xlabel('Correlation coefficient')
            plt.ylabel('Number of voxels')
        else:
            plt.xticks([]);
    #         plt.yticks([])

        plt.axvline(0,color='k')

        plt.title('%s (%d vox)'%(rname, np.sum(inds_this_roi)))

    plt.suptitle('Correlation coef. on validation set\nS%02d, %s'%(subject, fitting_type));

    if fig_save_folder is not None:
        plt.savefig(os.path.join(fig_save_folder,'corr_each_roi.pdf'))
        plt.savefig(os.path.join(fig_save_folder,'corr_each_roi.png'))
        
        
def plot_r2_vs_nc(subject, fitting_type,out, fig_save_folder=None):

    """
    Create scatter plots for each ROI, comparing each voxel's R2 prediction to the noise ceiling.
    """
    
    roi_labels_retino, roi_labels_categ, ret_group_inds, categ_group_inds, ret_group_names, categ_group_names, \
        n_rois_ret, n_rois_categ, n_rois = get_roi_info(subject, out)
        
    voxel_ncsnr = out['voxel_ncsnr'].ravel()[out['voxel_index'][0]]
    noise_ceiling = nsd_utils.ncsnr_to_nc(voxel_ncsnr)

    val_cc = out['val_cc'][:,0]
    val_r2 = get_r2(out)[:,0]
    
    cc_cutoff = -100

    xlims = [-0.1, 0.7]
    ylims = [-0.1, 0.7]
    
    plt.figure(figsize=(24,20))
    npx = int(np.ceil(np.sqrt(n_rois)))
    npy = int(np.ceil(n_rois/npx))
    

    for rr in range(n_rois):
   
        if rr<n_rois_ret:
            inds_this_roi = np.isin(roi_labels_retino, ret_group_inds[rr])
            rname = ret_group_names[rr]
        else:
            inds_this_roi = np.isin(roi_labels_categ, categ_group_inds[rr-n_rois_ret])
            rname = categ_group_names[rr-n_rois_ret]

        abv_thresh = val_cc>cc_cutoff
        inds2use = np.logical_and(inds_this_roi, abv_thresh)

        plt.subplot(npx,npy,rr+1)

        if np.sum(inds2use)>0:

            xvals = noise_ceiling[inds2use]/100
            yvals = val_r2[inds2use]

            plt.plot(xvals,yvals,'.')

        if rr==n_rois-4:
            plt.xlabel('Noise ceiling')
            plt.ylabel('r2 for full model')
        else:
            plt.xticks([])
        plt.axis('square')

        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.plot(xlims, ylims, color=[0.8, 0.8, 0.8])
        plt.axvline(color=[0.8, 0.8, 0.8])
        plt.axhline(color=[0.8, 0.8, 0.8])


        plt.title('%s (%d vox)'%(rname, np.sum(inds2use)))

    plt.suptitle('S%02d, %s\nComparing model performance to noise ceiling'%(subject, fitting_type))

    if fig_save_folder is not None:
        plt.savefig(os.path.join(fig_save_folder,'r2_vs_noiseceiling.png'))
        plt.savefig(os.path.join(fig_save_folder,'r2_vs_noiseceiling.pdf'))
        
##### SPATIAL RECEPTIVE FIELD PLOTS #####################        

def plot_spatial_rf_circles(subject, fitting_type, out, cc_cutoff = 0.20, screen_eccen_deg = 8.4, fig_save_folder = None):

    """
    Make a plot for each ROI showing the visual field coverage of pRFs in that ROI: 
    each circle is a voxel with the size of the circle indicating the pRF's size (1 SD)
    """

    pp=0
    best_models_deg = out['best_params'][0] * screen_eccen_deg
    if len(best_models_deg.shape)==2:
        best_models_deg = np.expand_dims(best_models_deg, axis=1)
        
#     best_ecc_deg, best_angle_deg, best_size_deg = get_prf_pars_deg(out, screen_eccen_deg)    

    roi_labels_retino, roi_labels_categ, ret_group_inds, categ_group_inds, ret_group_names, categ_group_names, \
        n_rois_ret, n_rois_categ, n_rois = get_roi_info(subject, out)
    
    val_cc = out['val_cc'][:,0]

    plt.figure(figsize=(24,18))

    npy = int(np.ceil(np.sqrt(n_rois)))
    npx = int(np.ceil(n_rois/npy))

    for rr in range(n_rois):

        if rr<n_rois_ret:
            inds_this_roi = np.isin(roi_labels_retino, ret_group_inds[rr])
            rname = ret_group_names[rr]
        else:
            inds_this_roi = np.isin(roi_labels_categ, categ_group_inds[rr-n_rois_ret])
            rname = categ_group_names[rr-n_rois_ret]

        abv_thresh = val_cc>cc_cutoff
        inds2use = np.where(np.logical_and(inds_this_roi, abv_thresh))[0]

        plt.subplot(npx,npy,rr+1)
        ax = plt.gca()

        for vi, vidx in enumerate(inds2use):

            plt.plot(best_models_deg[vidx,pp,0], best_models_deg[vidx,pp,1],'.',color='k')
            circ = matplotlib.patches.Circle((best_models_deg[vidx,pp,0], best_models_deg[vidx,pp,1]), best_models_deg[vidx,pp,2], 
                                             color = [0.8, 0.8, 0.8], fill=False)
            ax.add_artist(circ)

        plt.axis('square')

        plt.xlim([-screen_eccen_deg, screen_eccen_deg])
        plt.ylim([-screen_eccen_deg, screen_eccen_deg])
        plt.xticks(np.arange(-8,9,4))
        plt.yticks(np.arange(-8,9,4))
        if rr==n_rois-5:
            plt.xlabel('x coord (deg)')
            plt.ylabel('y coord (deg)')
        plt.title('%s (%d vox)'%(rname, len(inds2use)))

    plt.suptitle('pRF estimates\nshowing all voxels with corr > %.2f\nS%02d, %s'%(cc_cutoff, subject, fitting_type));

    if fig_save_folder is not None:
        plt.savefig(os.path.join(fig_save_folder,'spatial_prf_distrib.pdf'))
        plt.savefig(os.path.join(fig_save_folder,'spatial_prf_distrib.png'))


def plot_size_vs_eccen(subject, fitting_type,out, cc_cutoff=0.2, screen_eccen_deg = 8.4, fig_save_folder=None ):
    """
    Create a scatter plot for each ROI, showing the size of each voxel's best pRF estimate versus its eccentricity.
    """

    size_lims = screen_eccen_deg*np.array([0, 0.5])
    eccen_lims = [-1, screen_eccen_deg]
    
    best_ecc_deg, best_angle_deg, best_size_deg = get_prf_pars_deg(out, screen_eccen_deg)
    
    roi_labels_retino, roi_labels_categ, ret_group_inds, categ_group_inds, ret_group_names, categ_group_names, \
        n_rois_ret, n_rois_categ, n_rois = get_roi_info(subject, out)
    
    val_cc = out['val_cc'][:,0]
    
    npx = int(np.ceil(np.sqrt(n_rois)))
    npy = int(np.ceil(n_rois/npx))
    
    plt.figure(figsize=(24,20))

    for rr in range(n_rois):

        if rr<n_rois_ret:
            inds_this_roi = np.isin(roi_labels_retino, ret_group_inds[rr])
            rname = ret_group_names[rr]
        else:
            inds_this_roi = np.isin(roi_labels_categ, categ_group_inds[rr-n_rois_ret])
            rname = categ_group_names[rr-n_rois_ret]

        abv_thresh = val_cc>cc_cutoff
        inds2use = np.where(np.logical_and(inds_this_roi, abv_thresh))[0]

        plt.subplot(npx,npy,rr+1)
        ax = plt.gca()

        plt.plot(best_ecc_deg[inds2use], best_size_deg[inds2use], '.')

        plt.xlim(eccen_lims)
        plt.ylim(size_lims)
        if rr==n_rois-4:
            plt.xlabel('eccen (deg)')
            plt.ylabel('size (deg)')

        plt.title('%s (%d vox)'%(rname, len(inds2use)))

    plt.suptitle('pRF estimates\nshowing all voxels with corr > %.2f\nS%02d, %s'%(cc_cutoff, subject, fitting_type))
    
    if fig_save_folder is not None:
        plt.savefig(os.path.join(fig_save_folder,'size_vs_eccen.png'))
        plt.savefig(os.path.join(fig_save_folder,'size_vs_eccen.pdf'))
        
        

def plot_prf_stability_partial_versions(subject, out, cc_cutoff = 0.2, screen_eccen_deg = 8.4, fig_save_folder = None):
    
    plt.figure(figsize=(24,18));

    best_models_partial_deg = out['best_params'][0]*screen_eccen_deg
    n_partial_models = best_models_partial_deg.shape[1]
    val_cc = out['val_cc'][:,0]
    abv_thresh = val_cc>cc_cutoff   

    vox2plot = np.argsort(val_cc)[-20:-1]
    colors = cm.hsv(np.linspace(0,1,len(vox2plot)+1))

    for pp in range(n_partial_models):

        plt.subplot(4,4,pp+1)
        ax = plt.gca()

        for vi, vidx in enumerate(vox2plot):
    #         if vi>1: 
    #             break

            plt.plot(best_models_partial_deg[vidx,pp,0], best_models_partial_deg[vidx,pp,1],'.',color='k')
            circ = matplotlib.patches.Circle((best_models_partial_deg[vidx,pp,0], best_models_partial_deg[vidx,pp,1]), \
                                             best_models_partial_deg[vidx,pp,2], color = colors[vi,:], fill=False)
            ax.add_artist(circ)

        plt.axis('square')

        plt.xlim([-screen_eccen_deg, screen_eccen_deg])
        plt.ylim([-screen_eccen_deg, screen_eccen_deg])
        plt.xticks(np.arange(-8,9,4))
        plt.yticks(np.arange(-8,9,4))
        if pp==n_partial_models-4:
            plt.xlabel('x coord (deg)')
            plt.ylabel('y coord (deg)')

        plt.title('partial model version %d'%pp)

    # plt.suptitle('X coordinate of pRF fits')
    # plt.suptitle('Y coordinate of pRF fits')
    plt.suptitle('Stability of pRF fits for various versions of model (holding out sets of features)\nBest 20 voxels')

    if fig_save_folder:
        plt.savefig(os.path.join(fig_save_folder,'prf_stability_holdout.pdf'))
        plt.savefig(os.path.join(fig_save_folder,'prf_stability_holdout.png'))
        
##### TEXTURE MODEL PARAMETER PLOTS #####################                

def plot_example_weights_texture(subject, fitting_type, out, fig_save_folder):

    """
    Plotting one example voxel weights, to get an idea of how they are distributed
    """
    
    if 'texture' not in fitting_type:
        raise ValueError('this plot is just for texture model')        
        
    roi_labels_retino, roi_labels_categ, ret_group_inds, categ_group_inds, ret_group_names, categ_group_names, \
        n_rois_ret, n_rois_categ, n_rois = get_roi_info(subject, out)
    
    
    feature_info = copy.deepcopy(out['feature_info'])
    feature_type_labels, feature_type_names = feature_info
    n_feature_types = len(feature_type_names)

    val_cc = out['val_cc'][:,0]
    lambdas = out['lambdas']
    best_lambdas = out['best_lambdas']
    if len(best_lambdas.shape)==2:
        best_lambdas = best_lambdas[:,0]
    
    vox2plot = np.argsort(np.nan_to_num(val_cc))[-1] # choosing vox w best validation set performance
    vv=vox2plot
    
    colors = cm.plasma(np.linspace(0,1,n_feature_types))
    colors = np.flipud(colors)

    
    plt.figure(figsize=(22,8))

    weights = out['best_params'][1]
    n_features_total = np.shape(weights)[1]
    pp=0
    wts = weights[vv,:,pp]
    plt.plot(wts,'-',color='k')

    lh=[]
    f_count=0
    for ft in range(n_feature_types):    
        f_count = f_count+np.sum(feature_type_labels==ft)
        plt.axvline(f_count-0.5,color=[0.8, 0.8, 0.8])
        inds = np.where(feature_type_labels==ft)
        h=plt.plot(inds, weights[vv,inds,pp],'o',color=colors[ft,:])
        lh.append(h)

    plt.axhline(0,color=[0.8, 0.8, 0.8])
    plt.xlabel('feature')

    roi_ind_ret = np.where([np.isin(roi_labels_retino[vv], ret_group_inds[ii]) for ii in range(len(ret_group_inds))])[0]
    roi_ind_categ = np.where([np.isin(roi_labels_categ[vv], categ_group_inds[ii]) for ii in range(len(categ_group_inds))])[0]
    if len(roi_ind_ret)==0:
        rname = categ_group_names[roi_ind_categ[0]]
    elif len(roi_ind_categ)==0:
        rname = ret_group_names[roi_ind_ret[0]]
    else:
        rname = '%s/%s'%(ret_group_names[roi_ind_ret[0]],categ_group_names[roi_ind_categ[0]])

    plt.title(''%())
    plt.xlim([-1, n_features_total]);
    plt.suptitle('Weights\nS%02d, %s\nExample voxel %d, %s, rho=%.2f, lambda=%.5f'%( subject, fitting_type,vv,rname, val_cc[vv],lambdas[best_lambdas[vv]]),fontsize=16);
    # plt.legend(lh, feature_type_names)

    print(feature_type_names)
    
    if fig_save_folder is not None:
        plt.savefig(os.path.join(fig_save_folder,'example_weights.png'))
        plt.savefig(os.path.join(fig_save_folder,'example_weights.pdf'))

        
        

def plot_uniqvar_violin_texture(subject, fitting_type, out, cc_cutoff=0.2, fig_save_folder=None):

    """
    Make a violin plot showing the distribution across voxels of the unique variance explained by each feature type.
    Computed by finding difference between R2 for full model and R2 w that feature type removed.
    """
    
    if 'texture' not in fitting_type:
        raise ValueError('this plot is just for texture model')        
          
    assert(out['val_cc'].shape[1]>1)
    
    plt.figure(figsize=(16,8))
   
    roi_labels_retino, roi_labels_categ, ret_group_inds, categ_group_inds, ret_group_names, categ_group_names, \
        n_rois_ret, n_rois_categ, n_rois = get_roi_info(subject, out)

    feature_info = copy.deepcopy(out['feature_info'])
    feature_type_labels, feature_type_names = feature_info
    n_feature_types = len(feature_type_names)
    
    val_cc = out['val_cc']
    val_r2 = get_r2(out)

    # Compute variance explained by each feature type - how well does the model without that feature type
    # do, compared to the model with all features? 
    # (subtract later columns from the first column)
    var_expl = np.tile(np.expand_dims(val_r2[:,0], axis=1), [1,n_feature_types]) - val_r2[:,1:] 

    colors = cm.plasma(np.linspace(0,1,n_feature_types))
    colors = np.flipud(colors)

    abv_thresh = val_cc[:,0]>cc_cutoff
    inds2use = abv_thresh

    for ff in range(n_feature_types):

        parts = plt.violinplot(var_expl[inds2use,ff], [ff])
        for pc in parts['bodies']:
            pc.set_color(colors[ff,:])
        parts['cbars'].set_color(colors[ff,:])
        parts['cmins'].set_color(colors[ff,:])
        parts['cmaxes'].set_color(colors[ff,:])

    #     plt.bar(ff, avg_ve[ff], color=colors[ff,:])

    plt.xticks(ticks=np.arange(0,n_feature_types),labels=feature_type_names,rotation=45, ha='right',rotation_mode='anchor')
    plt.axhline(0, color=[0.8, 0.8, 0.8])
    # plt.xlabel('feature type')
    plt.ylabel('Unique variance explained')

    plt.title('Showing all voxels with corr > %.2f, all ROIs (%d vox)'%(cc_cutoff, np.sum(inds2use)))

    plt.suptitle('S%02d, %s\nUnique variance explained: R2 with full model  - R2 with feature type removed'%(subject, fitting_type))
    plt.gcf().subplots_adjust(bottom=0.4)
    
    if fig_save_folder is not None:
        plt.savefig(os.path.join(fig_save_folder,'violin_uniq_var_texturefeat_allrois.png'))
        plt.savefig(os.path.join(fig_save_folder,'violin_uniq_var_texturefeat_allrois.pdf'))

def plot_uniqvar_bars_texture(subject, fitting_type, out, cc_cutoff=0.2, fig_save_folder=None):

    """
    Make a bar plot showing the distribution (mean +/- SEM) across voxels of the unique variance 
    explained by each feature type.
    Computed by finding difference between R2 for full model and R2 w that feature type removed.
    """
    
    if 'texture' not in fitting_type:
        raise ValueError('this plot is just for texture model')        
         
    assert(out['val_cc'].shape[1]>1)
    
    plt.figure(figsize=(16,8))

    roi_labels_retino, roi_labels_categ, ret_group_inds, categ_group_inds, ret_group_names, categ_group_names, \
        n_rois_ret, n_rois_categ, n_rois = get_roi_info(subject, out)

    feature_info = copy.deepcopy(out['feature_info'])
    feature_type_labels, feature_type_names = feature_info
    n_feature_types = len(feature_type_names)
    
    val_cc = out['val_cc']
    val_r2 = get_r2(out)

    # Compute variance explained by each feature type - how well does the model without that feature type
    # do, compared to the model with all features? 
    # (subtract later columns from the first column)
    var_expl = np.tile(np.expand_dims(val_r2[:,0], axis=1), [1,n_feature_types]) - val_r2[:,1:] 

    colors = cm.plasma(np.linspace(0,1,n_feature_types))
    colors = np.flipud(colors)

    abv_thresh = val_cc[:,0]>cc_cutoff
    inds2use = abv_thresh

    for ff in range(n_feature_types):

        mean = np.mean(var_expl[inds2use,ff])
        sem = np.std(var_expl[inds2use,ff])/np.sqrt(np.sum(inds2use))

        plt.bar(ff, mean, color=colors[ff,:])
        plt.errorbar(ff, mean, sem, color = colors[ff,:], ecolor='k')

    plt.xticks(ticks=np.arange(0,n_feature_types),labels=feature_type_names,rotation=45, ha='right',rotation_mode='anchor')
    plt.axhline(0, color=[0.8, 0.8, 0.8])
    # plt.xlabel('feature type')
    plt.ylabel('Unique variance explained (mean +/- SEM across voxels)')

    plt.title('Showing all voxels with corr > %.2f, all ROIs (%d vox)'%(cc_cutoff, np.sum(inds2use)))

    plt.suptitle('S%02d, %s\nUnique variance explained: R2 with full model  - R2 with feature type removed'%(subject, fitting_type))
    plt.gcf().subplots_adjust(bottom=0.4)
    
    if fig_save_folder is not None:
        plt.savefig(os.path.join(fig_save_folder,'bars_uniq_var_texturefeat_allrois.png'))
        plt.savefig(os.path.join(fig_save_folder,'bars_uniq_var_texturefeat_allrois.pdf'))

def plot_uniqvar_bars_eachroi_texture(subject, fitting_type, out, cc_cutoff=0.2, fig_save_folder=None):
    """
    Make a bar plot for each ROI, showing the distribution (mean +/- SEM)across voxels of the 
    unique variance explained by each feature type.
    Computed by finding difference between R2 for full model and R2 w that feature type removed.
    """

    if 'texture' not in fitting_type:
        raise ValueError('this plot is just for texture model')        
        
    assert(out['val_cc'].shape[1]>1)
    
    roi_labels_retino, roi_labels_categ, ret_group_inds, categ_group_inds, ret_group_names, categ_group_names, \
        n_rois_ret, n_rois_categ, n_rois = get_roi_info(subject, out)
    
    feature_info = copy.deepcopy(out['feature_info'])
    feature_type_labels, feature_type_names = feature_info
    n_feature_types = len(feature_type_names)
    
    val_cc = out['val_cc']
    val_r2 = get_r2(out)

    # Compute variance explained by each feature type - how well does the model without that feature type
    # do, compared to the model with all features? 
    # (subtract later columns from the first column)
    var_expl = np.tile(np.expand_dims(val_r2[:,0], axis=1), [1,n_feature_types]) - val_r2[:,1:] 

    
    # Preferred feature type, based on unique var explained. Separate plot each ROI.
    plt.figure(figsize=(24,20))
    npx = int(np.ceil(np.sqrt(n_rois)))
    npy = int(np.ceil(n_rois/npx))

    colors = cm.plasma(np.linspace(0,1,n_feature_types))
    colors = np.flipud(colors)

    for rr in range(n_rois):

        if rr<n_rois_ret:
            inds_this_roi = np.isin(roi_labels_retino, ret_group_inds[rr])
            rname = ret_group_names[rr]
        else:
            inds_this_roi = np.isin(roi_labels_categ, categ_group_inds[rr-n_rois_ret])
            rname = categ_group_names[rr-n_rois_ret]

        abv_thresh = val_cc[:,0]>cc_cutoff
        inds2use = np.logical_and(inds_this_roi, abv_thresh)


        plt.subplot(npx,npy,rr+1)

        if np.sum(inds2use)>0:
            for ff in range(n_feature_types):
                mean = np.mean(var_expl[inds2use,ff])
                sem = np.std(var_expl[inds2use,ff])/np.sqrt(np.sum(inds2use))

                plt.bar(ff, mean, color=colors[ff,:])
                plt.errorbar(ff, mean, sem, color = colors[ff,:], ecolor='k')

        plt.axhline(0, color=[0.8, 0.8, 0.8])
        plt.ylim([-0.02, 0.04])
        plt.xticks(ticks=np.arange(0,n_feature_types),labels=feature_type_names,rotation=45, ha='right',rotation_mode='anchor')

        if rr==n_rois-4:
            plt.ylabel('Unique variance explained')
        elif rr<n_rois-4:
            plt.xticks([])

        plt.title('%s (%d vox)'%(rname, np.sum(inds2use)))

    plt.suptitle('S%02d, %s\nUnique variance explained: R2 with full model  - R2 with feature type removed\nMean +/- SEM across voxels'%(subject, fitting_type))
    plt.gcf().subplots_adjust(bottom=0.3)
    
    if fig_save_folder is not None:
        plt.savefig(os.path.join(fig_save_folder,'uniq_var_texturefeat_eachroi.pdf'))
        plt.savefig(os.path.join(fig_save_folder,'uniq_var_texturefeat_eachroi.png'))

def plot_feature_prefs_uniqvar_texture(subject, fitting_type, out, cc_cutoff = 0.2, fig_save_folder=None):
    """
    Plot a histogram showing the distribution of preferred feature type across all voxels -
    based on unique var explained, i.e. the feature type that leads to biggest reduction in R2 when excluded.
    """ 
    
    if 'texture' not in fitting_type:
        raise ValueError('this plot is just for texture model')        
      
    plt.figure(figsize=(16,8))
    
    roi_labels_retino, roi_labels_categ, ret_group_inds, categ_group_inds, ret_group_names, categ_group_names, \
        n_rois_ret, n_rois_categ, n_rois = get_roi_info(subject, out)

    npx = np.ceil(np.sqrt(n_rois))
    npy = np.ceil(n_rois/npx)
    
    feature_info = copy.deepcopy(out['feature_info'])
    feature_type_labels, feature_type_names = feature_info
    n_feature_types = len(feature_type_names)

    colors = cm.plasma(np.linspace(0,1,n_feature_types))
    colors = np.flipud(colors)

    val_cc = out['val_cc']
    val_r2 = get_r2(out)

    # Compute variance explained by each feature type - how well does the model without that feature type
    # do, compared to the model with all features? 
    # (subtract later columns from the first column)
    var_expl = np.tile(np.expand_dims(val_r2[:,0], axis=1), [1,n_feature_types]) - val_r2[:,1:] 

    max_ve  = np.argmax(var_expl, axis=1)

    abv_thresh = val_cc[:,0]>cc_cutoff
    inds2use = abv_thresh

    unvals = np.arange(0,n_feature_types)
    counts = [np.sum(np.logical_and(max_ve==ff, inds2use)) for ff in unvals]

    # unvals, counts = np.unique(max_ve[inds2use], return_counts=True)
    for ff in range(n_feature_types):
        plt.bar(unvals[ff], counts[ff], color=colors[ff,:])

    plt.xticks(ticks=np.arange(0,n_feature_types),labels=feature_type_names,rotation=45, ha='right',rotation_mode='anchor')

    # plt.xlabel('feature type')
    plt.ylabel('Number of voxels "preferring"')

    plt.title('Showing all voxels with corr > %.2f, all ROIs (%d vox)'%(cc_cutoff, np.sum(inds2use)))
    plt.gcf().subplots_adjust(bottom=0.4)
    plt.suptitle('S%02d, %s\nPreferred feature type based on maximum unique variance explained'%(subject, fitting_type))
    
    if fig_save_folder is not None:
        plt.savefig(os.path.join(fig_save_folder,'hist_highest_uniqvar_features_allrois.pdf'))
        plt.savefig(os.path.join(fig_save_folder,'hist_highest_uniqvar_features_allrois.png'))
        
        
def plot_feature_prefs_uniqvar_texture_eachroi(subject, fitting_type, out, cc_cutoff = 0.2, fig_save_folder=None):
    """
    Plot a histogram showing the distribution of preferred feature type across all voxels -
    based on unique var explained, i.e. the feature type that leads to biggest reduction in R2 when excluded.
    """ 
    
    if 'texture' not in fitting_type:
        raise ValueError('this plot is just for texture model')        
      
    
    plt.figure(figsize=(16,16))
    
    roi_labels_retino, roi_labels_categ, ret_group_inds, categ_group_inds, ret_group_names, categ_group_names, \
        n_rois_ret, n_rois_categ, n_rois = get_roi_info(subject, out)

    npx = np.ceil(np.sqrt(n_rois))
    npy = np.ceil(n_rois/npx)
    
    feature_info = copy.deepcopy(out['feature_info'])
    feature_type_labels, feature_type_names = feature_info
    n_feature_types = len(feature_type_names)

    colors = cm.plasma(np.linspace(0,1,n_feature_types))
    colors = np.flipud(colors)

    val_cc = out['val_cc']
    val_r2 = get_r2(out)

    # Compute variance explained by each feature type - how well does the model without that feature type
    # do, compared to the model with all features? 
    # (subtract later columns from the first column)
    var_expl = np.tile(np.expand_dims(val_r2[:,0], axis=1), [1,n_feature_types]) - val_r2[:,1:] 

    max_ve  = np.argmax(var_expl, axis=1)

    for rr in range(n_rois):

        if rr<n_rois_ret:
            inds_this_roi = np.isin(roi_labels_retino, ret_group_inds[rr])
            rname = ret_group_names[rr]
        else:
            inds_this_roi = np.isin(roi_labels_categ, categ_group_inds[rr-n_rois_ret])
            rname = categ_group_names[rr-n_rois_ret]

        abv_thresh = val_cc[:,0]>cc_cutoff
        inds2use = np.logical_and(inds_this_roi, abv_thresh)

        unvals = np.arange(0,n_feature_types)
        counts = [np.sum(np.logical_and(max_ve==ff, inds2use)) for ff in unvals]

        plt.subplot(npx,npy,rr+1)

        for ff in range(len(unvals)):
            plt.bar(unvals[ff], counts[ff], color=colors[ff,:])

        plt.xticks(ticks=np.arange(0,n_feature_types),labels=feature_type_names,rotation=45, ha='right',rotation_mode='anchor')

        if rr==n_rois-4:
            plt.ylabel('Number of voxels "preferring"')
        elif rr<n_rois-4:
            plt.xticks([])
    #         plt.yticks([])

        plt.title('%s (%d vox)'%(rname, np.sum(inds2use)))

    plt.suptitle('S%02d, %s\nFeature type with highest unique var explained - showing all voxels with corr > %.2f'%(subject, fitting_type, cc_cutoff));

    plt.gcf().subplots_adjust(bottom=0.3)
    if fig_save_folder:
        plt.savefig(os.path.join(fig_save_folder,'hist_highest_uniqvar_features_eachroi.pdf'))
        plt.savefig(os.path.join(fig_save_folder,'hist_highest_uniqvar_features_eachroi.png'))

        
        
        
              
        
def scatter_compare_partial_models(subject, fitting_type, out, pp1, pp2, fig_save_folder):

    """
    Make a scatter plot for each ROI, plotting the performance of one partial model versus another
    """
    
     
    assert(out['val_cc'].shape[1]>1)

    roi_labels_retino, roi_labels_categ, ret_group_inds, categ_group_inds, ret_group_names, categ_group_names, \
        n_rois_ret, n_rois_categ, n_rois = get_roi_info(subject, out)

    if len(out['best_params'])>6:
        partial_version_names = out['best_params'][6]
    else:
        partial_version_names = out['partial_version_names']
        
    feature_info = copy.deepcopy(out['feature_info'])
    feature_type_labels, feature_type_names = feature_info
    n_feature_types = len(feature_type_names)
    
    val_cc = out['val_cc']
    val_r2 = get_r2(out)


    cc_cutoff = -100

    xlims = [-0.4, 0.8]
    ylims = [-0.4, 0.8]

    plt.figure(figsize=(24,20))
    npx = int(np.ceil(np.sqrt(n_rois)))
    npy = int(np.ceil(n_rois/npx))
    
    colors = cm.plasma(np.linspace(0,1,n_feature_types))
    colors = np.flipud(colors)
    
    
    for rr in range(n_rois):

        if rr<n_rois_ret:
            inds_this_roi = np.isin(roi_labels_retino, ret_group_inds[rr])
            rname = ret_group_names[rr]
        else:
            inds_this_roi = np.isin(roi_labels_categ, categ_group_inds[rr-n_rois_ret])
            rname = categ_group_names[rr-n_rois_ret]

        abv_thresh = val_cc[:,0]>cc_cutoff
        inds2use = np.logical_and(inds_this_roi, abv_thresh)

        plt.subplot(npx,npy,rr+1)

        if np.sum(inds2use)>0:

            xvals = val_r2[inds2use,pp1]
            yvals = val_r2[inds2use,pp2]

            plt.plot(xvals,yvals,'.',color=colors[pp2-1])

        if rr==0:
            plt.xlabel('r2: %s'%partial_version_names[pp1])
            plt.ylabel('r2: %s'%partial_version_names[pp2])
        plt.axis('square')

        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.plot(xlims, ylims, color='k')
        plt.axvline(color=[0.8, 0.8, 0.8])
        plt.axhline(color=[0.8, 0.8, 0.8])
        plt.xticks([])

        plt.title('%s (%d vox)'%(rname, np.sum(inds2use)))

    plt.suptitle('S%02d, %s\nr2 %s vs. r2 %s'%(subject, fitting_type, partial_version_names[pp2], partial_version_names[pp1]))

    if fig_save_folder is not None:
        plt.savefig(os.path.join(fig_save_folder,'scatter_compare_%s_vs_%s.png'%(partial_version_names[pp2], \
                                                                                 partial_version_names[pp1])))
        plt.savefig(os.path.join(fig_save_folder,'scatter_compare_%s_vs_%s.pdf'%(partial_version_names[pp2], \
                                                                                 partial_version_names[pp1])))


     