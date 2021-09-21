import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import cortex

"""
General use functions for plotting encoding model fit results.
Input to most of these functions is 'out', which is a dictionary containing 
fit results. Created by the model fitting code in model_fitting/fit_model.py
"""

from utils import roi_utils, nsd_utils
from plotting_and_analysis.analysis_utils import get_r2, get_roi_info, get_combined_rois
from plotting_and_analysis.spatial_fits import get_prf_pars_deg
from plotting_and_analysis.plot_utils import get_full_surface, get_full_volume
from plotting_and_analysis import plot_utils

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

def plot_summary_pycortex(subject, fitting_type, out, port):

    """
    Use pycortex webgl function to plot some summary statistics for encoding model fits, in surface space.
    Plots pRF spatial parameters, and the model's prediction performance on validation set.
    """
    
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
    
    if out['brain_nii_shape'] is not None:
        print('Data is in 3d volume space')
        xfmname = 'func1pt8_to_anat0pt8_autoFSbbr'
        nii_shape = out['brain_nii_shape']
        mask_3d = np.reshape(voxel_mask, nii_shape, order='C')
        
        dat2plot = {'ROI labels (retinotopic)': cortex.Volume(data=get_full_volume(retlabs, voxel_mask, \
                                                                                   nii_shape),\
                                              subject=substr, cmap='Accent',vmin = 0, vmax = np.max(retlabs), \
                                              xfmname=xfmname, mask=mask_3d), \
            'ROI labels (category-selective)': cortex.Volume(data=get_full_volume(catlabs, voxel_mask, nii_shape),\
                                             subject=substr, cmap='Accent',vmin = 0, vmax = np.max(catlabs), \
                                                             xfmname=xfmname, mask=mask_3d), \
            'pRF eccentricity': cortex.Volume(data=get_full_volume(best_ecc_deg, voxel_mask, nii_shape), \
                                              subject=substr, cmap='PRGn',\
                                              vmin=0, vmax=7, xfmname=xfmname, mask=mask_3d),\
            'pRF angle': cortex.Volume(data=get_full_volume(best_angle_deg, voxel_mask, nii_shape), \
                                           subject=substr, cmap='Retinotopy_RYBCR', \
                                           vmin=0, vmax=360, xfmname=xfmname, mask=mask_3d), \
            'pRF size': cortex.Volume(data = get_full_volume(best_size_deg, voxel_mask, nii_shape),\
                                      subject=substr, cmap='PRGn', vmin=0, vmax=4, \
                                      xfmname=xfmname, mask=mask_3d), \
            'Correlation (validation set)': cortex.Volume(data = get_full_volume(val_cc, voxel_mask, nii_shape),\
                                                          subject=substr, cmap='PuBu', vmin=cmin, vmax=cmax, \
                                                         xfmname=xfmname, mask=mask_3d), \
            'R2 (validation set)': cortex.Volume(data = get_full_volume(val_r2, voxel_mask, nii_shape), \
                                                 subject=substr, cmap='PuBu', \
                                                 vmin=rmin, vmax=rmax, xfmname=xfmname, mask=mask_3d), \
           }

        
    else:
        print('Data is in nativesurface space')

        dat2plot = {'ROI labels (retinotopic)': cortex.Vertex(data = get_full_surface(retlabs, voxel_mask), \
                                                subject=substr, cmap='Accent',vmin = 0, vmax = np.max(retlabs)), \
                    'ROI labels (category-selective)': cortex.Vertex(data = get_full_surface(catlabs, voxel_mask),\
                                                                     subject=substr, cmap='Accent',vmin = 0, vmax =np.max(catlabs)),\
                    'pRF eccentricity': cortex.Vertex(data = get_full_surface(best_ecc_deg, voxel_mask), \
                                                      subject=substr, \
                                                      cmap='PRGn', vmin=0, vmax=7), \
                    'pRF angle': cortex.Vertex(data = get_full_surface(best_angle_deg, voxel_mask), \
                                               subject=substr, \
                                               cmap='Retinotopy_RYBCR', vmin=0, vmax=360), \
                    'pRF size': cortex.Vertex(data = get_full_surface(best_size_deg, voxel_mask), subject=substr, \
                                              cmap='PRGn', vmin=0, vmax=4), \
                    'Correlation (validation set)': cortex.Vertex(data = get_full_surface(val_cc, voxel_mask),\
                                                                  subject=substr, \
                                                                  cmap='PuBu', vmin=cmin, vmax=cmax), \
                    'R2 (validation set)': cortex.Vertex(data = get_full_surface(val_r2, voxel_mask), \
                                                         subject=substr, \
                                                         cmap='PuBu', vmin=rmin, vmax=rmax), \
                   }

    dat2plot.keys()
    
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
        
        
def plot_r2_vs_nc(subject, fitting_type, out, fig_save_folder=None, fig_size=None):

    """
    Create scatter plots for each ROI, comparing each voxel's R2 prediction to the noise ceiling.
    """
  
    voxel_ncsnr = out['voxel_ncsnr'].ravel()[out['voxel_index'][0]]
    noise_ceiling = nsd_utils.ncsnr_to_nc(voxel_ncsnr)/100
    val_r2 = get_r2(out)[:,0]

    inds2use = np.ones(np.shape(val_r2))==1

    sp = plot_utils.scatter_plot(color=None, xlabel='Noise Ceiling', ylabel='R2', xlims=[-0.1, 0.7], ylims=[-0.1, 0.7], \
                      xticks=[0, 0.2, 0.4, 0.6], yticks=[0, 0.2, 0.4, 0.6],\
                                                            show_diagonal=True, show_axes=True);

    if fig_size is None:
        fig_size = (20,18)
    plot_utils.create_roi_subplots(np.concatenate([noise_ceiling[:,np.newaxis],val_r2[:,np.newaxis]], axis=1), inds2use, sp, subject, out,\
                        suptitle='S%02d, %s\nComparing model performance to noise ceiling'%(subject, fitting_type), \
                       label_just_corner=True, figsize=fig_size)
    plt.gcf().subplots_adjust(bottom=0.5)
    if fig_save_folder is not None:
        plt.savefig(os.path.join(fig_save_folder,'r2_vs_noiseceiling.png'))
        plt.savefig(os.path.join(fig_save_folder,'r2_vs_noiseceiling.pdf'))
        
