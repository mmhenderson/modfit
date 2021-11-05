import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import numpy as np
import os
from utils import roi_utils

def plot_spatial_rf_circles(subject, fitting_type, out, roi_def=None, skip_inds=None, r2_cutoff = 0.10, screen_eccen_deg = 8.4, \
                            fig_save_folder = None):

    """
    Make a plot for each ROI showing the visual field coverage of pRFs in that ROI: 
    each circle is a voxel with the size of the circle indicating the pRF's size (1 SD)
    """

    pp=0
    best_models_deg = out['best_params'][0] * screen_eccen_deg
    if len(best_models_deg.shape)==2:
        best_models_deg = np.expand_dims(best_models_deg, axis=1)
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
    
    val_r2 = out['val_r2'][:,0]
    abv_thresh = val_r2>r2_cutoff
    
    plt.figure(figsize=(24,18))

    npy = int(np.ceil(np.sqrt(n_rois-len(skip_inds))))
    npx = int(np.ceil((n_rois-len(skip_inds))/npy))

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

            inds2use = np.where(np.logical_and(inds_this_roi, abv_thresh))[0]

            pi+=1
            plt.subplot(npx,npy,pi)
            ax = plt.gca()

            for vi, vidx in enumerate(inds2use):

                plt.plot(best_models_deg[vidx,pp,0], best_models_deg[vidx,pp,1],'.',color='k')
                circ = matplotlib.patches.Circle((best_models_deg[vidx,pp,0], best_models_deg[vidx,pp,1]), best_models_deg[vidx,pp,2], 
                                                 color = [0.8, 0.8, 0.8], fill=False)
                ax.add_artist(circ)

            plt.axis('square')

            plt.xlim([-1.5*screen_eccen_deg, 1.5*screen_eccen_deg])
            plt.ylim([-1.5*screen_eccen_deg, 1.5*screen_eccen_deg])
            plt.xticks(np.arange(-8,9,4))
            plt.yticks(np.arange(-8,9,4))

            boxcolor = [0.6, 0.6, 0.6]
            plt.plot([screen_eccen_deg/2,screen_eccen_deg/2], [screen_eccen_deg/2, -screen_eccen_deg/2],color=boxcolor)
            plt.plot([-screen_eccen_deg/2,-screen_eccen_deg/2], [screen_eccen_deg/2, -screen_eccen_deg/2],color=boxcolor)
            plt.plot([-screen_eccen_deg/2,screen_eccen_deg/2], [screen_eccen_deg/2, screen_eccen_deg/2],color=boxcolor)
            plt.plot([-screen_eccen_deg/2,screen_eccen_deg/2], [-screen_eccen_deg/2, -screen_eccen_deg/2],color=boxcolor)

            if pi==(npx-1)*npy+1:
                plt.xlabel('x coord (deg)')
                plt.ylabel('y coord (deg)')
            else:
                plt.xticks([])
                plt.yticks([])
            plt.title('%s (%d vox)'%(rname, len(inds2use)))

    plt.suptitle('pRF estimates\nshowing all voxels with R2 > %.2f\nS%02d, %s'%(r2_cutoff, subject, fitting_type));

    if fig_save_folder is not None:
        plt.savefig(os.path.join(fig_save_folder,'spatial_prf_distrib.pdf'))
        plt.savefig(os.path.join(fig_save_folder,'spatial_prf_distrib.png'))



def plot_size_vs_eccen(subject, fitting_type,out, roi_def=None, skip_inds=None, r2_cutoff=0.10, screen_eccen_deg = 8.4, \
                       fig_save_folder=None ):
    """
    Create a scatter plot for each ROI, showing the size of each voxel's best pRF estimate versus its eccentricity.
    """

    
    best_ecc_deg, best_angle_deg, best_size_deg = get_prf_pars_deg(out, screen_eccen_deg)
    
    fits_ignore = np.round(best_size_deg,0)==84 
    # ignoring this pRF, because this size value is made up - this
    # pRF was actually just a flat function covering whole visual field.
    size_lims = [-1, 1.25*np.max(best_size_deg[~fits_ignore])]
    eccen_lims = [-1,  1.25*np.max(best_ecc_deg)]
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
    
    val_r2 = out['val_r2'][:,0]
    abv_thresh = val_r2>r2_cutoff
    
    npy = int(np.ceil(np.sqrt(n_rois-len(skip_inds))))
    npx = int(np.ceil((n_rois-len(skip_inds))/npy))

    plt.figure(figsize=(24,20))

    pi=0
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

            inds2use = np.where(inds_this_roi & abv_thresh & ~fits_ignore)[0]

            pi+=1
            plt.subplot(npx,npy,pi)
            ax = plt.gca()

            xvals = best_ecc_deg[inds2use]
            yvals = best_size_deg[inds2use]
            plt.plot(xvals, yvals, '.')

            X = np.concatenate([xvals[:,np.newaxis], np.ones((len(inds2use),1))], axis=1)
            y = yvals[:,np.newaxis]
            linefit =  np.linalg.pinv(X) @ y
            
            yhat = xvals*linefit[0] + linefit[1]
            plt.plot(xvals, yhat, '-', color=[0.6, 0.6, 0.6])
            
            plt.xlim(eccen_lims)
            plt.ylim(size_lims)
            if pi==(npx-1)*npy+1:
                plt.xlabel('eccen (deg)')
                plt.ylabel('size (deg)')
            else:
                plt.xticks([])
                plt.yticks([])

            plt.axhline(0,color=[0.8, 0.8, 0.8])
            plt.axvline(0,color=[0.8, 0.8, 0.8])
            plt.title('%s (%d vox)'%(rname, len(inds2use)))

    plt.suptitle('pRF estimates\nshowing all voxels with R2 > %.2f\nS%02d, %s'%(r2_cutoff, subject, fitting_type))
    
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
    best_angle_deg  = np.mod(np.arctan2(best_models_deg[:,pp,1], best_models_deg[:,pp,0])*180/np.pi, 360)
    best_angle_deg[best_ecc_deg==0.0] = np.nan
    best_size_deg = best_models_deg[:,pp,2]
    
    return best_ecc_deg, best_angle_deg, best_size_deg


def plot_prf_grid(prf_models, xy_circ = [-0.4, -0.4], screen_eccen_deg = 8.4):

    """
    Visualize grid of pRF positions at each level of size.
    """
    
    unique_sizes = np.unique(np.round(prf_models[:,2],4))
    plt.figure(figsize=(18,12));

    for si, size in enumerate(unique_sizes):

        inds = np.where(np.round(prf_models[:,2],4)==size)[0]
        
        prf_models_plot = prf_models[inds,:]
        ind = np.argmin(np.abs(prf_models_plot[:,0] - xy_circ[0]) + np.abs(prf_models_plot[:,1] - xy_circ[1]))
        xy_circ_actual = [prf_models_plot[ind,0], prf_models_plot[ind,1]]

        plt.subplot(3,4,si+1)
        ax = plt.gca()
        plt.plot(prf_models[inds,0]*screen_eccen_deg, prf_models[inds,1]*screen_eccen_deg, '.')
        plt.plot(xy_circ_actual[0]*screen_eccen_deg, xy_circ_actual[1]*screen_eccen_deg, '.',color='k')
        circ = matplotlib.patches.Circle((xy_circ_actual[0]*screen_eccen_deg, xy_circ_actual[1]*screen_eccen_deg), \
                                         size*screen_eccen_deg, color = [0.2, 0.2, 0.2], fill=False)
        ax.add_artist(circ)
        plt.axis('square')
        plt.xlim([-1.5*screen_eccen_deg, 1.5*screen_eccen_deg])
        plt.ylim([-1.5*screen_eccen_deg, 1.5*screen_eccen_deg])
        plt.xticks(np.arange(-8,9,4))
        plt.yticks(np.arange(-8,9,4))

        plt.plot([screen_eccen_deg/2,screen_eccen_deg/2], [screen_eccen_deg/2, -screen_eccen_deg/2],color=[0.8, 0.8, 0.8])
        plt.plot([-screen_eccen_deg/2,-screen_eccen_deg/2], [screen_eccen_deg/2, -screen_eccen_deg/2],color=[0.8, 0.8, 0.8])
        plt.plot([-screen_eccen_deg/2,screen_eccen_deg/2], [screen_eccen_deg/2, screen_eccen_deg/2],color=[0.8, 0.8, 0.8])
        plt.plot([-screen_eccen_deg/2,screen_eccen_deg/2], [-screen_eccen_deg/2, -screen_eccen_deg/2],color=[0.8, 0.8, 0.8])

        if si>7:
            plt.xlabel('x coord (deg)')
        if np.mod(si,4)==0:
            plt.ylabel('y coord (deg)')

        plt.title('pRF sigma=%.2f deg'%(size*screen_eccen_deg))