import numpy as np
import os

from utils import prf_utils
from model_fitting import initialize_fitting
from utils import default_paths
from analyze_fits import analyze_gabor_params
from feature_extraction import fwrf_features

def choose_models():

    which_prf_grid=5
    models = initialize_fitting.get_prf_models(which_grid = which_prf_grid)
    angle_deg, eccen_deg = prf_utils.cart_to_pol(models[:,0]*8.4, models[:,1]*8.4)
    # choosing just a sub-set of the pRFs to simulate here, for speed of computation
    ecc_use = np.unique(eccen_deg.round(2))[1:-2:2]
    ang_use = np.unique(angle_deg.round(2))[0::2]
    size_use = np.unique(models[:,2].round(2))[0::2]

    n_prfs_total = models.shape[0]
    inds_use = np.zeros((n_prfs_total,),dtype=bool)

    egrid, agrid, sgrid = np.meshgrid(ecc_use, ang_use, size_use)
    for e,a,s in zip(egrid.ravel(), agrid.ravel(), sgrid.ravel()):
        x,y = prf_utils.pol_to_cart(a,e)
        x/=8.4; y/=8.4;
        dist = np.sum(np.abs(models-[x,y,s]), axis=1)
        ind = np.argmin(dist)
        assert inds_use[ind]==False
        inds_use[ind] = True

    prf_inds_do = np.where(inds_use)[0]
    
    return prf_inds_do

def make_sim_data(noise_mult=0.10):

    n_features = 96
    
    prf_inds_do = choose_models()
    n_prfs_use = len(prf_inds_do)
    n_voxels = n_prfs_use * n_features

    # simulating based on images shown to S1, should be similar for other subs
    ss = 1
    which_prf_grid = 5
    floader = fwrf_features.fwrf_feature_loader(subject=ss, image_set='S%d'%ss, \
                                     which_prf_grid=which_prf_grid, feature_type='gabor_solo')

    sf_unique, ori_unique = analyze_gabor_params.get_gabor_feature_info(n_ori=12, n_sf=8)

    n_sf = len(sf_unique)
    n_ori = len(ori_unique)
    simulated_voxel_orient = np.tile(np.tile(ori_unique, [n_sf]), [n_prfs_use])
    simulated_voxel_sf = np.tile(np.repeat(sf_unique, [n_ori]), [n_prfs_use])
    simulated_voxel_prf_inds = np.repeat(prf_inds_do, n_ori*n_sf)

    image_inds = np.arange(10000)
    n_images = len(image_inds)

    simulated_voxel_data = np.zeros((n_images, n_voxels))

    for mm, prf_ind in enumerate(prf_inds_do):

        voxel_inds = np.arange(mm*n_features, (mm+1)*n_features)

        feat, defined = floader.load(image_inds, prf_ind)

        noise = np.random.normal(0,1,np.shape(feat)) * noise_mult

        # each column represents a voxel tuned for one feature.
        # (having a 1 response to that feature and 0 elsewhere)
        simulated_voxel_data[:,voxel_inds] = feat + noise
        
        
    folder_save = os.path.join(default_paths.gabor_texture_feat_path, 'simulated_data')
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
        
    fn2save = os.path.join(folder_save, 'S%d_sim_data_addnoise_%.2f.npy'%(ss,noise_mult))
    print('saving to %s'%fn2save)
    np.save(fn2save, {'sim_data': simulated_voxel_data,\
                      'simulated_voxel_prf_inds': simulated_voxel_prf_inds, \
                      'simulated_voxel_orient': simulated_voxel_orient, \
                      'simulated_voxel_sf': simulated_voxel_sf})
    