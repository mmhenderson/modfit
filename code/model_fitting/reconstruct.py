from __future__ import division
import sys
import time
import numpy as np
import copy
import torch
import scipy.stats

from utils import numpy_utils, torch_utils, stats_utils

def get_population_recons(best_params, prf_models, voxel_data, roi_def, images, _feature_extractor, \
                          zscore=False, debug=False, dtype=np.float32):
    
    """ 
    Invert encoding model weights for a group of voxels, to get population-level stimulus representations.
    """
  
    params = best_params
    device = _feature_extractor.device

    n_trials, n_voxels = len(images), len(params[0])
    n_prfs = prf_models.shape[0]
    # n_features = params[1].shape[1]
    n_features = _feature_extractor.max_features
    n_features_max = n_features
    n_voxels = np.shape(voxel_data)[1]

    retlabs, facelabs, placelabs, bodylabs, ret_names, face_names, place_names, body_names = roi_def 
    nret = len(ret_names)
    nface = len(face_names)
    nplace = len(place_names)
    nbody = len(body_names)    
    n_rois = len(ret_names) + len(face_names) + len(place_names) + len(body_names)

    is_ret = np.arange(0, n_rois)<nret
    is_face = (np.arange(0, n_rois)>=nret) & (np.arange(0, n_rois)<nret+nface)
    is_place = (np.arange(0, n_rois)>=nret+nface) & (np.arange(0, n_rois)<nret+nface+nplace)
    is_body = np.arange(0, n_rois)>=nret+nface+nplace


    best_models, weights, bias, features_mt, features_st, best_model_inds = params
   
    if zscore:
        if hasattr(_feature_extractor, 'zgroup_labels') and \
                        _feature_extractor.zgroup_labels is not None:
            zscore_in_groups = True
            zgroup_labels = _feature_extractor.zgroup_labels
            print('will z-score columns in groups')
        else:
            zscore_in_groups = False
            print('will z-score each column')
    else:
        print('will not z-score')

    pred_models = np.full(fill_value=0, shape=(n_trials, n_features_max, n_prfs), dtype=dtype)
    feature_inds_defined_each_prf = np.full(fill_value=0, shape=(n_features_max, n_prfs), dtype=bool)

    pop_recons = np.zeros((n_trials, n_features, n_rois),dtype=dtype)

    recon_r2 = np.zeros((n_trials, n_rois))
    recon_angle = np.zeros((n_trials, n_rois))
    recon_corrcoef = np.zeros((n_trials, n_rois))

    start_time = time.time()    
    with torch.no_grad(): # make sure local gradients are off to save memory

        # First gather texture features for all pRFs.

        _feature_extractor.clear_big_features()

        for mm in range(n_prfs):
            if mm>1 and debug:
                break
            print('Getting features for prf %d: [x,y,sigma] is [%.2f %.2f %.4f]'%(mm, \
                                  prf_models[mm,0],  prf_models[mm,1],  prf_models[mm,2] ))
            # all_feat_concat is size [ntrials x nfeatures]
            # nfeatures may be less than n_features_max, because n_features_max is the 
            # largest number possible for any pRF.
            # feature_inds_defined is length max_features, and tells which of the features in 
            # max_features are includes in features.
            all_feat_concat, feature_inds_defined = \
                    _feature_extractor(images, prf_models[mm,:], mm, fitting_mode=False)

            all_feat_concat = torch_utils.get_value(all_feat_concat)
            if zscore:
                if zscore_in_groups:
                    all_feat_concat = numpy_utils.zscore_in_groups(all_feat_concat, zgroup_labels)
                else:
                    all_feat_concat = scipy.stats.zscore(all_feat_concat, axis=0)
                # if any entries in std are zero or nan, this gives bad result - fix these now.
                # these bad entries will also be zero in weights, so doesn't matter. 
                # just want to avoid nans.
                all_feat_concat[np.isnan(all_feat_concat)] = 0.0 
                all_feat_concat[np.isinf(all_feat_concat)] = 0.0 


            pred_models[:,feature_inds_defined,mm] = all_feat_concat
            feature_inds_defined_each_prf[:,mm] = feature_inds_defined        
            if mm>0:
                # this population approach won't work if the features are not same across prfs.
                assert(np.all(feature_inds_defined ==feature_inds_defined_each_prf[:,0]))

        _feature_extractor.clear_big_features()



        for rr in range(n_rois):

            if rr>1 and debug:
                break

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
            print('Processing %s'%rname)
            if np.sum(inds_this_roi)==0:
                print('no voxels - skipping')
            #     continue

            # weights is [voxels x features]
            W = weights[inds_this_roi,:,0]
            feature_inds = feature_inds_defined_each_prf[:,0]
            W = W[:,feature_inds]
            # validation voxel data is [trials x voxels]
            V = voxel_data[:,inds_this_roi]

            # reconstruction is [features x trials]
            recon_pop = np.linalg.pinv(W) @ V.T
            recon_pop = recon_pop.T

            pop_recons[:,:,rr] = recon_pop

            # "ground truth" feature for each trial, will be the average of the feature vectors for 
            # the different pRFs. Weighting them according to how many voxels in the roi prefer that
            # prf position. Probably other ways to do this...
            feature_vectors = pred_models[:,feature_inds,:]
            feature_vectors = feature_vectors[:,:,best_model_inds[inds_this_roi,0]]
            avg_feature = np.mean(feature_vectors, axis=2)
            # avg_feature = feature_vectors[:,:,0]

            # computing r2 for each single-trial recon (i.e. over the features dim)
            # so need to transpose for this fn
            recon_r2[:,rr] = stats_utils.get_r2(avg_feature.T, recon_pop.T)
            recon_corrcoef[:,rr] = stats_utils.get_corrcoef(avg_feature.T, recon_pop.T)
            dp = np.sum(avg_feature*recon_pop, axis=1)
            cosangle = dp / (np.sqrt(np.sum(avg_feature**2, axis=1)) * \
                             np.sqrt(np.sum(recon_pop**2, axis=1)))
            recon_angle[:,rr] = np.arccos(cosangle) * 180/np.pi

        recon_r2 = np.nan_to_num(recon_r2)
        recon_corrcoef = np.nan_to_num(recon_corrcoef)
        recon_angle = np.nan_to_num(recon_angle)
        
    return pop_recons, recon_r2, recon_corrcoef, recon_angle


def get_single_voxel_recons(best_params, prf_models, voxel_data, images, _feature_extractor, \
                          zscore=False, debug=False, dtype=np.float32):
    
    """ 
    Invert encoding model weights for a single voxel, get a model based stimulus representation.
    """
  
    params = best_params
    device = _feature_extractor.device

    n_trials, n_voxels = len(images), len(params[0])
    n_prfs = prf_models.shape[0]
    # n_features = params[1].shape[1]
    n_features = _feature_extractor.max_features
    n_features_max = n_features
    n_voxels = np.shape(voxel_data)[1]

    best_models, weights, bias, features_mt, features_st, best_model_inds = params
   
    if zscore:
        if hasattr(_feature_extractor, 'zgroup_labels') and \
                        _feature_extractor.zgroup_labels is not None:
            zscore_in_groups = True
            zgroup_labels = _feature_extractor.zgroup_labels
            print('will z-score columns in groups')
        else:
            zscore_in_groups = False
            print('will z-score each column')
    else:
        print('will not z-score')

    pred_models = np.full(fill_value=0, shape=(n_trials, n_features_max, n_prfs), dtype=dtype)
    feature_inds_defined_each_prf = np.full(fill_value=0, shape=(n_features_max, n_prfs), dtype=bool)

    voxel_recons = np.zeros((n_trials, n_features, n_voxels),dtype=dtype)

    recon_r2 = np.zeros((n_trials, n_voxels))
    recon_angle = np.zeros((n_trials, n_voxels))
    recon_corrcoef = np.zeros((n_trials, n_voxels))

    start_time = time.time()    
    with torch.no_grad(): # make sure local gradients are off to save memory

        # First gather texture features for all pRFs.

        _feature_extractor.clear_big_features()

        for mm in range(n_prfs):
            if mm>1 and debug:
                break
            print('Getting features for prf %d: [x,y,sigma] is [%.2f %.2f %.4f]'%(mm, \
                                  prf_models[mm,0],  prf_models[mm,1],  prf_models[mm,2] ))
            # all_feat_concat is size [ntrials x nfeatures]
            # nfeatures may be less than n_features_max, because n_features_max is the 
            # largest number possible for any pRF.
            # feature_inds_defined is length max_features, and tells which of the features in 
            # max_features are includes in features.
            all_feat_concat, feature_inds_defined = \
                    _feature_extractor(images, prf_models[mm,:], mm, fitting_mode=False)

            all_feat_concat = torch_utils.get_value(all_feat_concat)
            if zscore:
                if zscore_in_groups:
                    all_feat_concat = numpy_utils.zscore_in_groups(all_feat_concat, zgroup_labels)
                else:
                    all_feat_concat = scipy.stats.zscore(all_feat_concat, axis=0)
                # if any entries in std are zero or nan, this gives bad result - fix these now.
                # these bad entries will also be zero in weights, so doesn't matter. 
                # just want to avoid nans.
                all_feat_concat[np.isnan(all_feat_concat)] = 0.0 
                all_feat_concat[np.isinf(all_feat_concat)] = 0.0 


            pred_models[:,feature_inds_defined,mm] = all_feat_concat
            feature_inds_defined_each_prf[:,mm] = feature_inds_defined        
            if mm>0:
                # this population approach won't work if the features are not same across prfs.
                assert(np.all(feature_inds_defined ==feature_inds_defined_each_prf[:,0]))

        _feature_extractor.clear_big_features()

        for vv in range(n_voxels):

            if vv>1 and debug:
                break
            print('processing voxel %d of %d'%(vv, n_voxels))
            # weights is [voxels x features]
            W = weights[vv:vv+1,:,0]
            feature_inds = feature_inds_defined_each_prf[:,0]
            W = W[:,feature_inds]
            # validation voxel data is [trials x voxels]
            V = voxel_data[:,vv:vv+1]

            # reconstruction is [features x trials]
            recon = np.linalg.pinv(W) @ V.T
            # [trials x features]
            recon = recon.T

            voxel_recons[:,:,vv] = recon

            # "ground truth" feature for each trial
            feature_vector = pred_models[:,feature_inds,best_model_inds[vv,0]]
           
            # computing r2 for each single-trial recon (i.e. over the features dim)
            # so need to transpose for this fn
            recon_r2[:,vv] = stats_utils.get_r2(feature_vector.T, recon.T)
            recon_corrcoef[:,vv] = stats_utils.get_corrcoef(feature_vector.T, recon.T)
            dp = np.sum(feature_vector*recon, axis=1)
            cosangle = dp / (np.sqrt(np.sum(feature_vector**2, axis=1)) * \
                             np.sqrt(np.sum(recon**2, axis=1)))
            recon_angle[:,vv] = np.arccos(cosangle) * 180/np.pi

        recon_r2 = np.nan_to_num(recon_r2)
        recon_corrcoef = np.nan_to_num(recon_corrcoef)
        recon_angle = np.nan_to_num(recon_angle)
        
    return voxel_recons, recon_r2, recon_corrcoef, recon_angle

    