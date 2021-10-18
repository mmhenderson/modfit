from __future__ import division
import sys
import os
import time
import numpy as np
import copy
import torch
import scipy.stats

from utils import numpy_utils, torch_utils, stats_utils, default_paths

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


def get_single_voxel_recons(best_params, prf_models, voxel_data, images, _feature_extractor,\
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

    print('Preallocating arrays')
       
    # loading pre-computed linear discriminant analysis features
    # only have these made for sketch tokens so far
    assert('sketch_tokens' in _feature_extractor.features_file)      
    sketch_token_feat_path = default_paths.sketch_token_feat_path
    axes_to_do = ['animacy','indoor_outdoor']
    n_sem_axes = len(axes_to_do)
    sem_proj_slope = np.zeros((n_voxels, n_sem_axes))
    sem_proj_inter = np.zeros((n_voxels, n_sem_axes))
    sem_proj_r2 = np.zeros((n_voxels, n_sem_axes))
    recon_sem_proj = np.zeros((n_trials, n_voxels, n_sem_axes))
    actual_sem_proj = np.zeros((n_trials, n_voxels, n_sem_axes))
    
    lda_result_list = []
    for aa, discrim_type in enumerate(axes_to_do):
        features_file = os.path.join(sketch_token_feat_path, 'LDA', \
                                                  'S%d_LDA_%s.npy'%(_feature_extractor.subject, discrim_type))
        print('loading from %s'%features_file)
        lda_result = np.load(features_file, allow_pickle=True).item()
        lda_result_list.append(lda_result)
        
#     recon_angle = np.zeros((n_trials, n_voxels))
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

            sys.stdout.flush()
            
        _feature_extractor.clear_big_features()

        for vv in range(n_voxels):

            if vv>1 and debug:
                break
            print('processing voxel %d of %d'%(vv, n_voxels))
            sys.stdout.flush()
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

#             voxel_recons[:,:,vv] = recon

            # "ground truth" feature for each trial
            mm = best_model_inds[vv,0]
            feature_vector = pred_models[:,feature_inds,mm]
           
            # computing corr coef for each single-trial recon (i.e. over the features dim)
            recon_corrcoef[:,vv] = stats_utils.get_corrcoef(feature_vector.T, recon.T)
#             # dot product of mean centered recon w real feature
#             feature_vector_centered = feature_vector - np.tile(np.mean(feature_vector, \
#                                                    axis=1, keepdims=True), [1,n_features])
#             recon_centered = recon - np.tile(np.mean(recon, axis=1, keepdims=True),[1, n_features])
#             dp = np.sum(feature_vector_centered*recon_centered, axis=1)
#             cosangle = dp / (np.sqrt(np.sum(feature_vector_centered**2, axis=1)) * \
#                              np.sqrt(np.sum(recon_centered**2, axis=1)))
#             recon_angle[:,vv] = np.arccos(cosangle) * 180/np.pi
            
            # now going to project the reconstruction onto various semantic axes
           
            for aa, discrim_type in enumerate(axes_to_do):
                
                lda_result = lda_result_list[aa]
                lda_wts = lda_result['wts'][mm]
                lda_pre_mean = lda_result['pre_mean'][mm]

                # Projecting both the "real" feature vector and the 
                # "reconstruction" onto the semantic axis
                features_submean = feature_vector - np.tile(lda_pre_mean[np.newaxis,:], [n_trials,1])
                features_proj = features_submean @ lda_wts
                recon_submean = recon - np.tile(lda_pre_mean[np.newaxis,:], [n_trials,1])
                recon_proj = recon_submean @ lda_wts

                # finding the slope of relationship between real and actual projections
                a = np.concatenate([features_proj, np.ones(np.shape(features_proj))], axis=1)
                b = recon_proj
                x = np.linalg.pinv(a) @ b
                slope = x[0]
                inter = x[1]

                b_pred = a[:,0]*slope + inter
                ssr = np.sum((b_pred-b[:,0])**2)
                sst = np.sum((b[:,0] - np.mean(b[:,0]))**2)
                r2 = 1-ssr/sst
                    
                sem_proj_slope[vv,aa] = slope
                sem_proj_inter[vv,aa] = inter
                sem_proj_r2[vv,aa] = r2   
        
                recon_sem_proj[:,vv,aa] = np.squeeze(recon_proj)
                actual_sem_proj[:,vv,aa] = np.squeeze(features_proj)
        
    return recon_corrcoef, sem_proj_slope, sem_proj_inter, sem_proj_r2, \
                    recon_sem_proj, actual_sem_proj, axes_to_do

    