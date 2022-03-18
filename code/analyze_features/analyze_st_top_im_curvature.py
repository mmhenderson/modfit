import os,sys
import copy
import time
import numpy as np
import pandas as pd 
import scipy.stats
import argparse

import torch

from utils import default_paths, nsd_utils, texture_utils
from model_fitting import initialize_fitting
from feature_extraction import fwrf_features
from analyze_features import bent_gabor_bank

try:
    device = initialize_fitting.init_cuda()
except:
    device = 'cpu:0'
print('Found device:')
print(device)

def measure_curvrect_stats(bank, image_brick, batch_size=20, \
                           resize=True):
    
    # image brick should be [n_images x h x w]
    
    n_images = image_brick.shape[0]
    
    n_batches = int(np.ceil(n_images/batch_size))

    bend_values = bank.bend_values
    scale_values = bank.scale_values
    image_size = bank.image_size
    print(image_brick.shape)
    print(image_brick.shape[1:3])
    print(image_size)
    assert(np.all(image_size==np.array(image_brick.shape[1:3])))
    assert(np.mod(image_brick.shape[1],2)==0) # must have even n pixels
    
    curv_score_method1 = np.zeros((n_images,))
    lin_score_method1 = np.zeros((n_images,))   
    
    curv_score_method2 = np.zeros((n_images,))
    rect_score_method2 = np.zeros((n_images,))
    lin_score_method2 = np.zeros((n_images,))
    
    n_feats = (len(bend_values)-1)*len(scale_values)*len(bank.orient_values)
    mean_curv_over_space = np.zeros((n_images,n_feats))
    mean_rect_over_space = np.zeros((n_images,n_feats))    
    mean_lin_over_space = np.zeros((n_images,len(scale_values)*len(bank.orient_values)))
    
    for bb in range(n_batches):

        batch_inds = np.arange(batch_size*bb, np.min([batch_size*(bb+1), n_images]))

        image_batch = np.moveaxis(image_brick[batch_inds,:,:], [0,1,2], [2,0,1])
        
        print('processing images w filter bank')
        sys.stdout.flush()
        st = time.time()
        all_curv_filt_coeffs = bank.filter_image_batch_pytorch(image_batch, which_kernels='curv')
        all_rect_filt_coeffs = bank.filter_image_batch_pytorch(image_batch, which_kernels='rect')
        all_lin_filt_coeffs = bank.filter_image_batch_pytorch(image_batch, which_kernels='linear')
        
        elapsed = time.time() - st
        print('took %.5f sec to process batch of %d images (image size %d pix)'\
                  %(elapsed, len(batch_inds), image_batch.shape[0]))
 
        print('computing summary stats')
        # Compute some summary stats (trying to give many options here)
        max_curv_images = np.max(all_curv_filt_coeffs, axis=2)
        max_rect_images = np.max(all_rect_filt_coeffs, axis=2)
        max_lin_images = np.max(all_lin_filt_coeffs, axis=2)
        
        # method 1 - compare curved filters vs linear filters.
        unique_curv_inds = max_curv_images>max_lin_images
        unique_lin_inds = max_lin_images>max_curv_images
        
        unique_curv_ims = copy.deepcopy(max_curv_images)
        unique_curv_ims[~unique_curv_inds] = 0.0        
        unique_lin_ims = copy.deepcopy(max_lin_images)
        unique_lin_ims[~unique_lin_inds] = 0.0

        curv_score_method1[batch_inds] = np.mean(np.mean(unique_curv_ims, axis=0), axis=0);
        lin_score_method1[batch_inds] = np.mean(np.mean(unique_lin_ims, axis=0), axis=0);
        
        # method 2 - compare curved against both angular (rect) and linear filters.
        unique_curv_inds = ((max_curv_images>max_rect_images) & (max_curv_images>max_lin_images))
        unique_rect_inds = ((max_rect_images>max_curv_images) & (max_rect_images>max_lin_images))
        unique_lin_inds = ((max_lin_images>max_curv_images) & (max_lin_images>max_rect_images))

        unique_curv_ims = copy.deepcopy(max_curv_images)
        unique_curv_ims[~unique_curv_inds] = 0.0
        unique_rect_ims = copy.deepcopy(max_rect_images)
        unique_rect_ims[~unique_rect_inds] = 0.0
        unique_lin_ims = copy.deepcopy(max_lin_images)
        unique_lin_ims[~unique_lin_inds] = 0.0

        curv_score_method2[batch_inds] = np.mean(np.mean(unique_curv_ims, axis=0), axis=0);
        rect_score_method2[batch_inds] = np.mean(np.mean(unique_rect_ims, axis=0), axis=0);
        lin_score_method2[batch_inds] = np.mean(np.mean(unique_lin_ims, axis=0), axis=0);

        # averaging power over image dimensions        
        mean_curv_over_space[batch_inds,:] = np.mean(np.mean(all_curv_filt_coeffs, axis=0), axis=0).T
        mean_rect_over_space[batch_inds,:] = np.mean(np.mean(all_rect_filt_coeffs, axis=0), axis=0).T
        mean_lin_over_space[batch_inds,:] = np.mean(np.mean(all_lin_filt_coeffs, axis=0), axis=0).T
 
    curvrect = {'curv_score_method1': curv_score_method1, 
                'lin_score_method1': lin_score_method1, 
                'curv_score_method2': curv_score_method2, 
                 'rect_score_method2': rect_score_method2, 
                 'lin_score_method2': lin_score_method2,                 
                 'mean_curv_over_space': mean_curv_over_space,
                 'mean_rect_over_space': mean_rect_over_space,
                 'mean_lin_over_space': mean_lin_over_space,  
                 'curv_kernel_pars': bank.curv_kernel_pars, 
                 'rect_kernel_pars': bank.rect_kernel_pars, 
                 'lin_kernel_pars': bank.lin_kernel_pars}
    
    return curvrect

def measure_sketch_tokens_top_ims_curvrect(debug=False, which_prf_grid=5, batch_size=20):
    
    freq_values_cyc_per_pix = [0.5, 0.25, 0.125, 0.0625, 0.03125]
    bend_values = [0, 0.04, 0.08, 0.16, 0.32, 0.64]
    orient_values = np.linspace(0,np.pi*2, 9)[0:8]
    
    if debug:
        subjects = [1]
    else:
        subjects = np.arange(1,9)
    path_to_load = default_paths.sketch_token_feat_path
    feat_loaders = [fwrf_features.fwrf_feature_loader(subject=ss,\
                                which_prf_grid=which_prf_grid, \
                                feature_type='sketch_tokens',\
                                use_pca_feats = False) for ss in subjects]
    n_features = feat_loaders[0].max_features
    
    val_inds_ss = np.array(nsd_utils.get_subj_df(subject=1)['shared1000'])
    trninds_ss = np.where(val_inds_ss==False)[0]
        
    ims_list = []
    for ss in subjects:
        images = nsd_utils.get_image_data(subject=ss)
        images = images[trninds_ss,:,:,:]
        image_size = images.shape[2:4]
        images = nsd_utils.image_uncolorize_fn(images)
        ims_list.append(images)

    prf_models = initialize_fitting.get_prf_models(which_prf_grid) 

    # compute bounding boxes for each pRF
    bboxes = np.array([ texture_utils.get_bbox_from_prf(prf_models[mm,:], \
                                   image_size, n_prf_sd_out=2, \
                                   min_pix=None, verbose=False, \
                                   force_square=True) \
                for mm in range(prf_models.shape[0]) ])
    # to make this more efficient, only going to analyze a sub-set of the pRFs here.
    # choosing some that are evenly spaced in angle/eccen/size. 
    prf_eccens = np.round(np.sqrt(prf_models[:,0]**2 + prf_models[:,1]**2),2)
    prf_angles = np.round(np.mod(np.arctan2(prf_models[:,1],prf_models[:,0])*180/np.pi, 360),1)
    prf_sizes = np.round(prf_models[:,2],2)

    angles_do = np.unique(prf_angles)[0:-1:2]
    eccens_do = np.unique(prf_eccens)[0:-2:2]
    sizes_do = np.unique(prf_sizes)[0:-1:2]

    prf_inds_use = np.array([[[np.where((prf_angles==aa) \
                            & (prf_eccens==ee) \
                            & (prf_sizes==ss))[0][0] \
                            for aa in angles_do] for ee in eccens_do] for ss in sizes_do]).ravel()

    bboxes_use = bboxes[prf_inds_use,:]
    # make sure there are no duplicate bounding boxes (duplicate bboxes can occur even when
    # prfs are not actually identical)
    bboxes_unique, unique_inds = np.unique(bboxes_use, axis=0, return_index=True)
    prf_inds_use = prf_inds_use[unique_inds]
    # put the bigger pRFs first, in case they cause out of memory errors 
    prf_inds_use = np.flip(np.sort(prf_inds_use))

    n_prfs_use = len(prf_inds_use)
    
    print('There are %d unique pRF bounding boxes'%n_prfs_use)
    print('First 5 bboxes are:')
    print(bboxes[prf_inds_use[0:5],:])
    
    fn2save = os.path.join(default_paths.sketch_token_feat_path, 'Sketch_token_feature_curvrect_stats.npy')
    
    top_n_images = 96;
    top_n_each_subj = int(np.ceil(top_n_images/len(subjects)))
 
    curv_score_method1 = np.zeros((top_n_images, n_prfs_use, n_features))
    lin_score_method1 = np.zeros((top_n_images, n_prfs_use, n_features))
    
    curv_score_method2 = np.zeros((top_n_images, n_prfs_use, n_features))
    rect_score_method2 = np.zeros((top_n_images, n_prfs_use, n_features))
    lin_score_method2 = np.zeros((top_n_images, n_prfs_use, n_features))
    
    mean_curv = np.zeros((top_n_images, n_prfs_use, n_features))
    mean_rect = np.zeros((top_n_images, n_prfs_use, n_features))
    mean_lin = np.zeros((top_n_images, n_prfs_use, n_features))
    mean_curv_z = np.zeros((top_n_images, n_prfs_use, n_features))
    mean_rect_z = np.zeros((top_n_images, n_prfs_use, n_features))
    mean_lin_z = np.zeros((top_n_images, n_prfs_use, n_features))
    
    max_curv = np.zeros((top_n_images, n_prfs_use, n_features))
    max_rect = np.zeros((top_n_images, n_prfs_use, n_features))
    max_lin = np.zeros((top_n_images, n_prfs_use, n_features))
    max_curv_z = np.zeros((top_n_images, n_prfs_use, n_features))
    max_rect_z = np.zeros((top_n_images, n_prfs_use, n_features))
    max_lin_z = np.zeros((top_n_images, n_prfs_use, n_features))

    best_curv_kernel = np.zeros((top_n_images, n_prfs_use, n_features))    
    best_rect_kernel = np.zeros((top_n_images, n_prfs_use, n_features))    
    best_lin_kernel = np.zeros((top_n_images, n_prfs_use, n_features))    
    best_curv_kernel_z = np.zeros((top_n_images, n_prfs_use, n_features))    
    best_rect_kernel_z = np.zeros((top_n_images, n_prfs_use, n_features))    
    best_lin_kernel_z = np.zeros((top_n_images, n_prfs_use, n_features))    
          
    for mm, prf_model_index in enumerate(prf_inds_use):
       
        print('Processing pRF %d (loop iter %d of %d)\n'%(prf_model_index, mm, n_prfs_use))
        
        st = time.time()

        bbox = bboxes[prf_model_index,:]
        
        cropped_size = bbox[1]-bbox[0]
        if np.mod(cropped_size,2)!=0:
            # needs to be even n pixels, so shave one pixel off if needed
            cropped_size-=1
            bbox[1] = bbox[1]-1
            bbox[3] = bbox[3]-1
            
        print(bbox)
        print(cropped_size)
        print('\n')
        
        # adjusting the freqs so that they are constant cycles/pixel. 
        # since these images were cropped out of larger images at a fixed size, 
        # want this to be as if we filtered the entire image and then cropped.
        # but it is faster just to filter the crops.
        freq_values_cyc_per_image = np.array(freq_values_cyc_per_pix)*cropped_size
        bank = bent_gabor_bank.bent_gabor_feature_bank(freq_values = freq_values_cyc_per_image, \
                                       bend_values = bend_values, \
                                       orient_values = orient_values, \
                                       image_size=cropped_size, \
                                       device = device)
        
        for ff in range(n_features):
            
            if debug and ff>1:
                continue
            print('Processing feature %d of %d'%(ff, n_features))
            
            # making a stack of images to analyze, across all subs
            top_images_cropped = []
            
            for si, ss in enumerate(subjects):

                # get sketch tokens model response to each image at this pRF position
                feat, _ = feat_loaders[si].load(trninds_ss, prf_model_index=prf_model_index)
                assert(feat.shape[0]==ims_list[si].shape[0])

                # sort in descending order, to get top n images
                sorted_order = np.flip(np.argsort(feat[:,ff]))
            
                top_images = ims_list[si][sorted_order[0:top_n_each_subj],:,:,:]

                # taking just the patch around this pRF, because this is the region that 
                # contributed to computing the sketch tokens feature
                top_cropped = top_images[:,0,bbox[0]:bbox[1], bbox[2]:bbox[3]]

                top_images_cropped.append(top_cropped)
                
            top_images_cropped = np.concatenate(top_images_cropped, axis=0)
                
            assert(top_images_cropped.shape[0]==top_n_images)
            assert(top_images_cropped.shape[2]==cropped_size)
          
            curvrect = measure_curvrect_stats(bank, image_brick=top_images_cropped, \
                                              batch_size=batch_size, \
                                              resize=False, patchnorm=False)
          
            curv_score_method1[:,mm,ff] = curvrect['curv_score_method1']
            lin_score_method1[:,mm,ff] = curvrect['lin_score_method1']
            
            curv_score_method2[:,mm,ff] = curvrect['curv_score_method2']
            rect_score_method2[:,mm,ff] = curvrect['rect_score_method2']
            lin_score_method2[:,mm,ff] = curvrect['lin_score_method2']
            
            mean_curv[:,mm,ff] = np.mean(curvrect['mean_curv_over_space'], axis=1)
            mean_rect[:,mm,ff] = np.mean(curvrect['mean_rect_over_space'], axis=1)
            mean_lin[:,mm,ff] = np.mean(curvrect['mean_lin_over_space'], axis=1)
            
            max_curv[:,mm,ff] = np.max(curvrect['mean_curv_over_space'], axis=1)
            max_rect[:,mm,ff] = np.max(curvrect['mean_rect_over_space'], axis=1)
            max_lin[:,mm,ff] = np.max(curvrect['mean_lin_over_space'], axis=1)

            best_curv_kernel[:,mm,ff] = np.argmax(curvrect['mean_curv_over_space'], axis=1)
            best_rect_kernel[:,mm,ff] = np.argmax(curvrect['mean_rect_over_space'], axis=1)
            best_lin_kernel[:,mm,ff] = np.argmax(curvrect['mean_lin_over_space'], axis=1)

            # try z-scoring, see if stats make more sense
            curv_z = scipy.stats.zscore(curvrect['mean_curv_over_space'], axis=0)
            rect_z = scipy.stats.zscore(curvrect['mean_rect_over_space'], axis=0)
            lin_z = scipy.stats.zscore(curvrect['mean_lin_over_space'], axis=0)
 
            mean_curv_z[:,mm,ff] = np.mean(curv_z, axis=1)
            mean_rect_z[:,mm,ff] = np.mean(rect_z, axis=1)
            mean_lin_z[:,mm,ff] = np.mean(lin_z, axis=1)
            
            max_curv_z[:,mm,ff] = np.max(curv_z, axis=1)
            max_rect_z[:,mm,ff] = np.max(rect_z, axis=1)
            max_lin_z[:,mm,ff] = np.max(lin_z, axis=1)
            
            best_curv_kernel_z[:,mm,ff] = np.argmax(curv_z, axis=1)
            best_rect_kernel_z[:,mm,ff] = np.argmax(rect_z, axis=1)
            best_lin_kernel_z[:,mm,ff] = np.argmax(lin_z, axis=1)
        
        elapsed = time.time() - st;
        print('\nTook %.5f sec to do pRF %d (loop iter %d/%d, patch size %d pix)\n'%\
                              (elapsed, prf_model_index, mm, n_prfs_use, cropped_size))
            
            
        dict2save = {'curv_score_method1': curv_score_method1, \
                 'lin_score_method1': lin_score_method1, \
                 
                 'curv_score_method2': curv_score_method2, \
                 'rect_score_method2': rect_score_method2, \
                 'lin_score_method2': lin_score_method2, \

                 'mean_curv': mean_curv, \
                 'mean_rect': mean_rect, \
                 'mean_lin': mean_lin, \
                 'mean_curv_z': mean_curv_z, \
                 'mean_rect_z': mean_rect_z, \
                 'mean_lin_z': mean_lin_z, \
                 
                 'max_curv': max_curv, \
                 'max_rect': max_rect, \
                 'max_lin': max_lin, \
                 'max_curv_z': max_curv_z, \
                 'max_rect_z': max_rect_z, \
                 'max_lin_z': max_lin_z, \
                 
                 'best_curv_kernel': best_curv_kernel, \
                 'best_rect_kernel': best_rect_kernel, \
                 'best_lin_kernel': best_lin_kernel, \
                 'best_curv_kernel_z': best_curv_kernel_z, \
                 'best_rect_kernel_z': best_rect_kernel_z, \
                 'best_lin_kernel_z': best_lin_kernel_z, \
                   
                }
    
        print('saving to %s'%fn2save)
        np.save(fn2save, dict2save)   

def run_test():
     
    image_folder = '/lab_data/tarrlab/common/datasets/NSD_images/images/';
    
    subject_df = nsd_utils.get_subj_df(1);
    n_images = 1000
    image_inds = np.random.choice(np.arange(0,10000), n_images, replace=False)
    coco_id = np.array(subject_df['cocoId'])[image_inds]
    file_list = [os.path.join(image_folder,'%d.jpg'%cid) for cid in coco_id]

    
    # save_filename = os.path.join(default_paths.sketch_token_feat_path, 'test_random_cocoims_curv_rect_values.csv')
    fn2save = os.path.join(default_paths.sketch_token_feat_path, 'test_random_cocoims_curv_rect_values_wlocalnorm.npy')
    print('will save to %s\n'%(fn2save)) 
    
    
    image_size = 128;
    scale_values = np.linspace(2,8,4)
    bend_values= [0, 0.04, 0.08, 0.16, 0.32, 0.64]

    bank = bent_gabor_feature_bank(scale_values = scale_values, bend_values=bend_values, image_size=image_size)
    
    curvrect = measure_curvrect_stats(bank, file_list, batch_size=20)

    curvrect['image_inds'] = image_inds
              
    print('saving to %s\n'%(fn2save))    
    np.save(fn2save, curvrect)

    
if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    
    args = parser.parse_args()

    if args.debug:
        print('DEBUG MODE\n')
        
#     run_test()

    measure_sketch_tokens_top_ims_curvrect(debug=args.debug==1)