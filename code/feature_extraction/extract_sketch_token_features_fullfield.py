import numpy as np
import sys, os
import argparse
import torch
import time
import h5py
import gc
import torch.nn as nn

#import custom modules
from utils import prf_utils, torch_utils, texture_utils, default_paths, nsd_utils
from model_fitting import initialize_fitting
from feature_extraction import pca_feats

try:
    device = initialize_fitting.init_cuda()
except:
    device = 'cpu:0'

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def get_features(features_file, 
                  pooling_op, 
                  batch_size=100, 
                  debug=False, device=None):

    if device is None:
        device = 'cpu:0'
        
    with h5py.File(features_file, 'r') as data_set:
        ds_size = data_set['/features'].shape
    n_images = ds_size[3]
    n_feature_channels = ds_size[0]
    map_resolution = ds_size[1]
    # how big will the feature maps be after we do maxpooling?
    map_res_pooled = int(np.floor(map_resolution/pooling_op.kernel_size))
    n_features_total = map_res_pooled**2 * n_feature_channels
    print('number of features total: %d'%n_features_total)
    print('number of images: %d'%n_images)
    
    features = np.zeros((n_images, n_features_total), dtype=np.float32)
    n_batches = int(np.ceil(n_images/batch_size))

    if debug:
        features = np.zeros((10000, n_features_total), dtype=np.float32)
        
    for bb in range(n_batches):

        if debug and bb>1:
            continue

        batch_inds = np.arange(batch_size * bb, np.min([batch_size * (bb+1), n_images]))

        print('Loading features for images [%d - %d]'%(batch_inds[0], batch_inds[-1]))
        st = time.time()
        with h5py.File(features_file, 'r') as data_set:
            # Note this order is reversed from how it was saved in matlab originally.
            # The dimensions go [features x h x w x images]
            # Luckily h and w are swapped matlab to python anyway, so can just switch the first and last.
            values = np.copy(data_set['/features'][:,:,:,batch_inds])
            data_set.close()  
            
        print('values shape:')
        print(values.shape)
        
        fmaps_batch = np.moveaxis(values, [0,1,2,3],[1,2,3,0])

        elapsed = time.time() - st
        print('Took %.5f sec to load feature maps'%elapsed)

        maps_full_field = torch_utils._to_torch(fmaps_batch, device=device)

        maps_pooled = pooling_op(maps_full_field)
        
        maps_reshaped = torch.reshape(maps_pooled, [len(batch_inds), -1])
        
        if bb==0:
            print('size of raw fmaps this batch:')
            print(maps_full_field.shape)
            print('size of pooled fmaps this batch:')
            print(maps_pooled.shape)
            print('size of reshaped fmaps this batch:')
            print(maps_reshaped.shape)
            
        features[batch_inds,:] = maps_reshaped.detach().cpu().numpy()
                
        maps_full_field=None; maps_pooled=None; maps_reshaped=None
        gc.collect()
        torch.cuda.empty_cache()

    print(features.shape)
    
    return features


def proc_one_subject(subject, args):
    
    if args.use_node_storage:
        feat_path = default_paths.sketch_token_feat_path_localnode
    else:
        feat_path = default_paths.sketch_token_feat_path
    if args.debug:      
        feat_path = os.path.join(feat_path,'DEBUG')
     
    path_to_save = os.path.join(feat_path, 'PCA')
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
        
    if subject==999:
        fit_inds = None
    else: 
        subject_df = nsd_utils.get_subj_df(subject)
        fit_inds = np.array(subject_df['shared1000']==False)

    # These params are fixed
    map_resolution = 240
    
    if args.use_avgpool:
        pool_str = '_avgpool'
    else:
        pool_str = '_maxpool'
    
    pool_str += '_poolsize%d'%args.pooling_kernel_size
        
    if args.grayscale:
        features_file = os.path.join(feat_path, \
                            'S%d_features_grayscale_%d.h5py'%(subject, map_resolution))
        filename_save_pca = os.path.join(path_to_save, 'S%d_grayscale_noavg%s_PCA_grid0.h5py'%(subject, pool_str))
        save_weights_filename = os.path.join(path_to_save,'S%d_grayscale_noavg%s_PCA_weights_grid0.npy'%(subject, pool_str)) \
            if args.save_pca_weights else None
        ncomp_filename = os.path.join(path_to_save,'S%d_grayscale_noavg%s_PCA_grid0_ncomp.npy'%(subject, pool_str)) \
            if args.use_saved_ncomp else None
    else:
        features_file = os.path.join(feat_path, \
                            'S%d_features_%d.h5py'%(subject, map_resolution))
        filename_save_pca = os.path.join(path_to_save, 'S%d_noavg%s_PCA_grid0.h5py'%(subject, pool_str))
        save_weights_filename = os.path.join(path_to_save,'S%d_noavg%s_PCA_weights_grid0.npy'%(subject, pool_str)) \
            if args.save_pca_weights else None
        ncomp_filename = os.path.join(path_to_save,'S%d_noavg%s_PCA_grid0_ncomp.npy'%(subject, pool_str)) \
            if args.use_saved_ncomp else None
        
    if not os.path.exists(features_file):
        raise RuntimeError('Looking at %s for precomputed features, not found.'%features_file)
        
    ks = args.pooling_kernel_size
    if args.use_avgpool:
        pooling_op = torch.nn.AvgPool2d(kernel_size=ks, stride=ks, padding=0)
    else:
        pooling_op = torch.nn.MaxPool2d(kernel_size=ks, stride=ks, padding=0)

    print('pooling op:')
    print(pooling_op)
    
    features_raw = get_features(features_file, pooling_op, \
                            batch_size=args.batch_size, 
                            debug=args.debug, device=device)

    print(features_raw.shape)
    print(fit_inds.shape)
    pca_feats.run_pca_oneprf(features_raw, 
                                filename_save_pca, 
                                fit_inds=fit_inds, 
                                min_pct_var=args.min_pct_var, 
                                max_pc_to_retain=args.max_pc_to_retain,
                                save_weights=args.save_pca_weights==1, 
                                save_weights_filename=save_weights_filename, 
                                use_saved_ncomp=args.use_saved_ncomp, 
                                ncomp_filename=ncomp_filename,\
                                save_dtype=np.float32, compress=True, \
                                debug=args.debug)
    
    if args.rm_big==1:
        
        edges_file = features_file.split('_features_')[0] + '_edges_' + features_file.split('_features_')[1]
        if os.path.exists(features_file):
            print('removing raw file from %s'%features_file)
            os.remove(features_file)
        if os.path.exists(edges_file):
            print('removing raw file from %s'%edges_file)
            os.remove(edges_file)

        print('done removing')

def proc_other_image_set(image_set, args):
    
    if args.use_node_storage:
        feat_path = default_paths.sketch_token_feat_path_localnode
    else:
        feat_path = default_paths.sketch_token_feat_path
    if args.debug:      
        feat_path = os.path.join(feat_path,'DEBUG')
     
    path_to_save = os.path.join(feat_path, 'PCA')
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
        
    fit_inds = None
    
    # These params are fixed
    map_resolution = 240
    
    if args.use_avgpool:
        pool_str = '_avgpool'
    else:
        pool_str = '_maxpool'
    
    pool_str += '_poolsize%d'%args.pooling_kernel_size
    
        
    if args.grayscale:
        features_file = os.path.join(feat_path, \
                            '%s_features_grayscale_%d.h5py'%(image_set, map_resolution))
        
    else:
        features_file = os.path.join(feat_path, \
                            '%s_features_%d.h5py'%(image_set, map_resolution))
        
    if not os.path.exists(features_file):
        raise RuntimeError('Looking at %s for precomputed features, not found.'%features_file)
        
    ks = args.pooling_kernel_size
    if args.use_avgpool:
        pooling_op = torch.nn.AvgPool2d(kernel_size=ks, stride=ks, padding=0)
    else:
        pooling_op = torch.nn.MaxPool2d(kernel_size=ks, stride=ks, padding=0)

    print('pooling op:')
    print(pooling_op)
    
    features_raw = get_features(features_file, pooling_op, \
                            batch_size=args.batch_size, 
                            debug=args.debug, device=device)

    print('done getting raw features')
    sys.stdout.flush()

    subjects_pca=np.arange(1,9)

    for ss in subjects_pca:

        if args.grayscale:
            load_weights_filename = os.path.join(path_to_save,'S%d_grayscale_noavg%s_PCA_weights_grid0.npy'%(ss, pool_str))
            filename_save_pca = os.path.join(path_to_save, \
                                             '%s_grayscale_noavg%s_PCA_wtsfromS%d_grid0.h5py'%(image_set, pool_str, ss))
        else:
            load_weights_filename = os.path.join(path_to_save,'S%d_noavg%s_PCA_weights_grid0.npy'%(ss, pool_str))
            filename_save_pca = os.path.join(path_to_save, \
                                             '%s_noavg%s_PCA_wtsfromS%d_grid0.h5py'%(image_set, pool_str, ss))
        pca_feats.apply_pca_oneprf(features_raw, 
                                    filename_save_pca, 
                                    load_weights_filename=load_weights_filename, 
                                    save_dtype=np.float32, compress=True, \
                                    debug=args.debug)
    
    if args.rm_big==1:
        
        edges_file = features_file.split('_features_')[0] + '_edges_' + features_file.split('_features_')[1]
        if os.path.exists(features_file):
            print('removing raw file from %s'%features_file)
            os.remove(features_file)
        if os.path.exists(edges_file):
            print('removing raw file from %s'%edges_file)
            os.remove(edges_file)
        
        print('done removing')
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=0,
                    help="number of the subject, 1-8")
    parser.add_argument("--image_set", type=str,default='none',
                    help="name of the image set to use (if not an NSD subject)")
    parser.add_argument("--use_node_storage", type=int,default=0,
                    help="want to save and load from scratch dir on current node? 1 for yes, 0 for no")
    parser.add_argument("--batch_size", type=int,default=100,
                    help="batch size? default 100")
    parser.add_argument("--pooling_kernel_size", type=int,default=4,
                    help="pooling_kernel size? default 4")
    parser.add_argument("--use_avgpool", type=int,default=0,
                    help="want to use avg pooling? if false, use maxpool")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    parser.add_argument("--max_pc_to_retain", type=int,default=0,
                    help="max pc to retain? enter 0 for None")
    parser.add_argument("--min_pct_var", type=int,default=95,
                    help="min pct var to explain? default 95")
    parser.add_argument("--save_pca_weights", type=int,default=0,
                    help="want to save the weights to reproduce the pca later? 1 for yes, 0 for no")
    parser.add_argument("--use_saved_ncomp", type=int,default=0,
                    help="want to use a previously saved number of components? 1 for yes, 0 for no")
    parser.add_argument("--grayscale", type=int,default=0,
                    help="use features computed from grayscale images only? 1 for yes, 0 for no")
    parser.add_argument("--rm_big", type=int,default=0,
                    help="want to remove big feature maps files when done? 1 for yes, 0 for no")

    args = parser.parse_args()
    
    if args.subject==0:
        args.subject=None
    if args.image_set=='none':
        args.image_set=None
                         
    args.debug = (args.debug==1)     
    args.grayscale = (args.grayscale==1)     
    args.use_avgpool = (args.use_avgpool==1)     
    
    if args.subject is not None:
        
        proc_one_subject(subject = args.subject, args=args)
        
    elif args.image_set is not None:
        
        proc_other_image_set(image_set=args.image_set, args=args)
        

        
        