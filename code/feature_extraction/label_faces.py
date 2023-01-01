import argparse
import numpy as np
import sys, os
import time
import h5py
import torch
import pandas as pd
from src.retinafacetf2.retinaface import RetinaFace

#import custom modules
from utils import nsd_utils, default_paths, prf_utils, segmentation_utils
from model_fitting import initialize_fitting

try:
    device = initialize_fitting.init_cuda()
except:
    device = 'cpu:0'

def get_labels(image_data, debug=False):
    
    use_gpu_nms = False
    nms_thresh = 0.40
    det_thresh = 0.90;
    detector = RetinaFace(use_gpu_nms, nms_thresh)
    
    n_images = image_data.shape[0]
    
    faces_all = []
    landmarks_all = []
    
    n_faces_each = np.zeros((n_images,))
    
    for ii in range(n_images):
        
        if debug and ii>1:
            continue
            
        # prep the image for face detection code
        image = np.moveaxis(image_data[ii,:,:,:],[0],[2])
        image_preproc = np.flip(image, axis=2) # [b,g,r] opencv format
        img = np.ascontiguousarray(image_preproc) # this seems to be needed for opencv

        st = time.time()
        
        faces, landmarks = detector.detect(img, det_thresh)
    
        elapsed = time.time() - st
        print('%d of %d: detected %d faces, took %.5f s'%(ii, n_images, faces.shape[0], elapsed))
        sys.stdout.flush()
        
        faces_all.append(faces)
        landmarks_all.append(landmarks)
        n_faces_each[ii] = faces.shape[0]
        
    return faces_all, landmarks_all, n_faces_each
      
    
def proc_one_subject(subject, args):

    save_labels_path = os.path.join(default_paths.stim_labels_root, 'face_labels_retinaface')
    
    if not os.path.exists(save_labels_path):
        os.makedirs(save_labels_path)
    
    # Load and prepare the image set to work with 
    if subject==999:
        # 999 is a code i am using to indicate the independent set of coco images, which were
        # not actually shown to any NSD participants
        from utils import coco_utils
        image_data = coco_utils.load_indep_coco_images(n_pix=240)
    elif subject==998:
        from utils import coco_utils
        image_data = coco_utils.load_indep_coco_images_big(n_pix=240)
    else: 
        # load all images for the current subject, 10,000 ims
        image_data = nsd_utils.get_image_data(subject)  

    faces_all, landmarks_all, n_faces_each = get_labels(image_data, debug=args.debug)
    
    if args.debug:
        fn2save = os.path.join(save_labels_path, 'S%d_facelabels_DEBUG.npy'%(subject))
    else:
        fn2save = os.path.join(save_labels_path, 'S%d_facelabels.npy'%(subject))
    print('saving to %s'%fn2save)
    np.save(fn2save, {'faces': faces_all, \
                      'landmarks': landmarks_all, \
                      'n_faces_each': n_faces_each})
    # save


def write_binary_face_labels_csv(subject, min_pix = 10, which_prf_grid=5, debug=False):
 
    if subject==999:
        # 999 is a code i am using to indicate the independent set of coco images, which were
        # not actually shown to any NSD participants
        from utils import coco_utils
        subject_df = coco_utils.load_indep_coco_info()  
    elif subject==998:
        from utils import coco_utils
        subject_df = coco_utils.load_indep_coco_info_big()  
    else:
        subject_df = nsd_utils.get_subj_df(subject);
 
    # Params for the spatial aspect of the model (possible pRFs)
    models = initialize_fitting.get_prf_models(which_grid=which_prf_grid)    

    # Get masks for every pRF (circular), in coords of NSD images
    n_prfs = len(models)
    n_pix = 240 # this is the resolution that the face annotations are in, don't change this
    n_prf_sd_out = 2
    prf_masks = np.zeros((n_prfs, n_pix, n_pix))
    
    for prf_ind in range(n_prfs):    
        prf_params = models[prf_ind,:] 
        x,y,sigma = prf_params
        aperture=1.0
        prf_mask = prf_utils.get_prf_mask(center=[x,y], sd=sigma, \
                               patch_size=n_pix)
        prf_masks[prf_ind,:,:] = prf_mask.astype('int')

    # mask_sums = np.sum(np.sum(prf_masks, axis=1), axis=1)
    # min_overlap_pct = 0.20
    # min_pix_req = np.ceil(mask_sums*min_overlap_pct)
    # print(np.unique(min_pix_req))
    min_pix_req = min_pix*np.ones((n_prfs,))
    
    face_labels_path = os.path.join(default_paths.stim_labels_root, 'face_labels_retinaface')
    fn2load = os.path.join(face_labels_path, 'S%d_facelabels.npy'%(subject))   
    f = np.load(fn2load, allow_pickle=True).item()
 
    # first save labels for entire image
    fn2save = os.path.join(default_paths.stim_labels_root, 'S%d_face_binary.csv'%(subject))
    print('saving to %s'%(fn2save))
    face_labels = (f['n_faces_each']>0)
    df = pd.DataFrame(face_labels, columns=['has_face'])
    df.to_csv(fn2save, header=True)
    
    # then making labels for each pRF individually
    n_images = len(f['n_faces_each'])
    n_prfs = len(models)
    face_labels_binary = np.zeros((n_images, n_prfs),dtype=bool)
                           
    for image_ind in range(n_images):

        if debug and image_ind>1:
            continue

        print('Processing image %d of %d'%(image_ind, n_images))

        n_faces = f['faces'][image_ind].shape[0]

        for nn in range(n_faces):

            box = f['faces'][image_ind][nn,:]
            x1,y1,x2,y2,_ = box
            poly = [x1,y1,
                    x1,y2,
                    x2,y2,
                    x2,y1]

            face_mask = segmentation_utils.apply_mask_from_poly(np.ones((n_pix,n_pix)), poly)

            # find where this overlaps with any pRFs
            overlap_pix = np.tensordot(face_mask, prf_masks, [[0,1], [1,2]])
            has_overlap = overlap_pix > min_pix_req

            face_labels_binary[image_ind,has_overlap] = 1; 

            sys.stdout.flush()       
                           
    labels_folder = os.path.join(default_paths.stim_labels_root, 'S%d_within_prf_grid%d'%\
                                 (subject, which_prf_grid))
    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)
        
    for mm in range(n_prfs):
                           
        fn2save = os.path.join(labels_folder, 'S%d_face_binary_prf%d.csv'%(subject, mm))
        print('saving to %s'%(fn2save))
        df = pd.DataFrame({'has_face':face_labels_binary[:,mm]})
        df.to_csv(fn2save, header=True)
                           
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=0,
                    help="number of the subject, 1-8")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    
    args = parser.parse_args()
    
    if args.subject==0:
        args.subject=None
   
    args.debug = (args.debug==1)     
    
    os.chdir(default_paths.retinaface_path)
    
    save_labels_path = os.path.join(default_paths.stim_labels_root, 'face_labels_retinaface')
    fn2save = os.path.join(save_labels_path, 'S%d_facelabels.npy'%(args.subject))
    if not os.path.exists(fn2save):
        proc_one_subject(subject = args.subject, args=args)
    
    write_binary_face_labels_csv(subject = args.subject, debug=args.debug)
    