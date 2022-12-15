import argparse
import numpy as np
import sys, os
import time
import h5py
import torch
from src.retinafacetf2.retinaface import RetinaFace

#import custom modules
from utils import nsd_utils, default_paths
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
    
    proc_one_subject(subject = args.subject, args=args)
    