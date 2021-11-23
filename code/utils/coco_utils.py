# import basic modules
import sys
import os
import numpy as np
import pandas as pd
import PIL

from utils import default_paths, nsd_utils, prf_utils, segmentation_utils
from model_fitting import initialize_fitting

# import coco api tools
sys.path.append(os.path.join(default_paths.coco_api_path,'cocoapi','PythonAPI'))
from pycocotools.coco import COCO

def init_coco(coco_api_path=None):
    
    if coco_api_path is None:
        coco_api_path = default_paths.coco_api_path
        
    annot_file_val = os.path.join(coco_api_path, 'annotations','instances_val2017.json')
    coco_val = COCO(annot_file_val)
    annot_file_trn = os.path.join(coco_api_path, 'annotations', 'instances_train2017.json')
    coco_trn = COCO(annot_file_trn)
    
    return coco_trn, coco_val

print('Initializing coco api...')
coco_trn, coco_val = init_coco()

def init_coco_stuff(coco_api_path=None):
    
    if coco_api_path is None:
        coco_api_path = default_paths.coco_api_path
        
    annot_file_val = os.path.join(coco_api_path, 'annotations','stuff_val2017.json')
    coco_stuff_val = COCO(annot_file_val)
    annot_file_trn = os.path.join(coco_api_path, 'annotations', 'stuff_train2017.json')
    coco_stuff_trn = COCO(annot_file_trn)
    
    return coco_stuff_trn, coco_stuff_val

print('Initializing coco api...')
coco_stuff_trn, coco_stuff_val = init_coco_stuff()

def get_coco_cat_info(coco_object=None):
    
    """ 
    Get lists of all the category names, ids, supercategories in COCO.
    """
    
    if coco_object is None:
        coco_object = coco_val
        
    cat_objects = coco_object.loadCats(coco_object.getCatIds())
    cat_names=[cat['name'] for cat in cat_objects]   
    cat_ids=[cat['id'] for cat in cat_objects]

    supcat_names = list(set([cat['supercategory'] for cat in cat_objects]))
    supcat_names.sort()

    ids_each_supcat = []
    for sc in range(len(supcat_names)):
        this_supcat = [supcat_names[sc]==cat['supercategory'] for cat in cat_objects]
        ids = [cat_objects[ii]['id'] for ii in range(len(cat_names)) if this_supcat[ii]==True]
        ids_each_supcat.append(ids)

    return cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat
   
def get_ims_in_supcat(supcat_name, coco_ids, stuff=False):
    
    """
    For a given supercategory name, find all the images in 'coco_ids' that include an annotation of that super-category.
    Return boolean array same size as coco_ids.
    """
    if stuff:
        coco_v = coco_stuff_val
        coco_t = coco_stuff_trn
    else:
        coco_v = coco_val
        coco_t = coco_trn
       
    cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat = get_coco_cat_info(coco_v)
    sc_ind = [ii for ii in range(len(supcat_names)) if supcat_names[ii]==supcat_name]
    assert(len(sc_ind)==1)
    sc_ind=sc_ind[0]
    all_ims_in_supcat_val = np.concatenate([coco_v.getImgIds(catIds = cid) for cid in ids_each_supcat[sc_ind]], axis=0);
    all_ims_in_supcat_trn = np.concatenate([coco_t.getImgIds(catIds = cid) for cid in ids_each_supcat[sc_ind]], axis=0);
    all_ims_in_supcat = np.concatenate((all_ims_in_supcat_val, all_ims_in_supcat_trn), axis=0)
    
    ims_in_supcat = np.isin(coco_ids, all_ims_in_supcat)
    
    return np.squeeze(ims_in_supcat)

def list_supcats_each_image(coco_ids, stuff=False):
    
    """
    For all the different super-categories, list which images in coco_ids
    contain an instance of that super-category. 
    Returns a binary matrix, and also a list of which super-cats are in each image.
    """
    
    ims_each_supcat = []
    if stuff:
        coco_object = coco_stuff_val
    else:
        coco_object = coco_val
        
    cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat = get_coco_cat_info(coco_object)

    for sc, scname in enumerate(supcat_names):
        ims_in_supcat = get_ims_in_supcat(scname, coco_ids, stuff=stuff)
        ims_each_supcat.append(ims_in_supcat)
        
    ims_each_supcat = np.array(ims_each_supcat)
    supcats_each_image = [np.where(ims_each_supcat[:,ii])[0] for ii in range(ims_each_supcat.shape[1])]
    ims_each_supcat = ims_each_supcat.astype('int').T
    
    return ims_each_supcat, supcats_each_image

def get_ims_in_cat(cat_name, coco_ids, stuff=False):
    
    """
    For a given category name, find all the images in 'coco_ids' that include an annotation of that super-category.
    Return boolean array same size as coco_ids.
    """
    if stuff:
        coco_v = coco_stuff_val
        coco_t = coco_stuff_trn
    else:
        coco_v = coco_val
        coco_t = coco_trn
        
    cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat = get_coco_cat_info(coco_v)
    cid = [ii for ii in range(len(cat_names)) if cat_names[ii]==cat_name]
    assert(len(cid)==1)
    cid = cid[0]
    all_ims_in_cat_val = coco_v.getImgIds(catIds = cat_ids[cid])
    all_ims_in_cat_trn = coco_t.getImgIds(catIds = cat_ids[cid])
    all_ims_in_cat = np.concatenate((all_ims_in_cat_val, all_ims_in_cat_trn), axis=0)
    
    ims_in_cat = np.isin(coco_ids, all_ims_in_cat)
    
    return np.squeeze(ims_in_cat)

def list_cats_each_image(coco_ids, stuff=False):
    
    """
    For all the different categories, list which images in coco_ids
    contain an instance of that category. 
    Returns a binary matrix, and also a list of which categories are in each image.
    """
        
    ims_each_cat = []
    if stuff:
        coco_object = coco_stuff_val
    else:
        coco_object = coco_val
    cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat = get_coco_cat_info(coco_object)

    for cc, cname in enumerate(cat_names):
        ims_in_cat = get_ims_in_cat(cname, coco_ids, stuff=stuff)
        ims_each_cat.append(ims_in_cat)
        
    ims_each_cat = np.array(ims_each_cat)
    cats_each_image = [np.where(ims_each_cat[:,ii])[0] for ii in range(ims_each_cat.shape[1])]
    ims_each_cat = ims_each_cat.astype('int').T
    
    return ims_each_cat, cats_each_image


def write_binary_labels_csv(subject, stuff=False):
    """
    Creating a csv file where columns are binary labels for the presence/absence of categories
    and supercategories in COCO.
    10,000 images long (same size as image arrays at /user_data/mmhender/nsd_stimuli/stimuli/)
    """
    print('Gathering coco labels for subject %d'%subject)
    subject_df = nsd_utils.get_subj_df(subject);
    all_coco_ids = np.array(subject_df['cocoId'])
    
    if stuff:
        coco_object = coco_stuff_val
    else:
        coco_object = coco_val
    cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat = get_coco_cat_info(coco_object)
       
    ims_each_cat, cats_each_image = list_cats_each_image(all_coco_ids, stuff=stuff)
    ims_each_supcat, supcats_each_image = list_supcats_each_image(all_coco_ids, stuff=stuff)

    binary_df = pd.DataFrame(data=np.concatenate([ims_each_supcat, ims_each_cat], axis=1), \
                                 columns = supcat_names + cat_names)

    if not stuff:        
        animate_supcat_ids = [1,9]
        has_animate = np.array([np.any(np.isin(supcats_each_image[ii], animate_supcat_ids)) \
                                for ii in range(len(supcats_each_image))])
        binary_df['has_animate'] = has_animate.astype('int')

        fn2save = os.path.join(default_paths.stim_labels_root, 'S%d_cocolabs_binary.csv'%subject)
    else:
        fn2save = os.path.join(default_paths.stim_labels_root, 'S%d_cocolabs_stuff_binary.csv'%subject)
        
    print('Saving to %s'%fn2save)
    binary_df.to_csv(fn2save, header=True)
    
    return

def write_binary_labels_csv_within_prf(subject, min_overlap_pix=10, stuff=False, which_prf_grid=1, debug=False):
    """
    Creating a csv file where columns are binary labels for the presence/absence of categories
    and supercategories in COCO.
    Analyzing presence of categories for each pRF position separately - make a separate csv file for each prf.
    10,000 images long (same size as image arrays at /user_data/mmhender/nsd_stimuli/stimuli/)
    """

    subject_df = nsd_utils.get_subj_df(subject);
 
    # Params for the spatial aspect of the model (possible pRFs)
    models = initialize_fitting.get_prf_models(which_grid=which_prf_grid)    

    # Get masks for every pRF (circular), in coords of NSD images
    n_prfs = len(models)
    n_pix = 425
    n_prf_sd_out = 2
    prf_masks = np.zeros((n_prfs, n_pix, n_pix))

    for prf_ind in range(n_prfs):    
        prf_params = models[prf_ind,:] 
        x,y,sigma = prf_params
        aperture=1.0
        prf = prf_utils.gauss_2d(center=[x,y], sd=sigma, \
                               patch_size=n_pix, aperture=aperture, dtype=np.float32)
        # Creating a mask 2 SD from the center
        # cutoff of 0.14 approximates +/-2 SDs
        prf_mask = prf/np.max(prf)>0.14
        prf_masks[prf_ind,:,:] = prf_mask.astype('int')
        
    # Initialize arrays to store all labels for each pRF
    if stuff:
        coco_v = coco_stuff_val
        coco_t = coco_stuff_trn
    else:
        coco_v = coco_val
        coco_t = coco_trn
        
    cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat = get_coco_cat_info(coco_v)

    n_images = len(subject_df)
    n_categ = len(cat_names)
    n_supcateg = len(supcat_names)
    n_prfs = len(models)
    cat_labels_binary = np.zeros((n_images, n_categ, n_prfs))
    supcat_labels_binary = np.zeros((n_images, n_supcateg, n_prfs))

    for image_ind in range(n_images):

        if debug and image_ind>1:
            continue
            
        print('Processing image %d of %d'%(image_ind, n_images))

        # figure out if it's training or val set and use the right coco api dataset
        if np.array(subject_df['cocoSplit'])[image_ind]=='val2017':
            coco = coco_v
            coco_image_dir = '/lab_data/tarrlab/common/datasets/COCO/val2017'
        else:
            coco = coco_t
            coco_image_dir = '/lab_data/tarrlab/common/datasets/COCO/train2017'

        # for this image, figure out where all the annotations are   
        cocoid = np.array(subject_df['cocoId'])[image_ind]
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[cocoid]))
        masks = np.array([coco.annToMask(annotations[aa]) for aa in range(len(annotations))])

        # get image metadata, for this coco id
        img_info = coco.loadImgs(ids=[cocoid])[0]
        # how was the image cropped to get from coco original to NSD?
        crop_box_pixels = segmentation_utils.get_crop_box_pixels(np.array(subject_df['cropBox'])[image_ind], \
                                                                 [img_info['height'], img_info['width']])

        for aa in range(len(annotations)):

            # Adjust this annotation to match the image's size in NSD (crop/resize)
            mask = masks[aa,:,:]
            mask_cropped = mask[crop_box_pixels[0]:crop_box_pixels[1], crop_box_pixels[2]:crop_box_pixels[3]]
            newsize=[n_pix, n_pix]
            mask_cropped_resized = np.asarray(PIL.Image.fromarray(mask_cropped).resize(newsize, resample=PIL.Image.BILINEAR))

            # Loop over pRFs, identify whether there is overlap with the annotation.
            for prf_ind in range(n_prfs):

                prf_mask = prf_masks[prf_ind,:,:]

                has_overlap = np.tensordot(mask_cropped_resized, prf_mask, [[0,1], [0,1]])>min_overlap_pix
                
                if has_overlap:

                    cid = annotations[aa]['category_id']
                    column_ind = np.where(np.array(cat_ids)==cid)[0][0]
                    cat_labels_binary[image_ind, column_ind, prf_ind] = 1
                    supcat_column_ind = np.where([np.any(np.isin(ids_each_supcat[sc],cid)) \
                                  for sc in range(n_supcateg)])[0][0]
                    supcat_labels_binary[image_ind, supcat_column_ind, prf_ind] = 1

        sys.stdout.flush()            
             
                    
    # Now save as csv files for each pRF
    animate_supcat_ids = [1,9]

    for mm in range(n_prfs):
        
        if debug and mm>1:
            continue

        binary_df = pd.DataFrame(data=np.concatenate([supcat_labels_binary[:,:,mm], cat_labels_binary[:,:,mm]], axis=1), \
                                     columns = supcat_names + cat_names)
        if not stuff:
            animate_columns= np.array([supcat_labels_binary[:,animate_supcat_ids[ii],mm] \
                           for ii in range(len(animate_supcat_ids))]).T
            has_animate = np.any(animate_columns, axis=1)
            binary_df['has_animate'] = has_animate.astype('int')

        folder2save = os.path.join(default_paths.stim_labels_root, \
                                           'S%d_within_prf_grid%d'%(subject, which_prf_grid))
        if not os.path.exists(folder2save):
            os.makedirs(folder2save)
        if stuff:
            fn2save =  os.path.join(folder2save,'S%d_cocolabs_stuff_binary_prf%d.csv'%(subject, mm))
        else:
            fn2save =  os.path.join(folder2save,'S%d_cocolabs_binary_prf%d.csv'%(subject, mm))
        print('Saving to %s'%fn2save)
        binary_df.to_csv(fn2save, header=True)
        
        
def write_indoor_outdoor_csv(subject):
    """
    Creating binary labels for indoor/outdoor status of images (inferred based on presence of 
    various categories in coco and coco-stuff).
    """
    
    cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat = \
            get_coco_cat_info(coco_val)

    stuff_cat_objects, stuff_cat_names, stuff_cat_ids, stuff_supcat_names, stuff_ids_each_supcat = \
            get_coco_cat_info(coco_stuff_val)

    must_be_indoor = np.array([0, 0, 0, 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, 0, 0, 0, \
                 0, 1, 1, 1, 1, 1, 1, 0, \
                 1, 1, 1, 0, 1, 1, 1, 1, \
                 1, 0, 1, 1, 0, 0, 1, 1 ])

    must_be_outdoor = np.array([0, 0, 1, 1, 1, 1, 1, 1, 
                            1, 1, 1, 1, 1, 1, 0, 0, \
                            0, 1, 1, 1, 1, 1, 1, 1, \
                            0, 0, 0, 0, 0, 1, 1, 1, \
                            1, 1, 1, 1, 1, 1, 1, 0, \
                            0, 0, 0, 0, 0, 0, 0, 0, \
                            0, 0, 0, 0, 0, 0, 0, 0, \
                            0, 0, 0, 0, 0, 0, 0, 0, \
                            0, 0, 0, 0, 0, 0, 0, 0, \
                            0, 0, 0, 0, 0, 0, 0, 0])

    print('things labels assumed to be indoor:')
    print(np.array(cat_names)[must_be_indoor==1])
    print('things labels assumed to be outdoor:')
    print(np.array(cat_names)[must_be_outdoor==1])
    print('things labels that are ambiguous:')
    print(np.array(cat_names)[(must_be_outdoor==0) & (must_be_indoor==0)])



    stuff_must_be_indoor = np.array([0, 0, 0, 0, 0, 0, 1, 0, \
                           0, 1, 1, 1, 0, 0, 0, 1, \
                           1, 1, 1, 0, 0, 0, 1, 0, \
                           0, 1, 1, 0, 0, 0, 0, 1, \
                           0, 0, 0, 0, 0, 0, 0, 0, \
                           0, 0, 0, 0, 0, 0, 0, 0, \
                           0, 1, 0, 0, 0, 0, 0, 0, \
                           0, 0, 0, 0, 1, 0, 0, 0, \
                           1, 0, 0, 0, 0, 0, 0, 0, \
                           0, 0, 0, 0, 0, 0, 0, 0, \
                           0, 0, 1, 0, 1, 0, 0, 0, \
                           1, 0, 0, 0])

    stuff_must_be_outdoor = np.array([0, 0, 1, 1, 1, 1, 0, 0, \
                            0, 0, 0, 0, 0, 0, 1, 0, \
                            0, 0, 0, 1, 0, 1, 0, 0, \
                            0, 0, 0, 0, 1, 0, 0, 0, \
                            1, 1, 0, 1, 1, 1, 0, 0, \
                            0, 0, 1, 1, 1, 0, 0, 0, \
                            1, 0, 0, 0, 0, 1, 0, 1, \
                            1, 1, 1, 1, 0, 0, 1, 1, \
                            0, 1, 1, 1, 0, 0, 0, 1, \
                            0, 0, 1, 0, 0, 1, 0, 0, \
                            0, 0, 0, 0, 0, 0, 0, 0, \
                            0, 0, 0, 0])

    print('stuff labels assumed to be indoor:')
    print(np.array(stuff_cat_names)[stuff_must_be_indoor==1])
    print('stuff labels assumed to be outdoor:')
    print(np.array(stuff_cat_names)[stuff_must_be_outdoor==1])
    print('stuff labels that are ambiguous:')
    print(np.array(stuff_cat_names)[(stuff_must_be_outdoor==0) & (stuff_must_be_indoor==0)])

    fn2load = os.path.join(default_paths.stim_labels_root, 'S%d_cocolabs_binary.csv'%subject)
    coco_df = pd.read_csv(fn2load, index_col = 0)

    cat_labels = np.array(coco_df)[:,12:92]

    indoor_columns = np.array([cat_labels[:,cc] for cc in np.where(must_be_indoor)[0]]).T
    indoor_inds_things = np.any(indoor_columns, axis=1)
    indoor_sum_things = np.sum(indoor_columns==1, axis=1)

    outdoor_columns = np.array([cat_labels[:,cc] for cc in np.where(must_be_outdoor)[0]]).T
    outdoor_inds_things = np.any(outdoor_columns, axis=1)
    outdoor_sum_things = np.sum(outdoor_columns==1, axis=1)

    np.mean(outdoor_inds_things)

    fn2load = os.path.join(default_paths.stim_labels_root, 'S%d_cocolabs_stuff_binary.csv'%subject)
    coco_stuff_df = pd.read_csv(fn2load, index_col=0)

    stuff_cat_labels = np.array(coco_stuff_df)[:,16:108]

    indoor_columns = np.array([stuff_cat_labels[:,cc] for cc in np.where(stuff_must_be_indoor)[0]]).T
    indoor_inds_stuff = np.any(indoor_columns, axis=1)
    indoor_sum_stuff = np.sum(indoor_columns==1, axis=1)

    outdoor_columns = np.array([stuff_cat_labels[:,cc] for cc in np.where(stuff_must_be_outdoor)[0]]).T
    outdoor_inds_stuff = np.any(outdoor_columns, axis=1)
    outdoor_sum_stuff = np.sum(outdoor_columns==1, axis=1)

    outdoor_all = outdoor_inds_things | outdoor_inds_stuff
    indoor_all = indoor_inds_things | indoor_inds_stuff

    ambiguous = ~outdoor_all & ~indoor_all
    conflict = outdoor_all & indoor_all

    print('\n')
    print('proportion of images that are ambiguous (no indoor or outdoor object annotation):')
    print(np.mean(ambiguous))
    print('proportion of images with conflict (have annotation for both indoor and outdoor object):')
    print(np.mean(conflict))

    conflict_indoor = conflict & (indoor_sum_things+indoor_sum_stuff > outdoor_sum_things+outdoor_sum_stuff)
    conflict_outdoor = conflict & (outdoor_sum_things+outdoor_sum_stuff > indoor_sum_things+indoor_sum_stuff)
    conflict_unresolved = conflict & (indoor_sum_things+indoor_sum_stuff == outdoor_sum_things+outdoor_sum_stuff)
    print('conflicting images that can be resolved to indoor/outdoor/tie based on number annotations:')
    print([np.mean(conflict_indoor), np.mean(conflict_outdoor), np.mean(conflict_unresolved)])

    # correct the conflicting images that we are able to resolve

    outdoor_all[conflict_indoor] = 0
    indoor_all[conflict_outdoor] = 0

    ambiguous = ~outdoor_all & ~indoor_all
    conflict = outdoor_all & indoor_all

    print('\nafter resolving conflicts based on number of annotations:')
    print('proportion of images that are ambiguous (no indoor or outdoor object annotation):')
    print(np.mean(ambiguous))
    print('proportion of images with conflict (have annotation for both indoor and outdoor object):')
    print(np.mean(conflict))


    indoor_outdoor_df = pd.DataFrame({'has_indoor': indoor_all, 'has_outdoor': outdoor_all})
    fn2save = os.path.join(default_paths.stim_labels_root, 'S%d_indoor_outdoor.csv'%subject)

    print('Saving to %s'%fn2save)
    indoor_outdoor_df.to_csv(fn2save, header=True)

    return

def load_labels_each_prf(subject, which_prf_grid, image_inds, models, verbose=False):

    """
    Load csv files containing binary labels for coco images.
    """

    labels_folder = os.path.join(default_paths.stim_labels_root, \
                                     'S%d_within_prf_grid%d'%(subject, which_prf_grid))
    
    print('loading labels from folders at %s and %s (will be slow...)'%\
          (default_paths.stim_labels_root, labels_folder))
    discrim_type_list = ['indoor_outdoor','animacy','person','food','vehicle','animal']
    n_sem_axes = len(discrim_type_list)
    n_trials = image_inds.shape[0]
    n_prfs = models.shape[0]

    labels_all = np.zeros((n_trials, n_sem_axes, n_prfs)).astype(np.float32)

    # get indoor/outdoor labels first, this property is defined across whole images  
    aa=0;
    discrim_type = discrim_type_list[0]
    coco_labels_fn = os.path.join(default_paths.stim_labels_root, 'S%d_indoor_outdoor.csv'%subject)
    coco_df = pd.read_csv(coco_labels_fn, index_col=0)
    ims_to_use = np.sum(np.array(coco_df)==1, axis=1)==1
    labels = np.array(coco_df['has_indoor'])
    labels = labels[image_inds].astype(np.float32)
    labels[~ims_to_use[image_inds]] = np.nan
    labels_all[:,aa,:] = np.tile(labels[:,np.newaxis], [1,n_prfs])
    neach = [np.sum(labels==ll) for ll in np.unique(labels[~np.isnan(labels)])] + [np.sum(np.isnan(labels))]
    if verbose:
        print('n outdoor/n indoor/n ambiguous:')
        print(neach)

    for prf_model_index in range(n_prfs):

        if verbose:
            print('\nProcessing pRF %d of %d'%(prf_model_index, n_prfs))
        coco_labels_fn = os.path.join(labels_folder, \
                                  'S%d_cocolabs_binary_prf%d.csv'%(subject, prf_model_index))
        if verbose:
            print('Reading labels from %s...'%coco_labels_fn)
        coco_df = pd.read_csv(coco_labels_fn, index_col=0)

        for aa,discrim_type in enumerate(discrim_type_list[1:]):
            aa=aa+1
            
            # Gather semantic labels for the images, specific to this pRF position. 
            if discrim_type=='animacy':

                labels = np.array(coco_df['has_animate']).astype(np.float32)
                labels = labels[image_inds]
                labels_all[:,aa,prf_model_index] = labels
                if verbose and (prf_model_index==0):
                    neach = [np.sum(labels==ll) for ll in np.unique(labels[~np.isnan(labels)])] 
                    print('no animate/has animate:')
                    print(neach)

            elif discrim_type=='person' or discrim_type=='food' or discrim_type=='vehicle' or discrim_type=='animal':

                labels = np.array(coco_df[discrim_type]).astype(np.float32)
                labels = labels[image_inds]
                labels_all[:,aa,prf_model_index] = labels
                if verbose and (prf_model_index==0):
                    neach = [np.sum(labels==ll) for ll in np.unique(labels[~np.isnan(labels)])]                 
                    print('no %s/has %s:'%(discrim_type,discrim_type))
                    print(neach)
    
    return labels_all