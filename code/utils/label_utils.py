# import basic modules
import sys
import os
import numpy as np
import pandas as pd
import PIL
import copy

from utils import default_paths, nsd_utils, prf_utils, segmentation_utils


def write_binary_labels_csv(subject, stuff=False):
    """
    Creating a csv file where columns are binary labels for the presence/absence of categories
    and supercategories in COCO.
    10,000 images long (same size as image arrays at /user_data/mmhender/nsd_stimuli/stimuli/)
    """
    from utils import coco_utils

    if subject==999:
        # 999 is a code i am using to indicate the independent set of coco images, which were
        # not actually shown to any NSD participants
        subject_df = coco_utils.load_indep_coco_info()      
    elif subject==998:
        # this is the larger set of 50,000 independent images
        subject_df = coco_utils.load_indep_coco_info_big()
    else:
        subject_df = nsd_utils.get_subj_df(subject);
        
    all_coco_ids = np.array(subject_df['cocoId'])

    if stuff:
        coco_object = coco_utils.coco_stuff_val
    else:
        coco_object = coco_utils.coco_val
    cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat = coco_utils.get_coco_cat_info(coco_object)
       
    ims_each_cat, cats_each_image = coco_utils.list_cats_each_image(all_coco_ids, stuff=stuff)
    ims_each_supcat, supcats_each_image = coco_utils.list_supcats_each_image(all_coco_ids, stuff=stuff)

    binary_df = pd.DataFrame(data=np.concatenate([ims_each_supcat, ims_each_cat], axis=1), \
                                 columns = supcat_names + cat_names)

    if not stuff:        
        fn2save = os.path.join(default_paths.stim_labels_root, 'S%d_cocolabs_binary.csv'%subject)
    else:
        fn2save = os.path.join(default_paths.stim_labels_root, 'S%d_cocolabs_stuff_binary.csv'%subject)
        
    print('Saving to %s'%fn2save)
    binary_df.to_csv(fn2save, header=True)
    
    return

def write_binary_labels_csv_within_prf(subject, min_pix = 10, stuff=False, \
                                       which_prf_grid=1, debug=False):
    """
    Creating a csv file where columns are binary labels for the presence/absence of categories
    and supercategories in COCO.
    Analyzing presence of categories for each pRF position separately - make a separate csv file for each prf.
    10,000 images long (same size as image arrays in /nsd/stimuli/)
    """

    from utils import coco_utils

    if subject==999:
        # 999 is a code i am using to indicate the independent set of coco images, which were
        # not actually shown to any NSD participants
        subject_df = coco_utils.load_indep_coco_info()  
    elif subject==998:
        # this is the larger set of 50,000 independent images
        subject_df = coco_utils.load_indep_coco_info_big()
    else:
        subject_df = nsd_utils.get_subj_df(subject);
 
    # Params for the spatial aspect of the model (possible pRFs)
    models = prf_utils.get_prf_models(which_grid=which_prf_grid)    

    # Get masks for every pRF (circular), in coords of NSD images
    n_prfs = len(models)
    n_pix = 425
    n_prf_sd_out = 2
    prf_masks = np.zeros((n_prfs, n_pix, n_pix))
    
    for prf_ind in range(n_prfs):    
        prf_params = models[prf_ind,:] 
        x,y,sigma = prf_params
        aperture=1.0
        prf_mask = prf_utils.get_prf_mask(center=[x,y], sd=sigma, \
                               patch_size=n_pix)
        prf_masks[prf_ind,:,:] = prf_mask.astype('int')
    
    # the number of pixels required to overlap will depend on how many
    # pixels the pRF occupies.
    mask_sums = np.sum(np.sum(prf_masks, axis=1), axis=1)
#     min_pix_req = np.ceil(mask_sums*min_overlap_pct)
    min_pix_req = min_pix*np.ones((n_prfs,))
    
    # Initialize arrays to store all labels for each pRF
    if stuff:
        coco_v = coco_utils.coco_stuff_val
        coco_t = coco_utils.coco_stuff_trn
    else:
        coco_v = coco_utils.coco_val
        coco_t = coco_utils.coco_trn
        
    cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat = coco_utils.get_coco_cat_info(coco_v)

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
            mask_cropped_resized = np.asarray(PIL.Image.fromarray(mask_cropped).resize(newsize,\
                                            resample=PIL.Image.BILINEAR))

            # find where this overlaps with any pRFs
            overlap_pix = np.tensordot(mask_cropped_resized, prf_masks, [[0,1], [1,2]])
            has_overlap = overlap_pix > min_pix_req
            
            if np.any(has_overlap):
                
                cid = annotations[aa]['category_id']
                column_ind = np.where(np.array(cat_ids)==cid)[0][0]
                supcat_column_ind = np.where([np.any(np.isin(ids_each_supcat[sc],cid)) \
                                  for sc in range(n_supcateg)])[0][0]
                
                # Loop over pRFs that overlap
                for prf_ind in np.where(has_overlap)[0]:
                    
                    cat_labels_binary[image_ind, column_ind, prf_ind] = 1                    
                    supcat_labels_binary[image_ind, supcat_column_ind, prf_ind] = 1

        sys.stdout.flush()            
             
                    
    # Now save as csv files for each pRF
    for mm in range(n_prfs):
        
        if debug and mm>1:
            continue

        binary_df = pd.DataFrame(data=np.concatenate([supcat_labels_binary[:,:,mm], \
                                                      cat_labels_binary[:,:,mm]], axis=1), \
                                                      columns = supcat_names + cat_names)
    
        if debug:
            folder2save = os.path.join(default_paths.stim_labels_root, 'DEBUG', \
                                           'S%d_within_prf_grid%d'%(subject, which_prf_grid))
        else:
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
        
        
def make_indoor_outdoor_labels(subject):
    """
    Creating binary labels for indoor/outdoor status of images (inferred based on presence of 
    various categories in coco and coco-stuff).
    """
    from utils import coco_utils

    cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat = \
            coco_utils.get_coco_cat_info(coco_utils.coco_val)

    stuff_cat_objects, stuff_cat_names, stuff_cat_ids, stuff_supcat_names, stuff_ids_each_supcat = \
            coco_utils.get_coco_cat_info(coco_utils.coco_stuff_val)

    
    fn2load = os.path.join(os.path.dirname(os.path.abspath(__file__)),'files','Indoor_outdoor_categ.npy')
    d = np.load(fn2load, allow_pickle=True).item()
    things_keys = np.array(list(d.keys()))
    things_values = np.array(list(d.values()))
    assert(np.all([kk==cc for kk,cc in zip(things_keys, cat_names)]))
     
    fn2load = os.path.join(os.path.dirname(os.path.abspath(__file__)),'files','Indoor_outdoor_stuff_categ.npy')
    d = np.load(fn2load, allow_pickle=True).item()
    stuff_keys = np.array(list(d.keys()))
    stuff_values = np.array(list(d.values()))
    assert(np.all([kk==cc for kk,cc in zip(stuff_keys, stuff_cat_names)]))
          
    fn2load = os.path.join(default_paths.stim_labels_root, 'S%d_cocolabs_binary.csv'%subject)
    coco_df = pd.read_csv(fn2load, index_col = 0)
    cat_labels = np.array(coco_df)[:,12:92]

    indoor_things_sum = np.sum(cat_labels[:,things_values=='indoor'], axis=1)
    outdoor_things_sum = np.sum(cat_labels[:,things_values=='outdoor'], axis=1)
    
    fn2load = os.path.join(default_paths.stim_labels_root, 'S%d_cocolabs_stuff_binary.csv'%subject)
    coco_stuff_df = pd.read_csv(fn2load, index_col=0)
    stuff_cat_labels = np.array(coco_stuff_df)[:,16:108]

    indoor_stuff_sum = np.sum(stuff_cat_labels[:,stuff_values=='indoor'], axis=1)
    outdoor_stuff_sum = np.sum(stuff_cat_labels[:,stuff_values=='outdoor'], axis=1)

    indoor_sum = indoor_things_sum + indoor_stuff_sum
    outdoor_sum = outdoor_things_sum + outdoor_stuff_sum

    has_indoor = indoor_sum>0 
    has_outdoor = outdoor_sum>0 

    conflict = has_indoor & has_outdoor
    conflict_indoor = conflict & (indoor_sum > outdoor_sum)
    conflict_outdoor = conflict & (outdoor_sum > indoor_sum)
   
    # correct the conflicting images that we are able to resolve
    has_outdoor[conflict_indoor] = 0
    has_indoor[conflict_outdoor] = 0
    
    has_indoor = has_indoor.astype(float)
    has_outdoor = has_outdoor.astype(float)

    ambig = (has_indoor+has_outdoor)!=1
    binary_labels = copy.deepcopy(has_outdoor).astype(float)
    binary_labels[ambig] = np.nan
    
    fn2save = os.path.join(default_paths.stim_labels_root, 'S%d_indoor_outdoor.npy'%subject)

    print('Saving to %s'%fn2save)
    np.save(fn2save, {'has_indoor': has_indoor, 'has_outdoor': has_outdoor, \
                      'indoor-outdoor': binary_labels}, allow_pickle=True)

    return

def make_buildings_labels(subject, which_prf_grid):

    from utils import coco_utils

    stuff_cat_objects, stuff_cat_names, stuff_cat_ids, stuff_supcat_names, stuff_ids_each_supcat = \
            coco_utils.get_coco_cat_info(coco_utils.coco_stuff_val) 

    labels_folder = os.path.join(default_paths.stim_labels_root,'S%d_within_prf_grid%d'%(subject, which_prf_grid))
    
    fn2load = os.path.join(os.path.dirname(os.path.abspath(__file__)),'files','Building_stuff_categ.npy')
    d = np.load(fn2load, allow_pickle=True).item()
    assert np.all(np.array(list(d.keys()))==np.array(stuff_cat_names))
    building_cat_inds = [d[kk]==1 for kk in stuff_cat_names]
    
    fn2load = os.path.join(default_paths.stim_labels_root,'S%d_cocolabs_stuff_binary.csv'%(subject))
    coco_stuff_df = pd.read_csv(fn2load, index_col=0)
    stuff_cat_labels = np.array(coco_stuff_df)[:,16:108]
    
    has_building_wholeimage = np.any(stuff_cat_labels[:,building_cat_inds], axis=1).astype(float)
    
    # then pRF-specific labels
    models = prf_utils.get_prf_models(which_grid=which_prf_grid)    
    n_prfs = len(models)
    
    n_images = coco_stuff_df.shape[0]
    has_building = np.zeros((n_images,n_prfs),dtype=float)
    
    for prf_model_index in range(n_prfs):

        fn2load = os.path.join(labels_folder, \
                              'S%d_cocolabs_stuff_binary_prf%d.csv'%(subject, prf_model_index))
        coco_stuff_df = pd.read_csv(fn2load, index_col=0)
        stuff_cat_labels = np.array(coco_stuff_df)[:,16:108]
        
        has_building[:,prf_model_index] = np.any(stuff_cat_labels[:,building_cat_inds], axis=1)
        
    fn2save = os.path.join(default_paths.stim_labels_root, 'S%d_building.npy'%(subject))
    print('Saving to %s'%fn2save)
    np.save(fn2save, {'has_building_wholeimage': has_building_wholeimage, 
                     'has_building': has_building}, allow_pickle=True)

    return

def make_face_labels(subject, which_prf_grid):

    labels_folder = os.path.join(default_paths.stim_labels_root,'S%d_within_prf_grid%d'%(subject, which_prf_grid))
   
    fn2load = os.path.join(default_paths.stim_labels_root, 'S%d_face_binary.csv'%(subject))
    face_df = pd.read_csv(fn2load)
    
    has_face_wholeimage = np.array(face_df['has_face']).astype(float)
    
    # then pRF-specific labels
    models = prf_utils.get_prf_models(which_grid=which_prf_grid)    
    n_prfs = len(models)
    
    n_images = face_df.shape[0]
    has_face = np.zeros((n_images,n_prfs),dtype=float)
    
    for prf_model_index in range(n_prfs):
    
        fn2load = os.path.join(labels_folder, 'S%d_face_binary_prf%d.csv'%(subject, prf_model_index))
        face_df = pd.read_csv(fn2load)
        
        has_face[:,prf_model_index] = face_df['has_face']
        
    fn2save = os.path.join(default_paths.stim_labels_root, 'S%d_face.npy'%(subject))
    print('Saving to %s'%fn2save)
    np.save(fn2save, {'has_face_wholeimage': has_face_wholeimage, 
                     'has_face': has_face}, allow_pickle=True)

    return


def make_realworldsize_labels(subject, which_prf_grid):
    """
    Creating binary labels for real-world-size of image patches (based on approx sizes
    of object categories in coco).
    """

    from utils import coco_utils

    cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat = \
                coco_utils.get_coco_cat_info(coco_utils.coco_val)

    labels_folder = os.path.join(default_paths.stim_labels_root,'S%d_within_prf_grid%d'%(subject, which_prf_grid))
    
    # load a dict of sizes for each category
    fn2save = os.path.join(os.path.dirname(os.path.abspath(__file__)),'files','Realworldsize_categ.npy')
    names_to_sizes = np.load(fn2save, allow_pickle=True).item()
    # make [3 x ncateg] one-hot array of size labels for each categ
    categ_size_labels = np.array([[names_to_sizes[kk]==ii for kk in names_to_sizes.keys()] \
                                  for ii in range(3)])
    # print size lists as a sanity check
    for ii in range(3):
        print('things categories with size %d:'%ii)
        print(np.array(cat_names)[categ_size_labels[ii,:]])
        
        
    # first making labels for entire image
    fn2load = os.path.join(default_paths.stim_labels_root,'S%d_cocolabs_binary.csv'%(subject))
    coco_df = pd.read_csv(fn2load, index_col = 0)
    cat_labels = np.array(coco_df)[:,12:92]
    
    sum_small = np.sum(cat_labels[:,categ_size_labels[0,:]], axis=1)
    sum_medium = np.sum(cat_labels[:,categ_size_labels[1,:]], axis=1)
    sum_large = np.sum(cat_labels[:,categ_size_labels[2,:]], axis=1)
    sums = np.array([sum_small, sum_medium, sum_large]).T
    
    # to resolve conflicts, use the counts in each size group.
    # if two groups have the same count, use both labels (treated as ambiguous)
    less_than_max = sums<np.max(sums, axis=1, keepdims=True)
    sums[less_than_max] = 0

    s = (sums[:,0]>0).astype(float)
    m = (sums[:,1]>0).astype(float)
    l = (sums[:,2]>0).astype(float)

    has_small_wholeimage = s
    has_medium_wholeimage = m
    has_large_wholeimage = l
    
    ambig = (s+l)!=1
    binary_labels_wholeimage = copy.deepcopy(l)
    binary_labels_wholeimage[ambig] = np.nan

    n_images = has_small_wholeimage.shape[0]
    
    # then pRF-specific labels
    models = prf_utils.get_prf_models(which_grid=which_prf_grid)    
    n_prfs = len(models)
    
    has_small = np.zeros((n_images, n_prfs),dtype=float)
    has_medium = np.zeros((n_images, n_prfs),dtype=float)
    has_large = np.zeros((n_images, n_prfs),dtype=float)
    
    binary_labels = np.zeros((n_images, n_prfs),dtype=float)
    
    for prf_model_index in range(n_prfs):
       
        fn2load = os.path.join(labels_folder, \
                          'S%d_cocolabs_binary_prf%d.csv'%(subject, prf_model_index))
        coco_df = pd.read_csv(fn2load, index_col = 0)

        cat_labels = np.array(coco_df)[:,12:92]

        s = np.any(cat_labels[:,categ_size_labels[0,:]], axis=1).astype(float)
        m = np.any(cat_labels[:,categ_size_labels[1,:]], axis=1).astype(float)
        l = np.any(cat_labels[:,categ_size_labels[2,:]], axis=1).astype(float)

        has_small[:,prf_model_index] = s
        has_medium[:,prf_model_index] = m
        has_large[:,prf_model_index] = l
        
        ambig = (s+l)!=1
        binary_labels[:,prf_model_index] = copy.deepcopy(l)
        binary_labels[ambig,prf_model_index] = np.nan


    fn2save = os.path.join(default_paths.stim_labels_root, 'S%d_realworldsize.npy'%(subject))
    print('Saving to %s'%fn2save)
    np.save(fn2save,{'has_small':has_small, 'has_medium': has_medium, 'has_large':has_large,\
                     'small-large': binary_labels,\
                    'has_small_wholeimage': has_small_wholeimage,\
                    'has_medium_wholeimage': has_medium_wholeimage,\
                    'has_large_wholeimage': has_large_wholeimage,
                    'small-large_wholeimage': binary_labels_wholeimage}, allow_pickle=True)
    
    return

def make_animacy_labels(subject, which_prf_grid):
    """
    Creating labels for animacy of objects in image patches
    """

    labels_folder = os.path.join(default_paths.stim_labels_root,'S%d_within_prf_grid%d'%(subject, which_prf_grid))
    
    # first making labels for entire image
    fn2load = os.path.join(default_paths.stim_labels_root,'S%d_cocolabs_binary.csv'%(subject))
    coco_df = pd.read_csv(fn2load, index_col = 0)
    supcat_labels = np.array(coco_df)[:,0:12]
    animate_supcats = np.array([1,9])
    inanimate_supcats = np.array([ii for ii in range(12) if ii not in animate_supcats])
    a = np.any(supcat_labels[:,animate_supcats]==1, axis=1).astype(float)
    i = np.any(supcat_labels[:,inanimate_supcats]==1, axis=1).astype(float)
    
    has_animate_wholeimage = a
    has_inanimate_wholeimage = i

    ambig = (a+i)!=1
    binary_labels_wholeimage = copy.deepcopy(i)
    binary_labels_wholeimage[ambig] = np.nan

    n_images = has_animate_wholeimage.shape[0]
    
    # then pRF-specific labels
    models = prf_utils.get_prf_models(which_grid=which_prf_grid)    
    n_prfs = len(models)
    
    has_animate = np.zeros((n_images, n_prfs),dtype=float)
    has_inanimate = np.zeros((n_images, n_prfs),dtype=float)
    binary_labels = np.zeros((n_images, n_prfs),dtype=float)
    
    for prf_model_index in range(n_prfs):
      
        fn2load = os.path.join(labels_folder,'S%d_cocolabs_binary_prf%d.csv'%(subject, prf_model_index))
        coco_df = pd.read_csv(fn2load, index_col = 0)
        supcat_labels = np.array(coco_df)[:,0:12]
        animate_supcats = np.array([1,9])
        inanimate_supcats = np.array([ii for ii in range(12) if ii not in animate_supcats])
        a = np.any(supcat_labels[:,animate_supcats]==1, axis=1).astype(float)
        i = np.any(supcat_labels[:,inanimate_supcats]==1, axis=1).astype(float)

        has_animate[:,prf_model_index] = a
        has_inanimate[:,prf_model_index] = i

        ambig = (a+i)!=1
        binary_labels[:,prf_model_index] = copy.deepcopy(i)
        binary_labels[ambig,prf_model_index] = np.nan
        
    fn2save = os.path.join(default_paths.stim_labels_root, 'S%d_animacy.npy'%(subject))
    print('Saving to %s'%fn2save)
    np.save(fn2save,{'has_animate': has_animate, \
                     'has_inanimate': has_inanimate,\
                     'animate-inanimate': binary_labels,\
                    'has_animate_wholeimage': has_animate_wholeimage,\
                    'has_inanimate_wholeimage': has_inanimate_wholeimage, \
                    'animate-inanimate_wholeimage': binary_labels_wholeimage}, allow_pickle=True)
    
    return

def load_highlevel_labels_each_prf(subject, which_prf_grid, image_inds, models):

    """
    Load labels for high-level categories
    (binary labels for each axis; if both/neither categ present then label is nan)
    """
    
    discrim_type_list = ['face-building',\
                         'face-none','building-none',\
                         'animate-inanimate','small-large','indoor-outdoor']
    unique_labs_each = [np.arange(2) for dd in discrim_type_list]
    
    n_sem_axes = len(discrim_type_list)
    n_prfs = models.shape[0]
    n_trials = image_inds.shape[0]
    
    labels_all = np.zeros((n_trials, n_sem_axes, n_prfs)).astype(np.float32)
    
    for axis_ind, ax in enumerate(discrim_type_list):
        
        if 'face-none' in ax:
            fn = os.path.join(default_paths.stim_labels_root, 'S%d_face.npy'%(subject))
            d = np.load(fn, allow_pickle=True).item()
            labs = 1-d['has_face']
        elif 'building-none' in ax:
            fn = os.path.join(default_paths.stim_labels_root, 'S%d_building.npy'%(subject))
            d = np.load(fn, allow_pickle=True).item()
            labs = 1-d['has_building']
        elif 'face-building' in ax:
            fn = os.path.join(default_paths.stim_labels_root, 'S%d_face.npy'%(subject))
            d = np.load(fn, allow_pickle=True).item()
            flabs = d['has_face']
            fn = os.path.join(default_paths.stim_labels_root, 'S%d_building.npy'%(subject))
            d = np.load(fn, allow_pickle=True).item()
            blabs = d['has_building']
            binary_labels = copy.deepcopy(blabs)
            ambig = (flabs+blabs)!=1
            binary_labels[ambig] = np.nan
            labs = binary_labels
        elif 'animate-inanimate' in ax:
            fn = os.path.join(default_paths.stim_labels_root, 'S%d_animacy.npy'%(subject))
            d = np.load(fn, allow_pickle=True).item()
            labs = d['animate-inanimate']
        elif 'small-large' in ax:
            fn = os.path.join(default_paths.stim_labels_root, 'S%d_realworldsize.npy'%(subject))
            d = np.load(fn, allow_pickle=True).item()
            labs = d['small-large']
        elif 'indoor-outdoor' in ax:
            fn = os.path.join(default_paths.stim_labels_root, 'S%d_indoor_outdoor.npy'%(subject))
            d = np.load(fn, allow_pickle=True).item()
            labs = np.tile(d['indoor-outdoor'][:,None], [1,n_prfs])
        
        labels_all[:,axis_ind,:] = labs[image_inds,:]
    
    return labels_all, discrim_type_list, unique_labs_each

def load_highlevel_categ_labels_each_prf(subject, which_prf_grid, image_inds, models):

    """
    Load labels for presence of each high-level category
    (these are not binary; an image can have both animate and inanimate label for example)
    """
    categ_list = ['face','building','animate','inanimate','small','large','indoor','outdoor']
    
    n_categ = len(categ_list)
    n_prfs = models.shape[0]
    n_trials = image_inds.shape[0]
    
    labels_all = np.zeros((n_trials, n_categ, n_prfs)).astype(np.float32)

    for axis_ind, ax in enumerate(categ_list):
        
        if 'face' in ax:
            fn = os.path.join(default_paths.stim_labels_root, 'S%d_face.npy'%(subject))
            d = np.load(fn, allow_pickle=True).item()
            labs = d['has_face']
        elif 'building' in ax:
            fn = os.path.join(default_paths.stim_labels_root, 'S%d_building.npy'%(subject))
            d = np.load(fn, allow_pickle=True).item()
            labs = d['has_building']
        elif ax=='animate':
            fn = os.path.join(default_paths.stim_labels_root, 'S%d_animacy.npy'%(subject))
            d = np.load(fn, allow_pickle=True).item()
            labs = d['has_animate']
        elif ax=='inanimate':
            fn = os.path.join(default_paths.stim_labels_root, 'S%d_animacy.npy'%(subject))
            d = np.load(fn, allow_pickle=True).item()
            labs = d['has_inanimate']
        elif 'small' in ax:
            fn = os.path.join(default_paths.stim_labels_root, 'S%d_realworldsize.npy'%(subject))
            d = np.load(fn, allow_pickle=True).item()
            labs = d['has_small']
        elif 'large' in ax:
            fn = os.path.join(default_paths.stim_labels_root, 'S%d_realworldsize.npy'%(subject))
            d = np.load(fn, allow_pickle=True).item()
            labs = d['has_large']
        elif 'indoor' in ax:
            fn = os.path.join(default_paths.stim_labels_root, 'S%d_indoor_outdoor.npy'%(subject))
            d = np.load(fn, allow_pickle=True).item()
            labs = np.tile(d['has_indoor'][:,None], [1, n_prfs])
        elif 'outdoor' in ax:
            fn = os.path.join(default_paths.stim_labels_root, 'S%d_indoor_outdoor.npy'%(subject))
            d = np.load(fn, allow_pickle=True).item()
            labs = np.tile(d['has_outdoor'][:,None], [1, n_prfs])
            
        labels_all[:,axis_ind,:] = labs[image_inds,:]
            
    
    return labels_all, categ_list


def count_highlevel_labels(which_prf_grid=5):

    """
    Count occurences of each high-level semantic label in the entire image set
    for each NSD participant.
    """
    
    models = prf_utils.get_prf_models(which_prf_grid)
    n_prfs = models.shape[0]
    
    subjects = np.array(list(np.arange(1,9)) + [999,998])
    n_subjects = len(subjects)
    n_levels = 3; # levels are [label1, label2, ambiguous]
    
    for si, ss in enumerate(subjects):
    
        if (ss!=999) and (ss!=998):
            image_order = nsd_utils.get_master_image_order()    
            session_inds = nsd_utils.get_session_inds_full()
            sessions = np.arange(nsd_utils.max_sess_each_subj[ss-1])
            inds2use = np.isin(session_inds, sessions) # remove any sessions that weren't shown
            # list of all the image indices shown on each trial
            image_order = image_order[inds2use] 
            # reduce to the ~10,000 unique images
            image_order = np.unique(image_order) 
        elif ss==999:
            image_order = np.arange(10000)
        elif ss==998:
            image_order = np.arange(50000)
        
        print('analyzing counts for S%d, %d images'%(ss, len(image_order)))
        n_trials = len(image_order)
        sys.stdout.flush()
        
        labels, discrim_type_list, unique_labs_each = \
                load_highlevel_labels_each_prf(ss, which_prf_grid, image_order, models)
        axis_names = discrim_type_list
        if si==0:
            n_axes = len(axis_names)
            counts_binary = np.zeros((n_subjects, n_prfs, n_axes, n_levels))
        
        for ai in range(n_axes):
            # labels is [trials x axes x pRFs]
            counts_binary[si, :, ai, 0] = np.sum(labels[:,ai,:]==0, axis=0)
            counts_binary[si, :, ai, 1] = np.sum(labels[:,ai,:]==1, axis=0)
            counts_binary[si, :, ai, 2] = np.sum(np.isnan(labels[:,ai,:]), axis=0)
        
        assert(np.all(np.sum(counts_binary[si,:,:,:], axis=2)==n_trials))
          
        labels, categ_names = \
                load_highlevel_categ_labels_each_prf(ss, which_prf_grid, image_order, models)
        if si==0:
            n_categ = len(categ_names)
            counts_categ = np.zeros((n_subjects, n_prfs, n_categ))
    
        for ci, cc in enumerate(categ_names):
            counts_categ[si, :, ci] = np.sum(labels[:,ci,:], axis=0)

        
    fn2save = os.path.join(default_paths.stim_labels_root, 'Highlevel_counts_all.npy')
    print('saving to %s'%fn2save)
    d = {'subjects': subjects, \
         'counts_binary': counts_binary, \
         'axis_names': axis_names, \
         'counts_categ': counts_categ, \
         'categ_names': categ_names}
    print(d.keys())
    np.save(fn2save, d, allow_pickle=True)

    return 


def count_total_coco_labels(which_prf_grid=5):

    """
    Count total things/stuff occurences in the entire image set
    for each NSD participant.
    """
    
    models = prf_utils.get_prf_models(which_prf_grid)
    n_prfs = models.shape[0]
    
    subjects = np.array(list(np.arange(1,9)) + [999,998])
    n_subjects = len(subjects)
    
    counts_coco_things = np.zeros((n_subjects, n_prfs))
    counts_coco_stuff = np.zeros((n_subjects, n_prfs))
                             
    for si, ss in enumerate(subjects):
    
        if (ss!=999) and (ss!=998):
            image_order = nsd_utils.get_master_image_order()    
            session_inds = nsd_utils.get_session_inds_full()
            sessions = np.arange(nsd_utils.max_sess_each_subj[ss-1])
            inds2use = np.isin(session_inds, sessions) # remove any sessions that weren't shown
            # list of all the image indices shown on each trial
            image_order = image_order[inds2use] 
            # reduce to the ~10,000 unique images
            image_order = np.unique(image_order) 
        elif ss==999:
            image_order = np.arange(10000)
        elif ss==998:
            image_order = np.arange(50000)
        
        print('analyzing counts for S%d, %d images'%(ss, len(image_order)))
        n_trials = len(image_order)
        sys.stdout.flush()
        labels_folder = os.path.join(default_paths.stim_labels_root,\
                                 'S%d_within_prf_grid%d'%(ss, which_prf_grid))
    
        for prf_model_index in range(n_prfs):
      
            fn2load = os.path.join(labels_folder,'S%d_cocolabs_binary_prf%d.csv'%(ss, prf_model_index))
            coco_df = pd.read_csv(fn2load, index_col = 0)
            coco_things_binary = np.array(coco_df)[:,12:92]
            num_things = np.sum(coco_things_binary, axis=1)
        
            counts_coco_things[si,prf_model_index] = np.sum(num_things)
            
            fn2load = os.path.join(labels_folder,'S%d_cocolabs_stuff_binary_prf%d.csv'%(ss, prf_model_index))
            coco_df = pd.read_csv(fn2load, index_col = 0)
            coco_stuff_binary = np.array(coco_df)[:,16:108]
            num_stuff = np.sum(coco_stuff_binary, axis=1)
        
            counts_coco_stuff[si,prf_model_index] = np.sum(num_stuff)
            
        
    fn2save = os.path.join(default_paths.stim_labels_root, 'Coco_counts_all.npy')
    print('saving to %s'%fn2save)
    d = {'subjects': subjects, \
         'counts_coco_things': counts_coco_things, \
         'counts_coco_stuff': counts_coco_stuff}
    print(d.keys())
    np.save(fn2save, d, allow_pickle=True)

    return 



