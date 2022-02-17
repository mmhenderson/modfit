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

