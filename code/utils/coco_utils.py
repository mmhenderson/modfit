# import basic modules
import sys
import os
import numpy as np

from utils import default_paths

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

def get_coco_cat_info(coco_object):
    
    cat_objects = coco_object.loadCats(coco_object.getCatIds())
    cat_names=[cat['name'] for cat in cat_objects]   
    cat_ids=[cat['id'] for cat in cat_objects]

    supcat_names = list(set([cat['supercategory'] for cat in cat_objects]))

    ids_each_supcat = []
    for sc in range(len(supcat_names)):
        this_supcat = [supcat_names[sc]==cat['supercategory'] for cat in cat_objects]
        ids = [cat_objects[ii]['id'] for ii in range(len(cat_names)) if this_supcat[ii]==True]
        ids_each_supcat.append(ids)

    return cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat
   
def get_ims_in_supcat(coco_trn, coco_val, supcat_name, all_coco_ids):
    
    """
    For a given supercategory name, find all the images in 'all_coco_ids' that include an annotation of that super-category.
    """
    
    cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat = get_coco_cat_info(coco_val)
    sc_ind = [ii for ii in range(len(supcat_names)) if supcat_names[ii]==supcat_name]
    assert(len(sc_ind)==1)
    sc_ind=sc_ind[0]
    all_ims_in_supcat_val = np.concatenate([coco_val.getImgIds(catIds = cid) for cid in ids_each_supcat[sc_ind]], axis=0);
    all_ims_in_supcat_trn = np.concatenate([coco_trn.getImgIds(catIds = cid) for cid in ids_each_supcat[sc_ind]], axis=0);
    all_ims_in_supcat = np.concatenate((all_ims_in_supcat_val, all_ims_in_supcat_trn), axis=0)
    
    ims_in_supcat = np.isin(all_coco_ids, all_ims_in_supcat)
    
    return np.squeeze(ims_in_supcat)