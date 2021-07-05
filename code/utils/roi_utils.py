import numpy as np
import os
import pandas as pd

ret_group_names = ['V1', 'V2', 'V3','hV4','VO1-2','PHC1-2','LO1-2','TO1-2','V3ab','IPS0-5','SPL1','FEF']
ret_group_inds = [[1,2],[3,4],[5,6],[7],[8,9],[10,11],[14,15],[12,13],[16,17],[18,19,20,21,22,23],[24],[25]]

def load_roi_label_mapping(nsd_root, subject, verbose=False):
    """
    Load files that describe the mapping from numerical labels in NSD ROI definition files, to text labels for the ROIs.
    """
    
    filename_prf = os.path.join(nsd_root,'nsddata','freesurfer','subj%02d'%subject, 'label', 'prf-visualrois.mgz.ctab')
    names = np.array(pd.read_csv(filename_prf))
    names = [str(name) for name in names]
    prf_num_labels = [int(name[2:np.char.find(name,' ')]) for name in names]
    prf_text_labels=[]
    for name in names:
        if np.char.find(name,'\\')>-1:
            prf_text_labels.append(name[np.char.find(name,' ')+1:np.char.find(name,'\\')])
        else:
            prf_text_labels.append(name[np.char.find(name,' ')+1:np.char.find(name,"'",2)])
    if verbose:
        print(prf_num_labels)
        print(prf_text_labels)

    filename_ret = os.path.join(nsd_root,'nsddata','freesurfer','subj%02d'%subject, 'label', 'Kastner2015.mgz.ctab')
    names = np.array(pd.read_csv(filename_ret))
    names = [str(name) for name in names]
    ret_num_labels = [int(name[2:np.char.find(name,' ')]) for name in names]
    ret_text_labels=[]
    for name in names:
        if np.char.find(name,'\\')>-1:
            ret_text_labels.append(name[np.char.find(name,' ')+1:np.char.find(name,'\\')])
        else:
            ret_text_labels.append(name[np.char.find(name,' ')+1:np.char.find(name,"'",2)])
    if verbose:
        print(ret_num_labels)
        print(ret_text_labels)

    # kastner atlas and prf have same values/names for all shared elements - so can just use kastner going forward.
    assert(np.array_equal(prf_num_labels,ret_num_labels[0:len(prf_num_labels)]))
    assert(np.array_equal(prf_text_labels,ret_text_labels[0:len(prf_text_labels)]))

    filename_faces = os.path.join(nsd_root,'nsddata','freesurfer','subj%02d'%subject, 'label', 'floc-faces.mgz.ctab')
    names = np.array(pd.read_csv(filename_faces))
    names = [str(name) for name in names]
    faces_num_labels = [int(name[2:np.char.find(name,' ')]) for name in names]
    faces_text_labels=[]
    for name in names:
        if np.char.find(name,'\\')>-1:
            faces_text_labels.append(name[np.char.find(name,' ')+1:np.char.find(name,'\\')])
        else:
            faces_text_labels.append(name[np.char.find(name,' ')+1:np.char.find(name,"'",2)])
    if verbose:
        print(faces_num_labels)
        print(faces_text_labels)

    filename_places = os.path.join(nsd_root,'nsddata','freesurfer','subj%02d'%subject, 'label', 'floc-places.mgz.ctab')
    names = np.array(pd.read_csv(filename_places))
    names = [str(name) for name in names]
    places_num_labels = [int(name[2:np.char.find(name,' ')]) for name in names]
    places_text_labels=[]
    for name in names:
        if np.char.find(name,'\\')>-1:
            places_text_labels.append(name[np.char.find(name,' ')+1:np.char.find(name,'\\')])
        else:
            places_text_labels.append(name[np.char.find(name,' ')+1:np.char.find(name,"'",2)])
    if verbose:
        print(places_num_labels)
        print(places_text_labels)

    return [ret_num_labels, ret_text_labels], [faces_num_labels, faces_text_labels], [places_num_labels, places_text_labels]


def print_overlap(labels1_full, labels2_full, lab1, lab2):
    
    """
    Look through all pairs of ROIs in two different label files, and print any regions that have overlapping voxels."""
    
    lab1_num = lab1[0]
    lab1_text = lab1[1]
    lab2_num = lab2[0]
    lab2_text = lab2[1]

    for li1, lnum1 in enumerate(lab1_num):
        has1 = (labels1_full==lnum1).flatten().astype(bool)   
        for li2, lnum2 in enumerate(lab2_num):
            has2 = (labels2_full==lnum2).flatten().astype(bool) 
            if np.sum(has1 & has2)>0:
                print('%s and %s:'%(lab1_text[li1],lab2_text[li2]))
                print(' %d vox of overlap'%np.sum(has1 & has2))


                
                
# roi_map = {1: 'V1v', 2: 'V1d', 3: 'V2v', 4: 'V2d', 5: 'V3v', 6: 'V3d', 7: 'hV4', 8: 'VO1', 9: 'VO2', \
#            10: 'PHC1', 11: 'PHC2', 12: 'MST', 13: 'hMT', 14: 'LO2', 15: 'LO1', 16: 'V3b', 17: 'V3a', \
#            18: 'IPS0', 19: 'IPS1', 20: 'IPS2', 21: 'IPS3', 22: 'IPS4', 23: 'IPS5', 24: 'SPL1', 25: 'FEF',\
#            0: 'other'}

# def iterate_roi(group, voxelroi, roimap, group_name=None):
#     for k,g in enumerate(group):
#         g_name = ('' if group_name is None else group_name[k])
#         mask = np.zeros(shape=voxelroi.shape, dtype=bool)
#         for i,roi in enumerate(g):
#             if group_name is None:
#                 g_name += roimap[roi] + ('-' if i+1<len(g) else '')
#             mask = np.logical_or(mask, voxelroi==roi)
#         yield mask, g_name