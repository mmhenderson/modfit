import numpy as np
import os, sys
import pandas as pd
import nibabel as nib

ret_group_names = ['V1', 'V2', 'V3','hV4','VO1-2','PHC1-2','LO1-2','TO1-2','V3ab','IPS0-5','SPL1','FEF']
ret_group_inds = [[1,2],[3,4],[5,6],[7],[8,9],[10,11],[14,15],[12,13],[16,17],[18,19,20,21,22,23],[24],[25]]

from utils import default_paths
from utils import nsd_utils
from utils.nsd_utils import load_from_nii, load_from_mgz

def get_paths():      
    return default_paths.nsd_root, default_paths.stim_root, default_paths.beta_root

nsd_root, stim_root, beta_root = get_paths()

def load_roi_label_mapping(subject, verbose=False):
    """
    Load files (ctab) that describe the mapping from numerical labels to text labels for the ROIs.
    These correspond to the mask definitions of each type of ROI (either nii or mgz files).
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
                
     
def get_voxel_roi_info(subject, volume_space=True):

    """
    For a specified subject, load all definitions of all ROIs for this subject.
    The ROIs included here are retinotopic visual regions (defined using a combination of Kastner 2015 atlas
    and pRF mapping data), and category-selective (face and place) ROIs.
    Will return two separate bricks of labels - one for the retinotopic and one for the category-selective labels. 
    These are partially overlapping, so can choose later which definition to use for the overlapping voxels.
    Can be done in either volume space (volume_space=True) or surface space (volume_space=False).
    If surface space, then each voxel is a "vertex" of mesh.
    
    """
   
     # First loading each ROI definitions file - lists nvoxels long, with diff numbers for each ROI.
    if volume_space:

        roi_path = os.path.join(nsd_root, 'nsddata', 'ppdata', 'subj%02d'%subject, 'func1pt8mm', 'roi')
       
        print('\nVolume space: ROI defs are located at: %s\n'%roi_path)

        prf_labels_full  = load_from_nii(os.path.join(roi_path, 'prf-visualrois.nii.gz'))
        # save the shape, so we can project back to volume space later.
        brain_nii_shape = np.array(prf_labels_full.shape)
        prf_labels_full = prf_labels_full.flatten()

        kast_labels_full = load_from_nii(os.path.join(roi_path, 'Kastner2015.nii.gz')).flatten()
        face_labels_full = load_from_nii(os.path.join(roi_path, 'floc-faces.nii.gz')).flatten()
        place_labels_full = load_from_nii(os.path.join(roi_path, 'floc-places.nii.gz')).flatten()
        
        # Masks of ncsnr values for each voxel 
        ncsnr_full = load_from_nii(os.path.join(beta_root, 'subj%02d'%subject, 'func1pt8mm', \
                                                'betas_fithrf_GLMdenoise_RR', 'ncsnr.nii.gz')).flatten()

    else:
        
        roi_path = os.path.join(nsd_root,'nsddata', 'freesurfer', 'subj%02d'%subject, 'label')

        print('\nSurface space: ROI defs are located at: %s\n'%roi_path)
        
        # Surface space, concatenate the two hemispheres
        # always go left then right, to match the data which also gets concatenated same way
        prf_labs1 = load_from_mgz(os.path.join(roi_path, 'lh.prf-visualrois.mgz'))[:,0,0]
        prf_labs2 = load_from_mgz(os.path.join(roi_path, 'rh.prf-visualrois.mgz'))[:,0,0]
        prf_labels_full = np.concatenate((prf_labs1, prf_labs2), axis=0)

        kast_labs1 = load_from_mgz(os.path.join(roi_path, 'lh.Kastner2015.mgz'))[:,0,0]
        kast_labs2 = load_from_mgz(os.path.join(roi_path, 'rh.Kastner2015.mgz'))[:,0,0]
        kast_labels_full = np.concatenate((kast_labs1, kast_labs2), axis=0)

        face_labs1 = load_from_mgz(os.path.join(roi_path, 'lh.floc-faces.mgz'))[:,0,0]
        face_labs2 = load_from_mgz(os.path.join(roi_path, 'rh.floc-faces.mgz'))[:,0,0]
        face_labels_full = np.concatenate((face_labs1, face_labs2), axis=0)

        place_labs1 = load_from_mgz(os.path.join(roi_path, 'lh.floc-places.mgz'))[:,0,0]
        place_labs2 = load_from_mgz(os.path.join(roi_path, 'rh.floc-places.mgz'))[:,0,0]
        place_labels_full = np.concatenate((place_labs1, place_labs2), axis=0)
  
        # Masks of ncsnr values for each voxel 
        n1 = load_from_mgz(os.path.join(beta_root, 'subj%02d'%subject, 'nativesurface', \
                                                'betas_fithrf_GLMdenoise_RR', 'lh.ncsnr.mgh')).flatten()
        n2 = load_from_mgz(os.path.join(beta_root, 'subj%02d'%subject, 'nativesurface', \
                                                'betas_fithrf_GLMdenoise_RR', 'rh.ncsnr.mgh')).flatten()
        ncsnr_full = np.concatenate((n1, n2), axis=0)
  
        brain_nii_shape = None

    # boolean masks of which voxels had definitions in each of these naming schemes
    has_prf_label = (prf_labels_full>0).astype(bool)
    has_kast_label = (kast_labels_full>0).astype(bool)
    has_face_label = (face_labels_full>0).astype(bool)
    has_place_label = (place_labels_full>0).astype(bool)

    # To combine all regions, first starting with the kastner atlas for retinotopic ROIs.
    roi_labels_retino = np.copy(kast_labels_full)
    print('%d voxels of overlap between kastner and prf definitions, using prf defs'%np.sum(has_kast_label & has_prf_label))
    # Partially overwrite these defs with prf defs, which are more accurate when they exist.
    roi_labels_retino[has_prf_label] = prf_labels_full[has_prf_label]
    print('unique values in retino labels:')
    print(np.unique(roi_labels_retino))

    # Next, re-numbering the face/place ROIs so that they have unique numbers not overlapping w retino...
    max_ret_label = np.max(roi_labels_retino)
    face_labels_renumbered = np.copy(face_labels_full)
    face_labels_renumbered[has_face_label] = face_labels_renumbered[has_face_label] + max_ret_label
    max_face_label = np.max(face_labels_renumbered)
    place_labels_renumbered = np.copy(place_labels_full)
    place_labels_renumbered[has_place_label] = place_labels_renumbered[has_place_label] + max_face_label

    # Now going to make a separate volume for labels of the category-selective ROIs. 
    # These overlap with the retinotopic defs quite a bit, so want to save both for greater flexibility later on.
    roi_labels_categ = np.copy(face_labels_renumbered)
    print('%d voxels of overlap between face and place definitions, using place defs'%np.sum(has_face_label & has_place_label))
    roi_labels_categ[has_place_label] = place_labels_renumbered[has_place_label] # overwrite with prf rois
    print('unique values in categ labels:')
    print(np.unique(roi_labels_categ))

    # how much overlap between these sets of roi definitions?
    print('%d voxels are defined (differently) in both retinotopic areas and category areas'%np.sum((has_kast_label | has_prf_label) & (has_face_label | has_place_label)))

    # Now masking out all voxels that have any definition, and using them for the analysis. 
    voxel_mask = np.logical_or(roi_labels_retino>0, roi_labels_categ>0)
    voxel_idx = np.where(voxel_mask) # numerical indices into the big 3D array
    print('\n%d voxels are defined across all areas, and will be used for analysis\n'%np.size(voxel_idx))

    # Now going to print out some more information about these rois and their individual sizes...
    print('Loading numerical label/name mappings for all ROIs:')
    ret, face, place = load_roi_label_mapping(subject, verbose=True)

    print('\nSizes of all defined ROIs in this subject:')
    ret_vox_total = 0
    
    # checking these grouping labels to make sure we have them correct (print which subregions go to which label)
    for gi, group in enumerate(ret_group_inds):
        n_this_region = np.sum(np.isin(roi_labels_retino, group))
        print('Region %s has %d voxels. Includes subregions:'%(ret_group_names[gi],n_this_region))
        inds = np.where(np.isin(ret[0],group))[0]
        print(list(np.array(ret[1])[inds]))
        ret_vox_total = ret_vox_total + n_this_region

    assert(np.sum(roi_labels_retino>0)==ret_vox_total)
    print('\n')
    categ_vox_total = 0
    categ_group_names = list(np.concatenate((np.array(face[1]), np.array(place[1])), axis=0))
    categ_group_inds = [[ii+26] for ii in range(len(categ_group_names))]

    # checking these grouping labels to make sure we have them correct (print which subregions go to which label)
    for gi, group in enumerate(categ_group_inds):
        n_this_region = np.sum(np.isin(roi_labels_categ, group))
        print('Region %s has %d voxels.'%(categ_group_names[gi],n_this_region))   
        categ_vox_total = categ_vox_total + n_this_region

    assert(np.sum(roi_labels_categ>0)==categ_vox_total)
    
    return voxel_mask, voxel_idx, [roi_labels_retino, roi_labels_categ], ncsnr_full, brain_nii_shape


                
                  
def view_data(vol_shape, idx_mask, data_vol, order='C', save_to=None):
    view_vol = np.ones(np.prod(vol_shape), dtype=np.float32) * np.nan
    view_vol[idx_mask.astype('int').flatten()] = data_vol
    view_vol = view_vol.reshape(vol_shape, order=order)
    if save_to:
        nib.save(nib.Nifti1Image(view_vol, affine=np.eye(4)), save_to)
    return view_vol



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
