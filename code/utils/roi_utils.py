import numpy as np
import os, sys
import pandas as pd
import nibabel as nib
import copy

ret_group_names = ['V1', 'V2', 'V3','hV4','VO1-2','PHC1-2','LO1-2','TO1-2','V3ab','IPS0-5','SPL1','FEF']
ret_group_inds = [[1,2],[3,4],[5,6],[7],[8,9],[10,11],[14,15],[12,13],[16,17],[18,19,20,21,22,23],[24],[25]]

from utils import default_paths
from utils import nsd_utils


class nsd_roi_def():
    
    """
    An object containing ROI definitions for a subject in NSD dataset.
    The ROIs included here are retinotopic visual regions (defined using a combination of 
    Kastner 2015 atlas and pRF mapping data), and category-selective (face, place, body) ROIs.
    Definitions will be partly overlapping between different methods 
    (for instance a voxel can be in both V3A and OPA)
    """
    
    def __init__(self, subject, volume_space=True, include_all=True, verbose=False):
        
        self.subject=subject
        self.volume_space=volume_space
        self.include_all=include_all
      
        self.init_names(verbose)
        self.init_labels(verbose)
        
    def init_names(self, verbose):
        
        ret, face, place, body = load_roi_label_mapping(self.subject, verbose=verbose)
        self.face_names = face[1]
        self.place_names = place[1]
        self.body_names = body[1]
        self.ret_names = ret_group_names
        
        self.nret = len(self.ret_names)
        self.nface = len(self.face_names)
        self.nplace = len(self.place_names)
        self.nbody = len(self.body_names)    
        
        self.n_rois = len(self.ret_names) + len(self.face_names) \
                        + len(self.place_names) + len(self.body_names)
        self.roi_names = self.ret_names+self.face_names+self.place_names+self.body_names

        self.is_ret = np.arange(0, self.n_rois)<self.nret
        self.is_face = (np.arange(0, self.n_rois)>=self.nret) & \
                        (np.arange(0, self.n_rois)<self.nret+self.nface)
        self.is_place = (np.arange(0, self.n_rois)>=self.nret+self.nface) & \
                        (np.arange(0, self.n_rois)<self.nret+self.nface+self.nplace)
        self.is_body = np.arange(0, self.n_rois)>=self.nret+self.nface+self.nplace
        
    def init_labels(self, verbose):
        
        voxel_mask, voxel_index, voxel_roi, voxel_ncsnr, brain_nii_shape = \
                    get_voxel_roi_info(self.subject, \
                               volume_space=self.volume_space, \
                               include_all=self.include_all, \
                               verbose=verbose)
        self.voxel_mask = voxel_mask
        self.nii_shape = brain_nii_shape
        
        [roi_labels_retino, roi_labels_face, roi_labels_place, roi_labels_body] = \
                    copy.deepcopy(voxel_roi)
        
        # make these zero-indexed, where 0 is first ROI and -1 is not in any ROI
        self.facelabs = roi_labels_face[voxel_index] - 1        
        self.facelabs[self.facelabs==-2] = -1
        self.placelabs = roi_labels_place[voxel_index] - 1
        self.placelabs[self.placelabs==-2] = -1
        self.bodylabs = roi_labels_body[voxel_index] - 1
        self.bodylabs[self.bodylabs==-2] = -1

        roi_labels_retino = roi_labels_retino[voxel_index]
        self.retlabs = (-1)*np.ones(np.shape(roi_labels_retino))

        for rr in range(len(ret_group_names)):   
            inds_this_roi = np.isin(roi_labels_retino, ret_group_inds[rr])
            self.retlabs[inds_this_roi] = rr

    def get_indices_from_name(self, roi_name):
        
        if np.any([name==roi_name for name in self.roi_names]):
            rr = np.where([name==roi_name for name in self.roi_names])[0][0]
            return self.get_indices(rr)
        else:
            raise ValueError('%s not in list of ROIs, see self.roi_names for options.'%roi_name)
        
    def get_indices(self, rr):
        
        # rr is an index into self.roi_names, in range 0-self.n_rois
        if (rr<0) or (rr>(self.n_rois-1)):
            raise ValueError('rr needs to be between 0-%d'%self.n_rois)
        
        if self.is_ret[rr]:
            inds_this_roi = self.retlabs==rr
        elif self.is_face[rr]:
            inds_this_roi = self.facelabs==(rr-self.nret)
        elif self.is_place[rr]:
            inds_this_roi = self.placelabs==(rr-self.nret-self.nface)
        elif self.is_body[rr]:
            inds_this_roi = self.bodylabs==(rr-self.nret-self.nface-self.nplace)

        return inds_this_roi
    
    def get_sizes(self):

        n_each = np.zeros((self.n_rois,),dtype=int)
        for rr in range(self.n_rois):
            inds_this_roi = self.get_indices(rr)           
            n_total = np.sum(inds_this_roi)
            n_each[rr] = n_total

        return n_each

    def print_overlap(self):

        for rr in range(self.n_rois):
            inds_this_roi = self.get_indices(rr)         
            n_total = np.sum(inds_this_roi)
            print('%s: %d vox total'%(self.roi_names[rr], n_total))
            for rr2 in range(self.n_rois):                
                if rr2==rr:
                    continue                   
                inds_this_roi2 = self.get_indices(rr2)
                n_overlap = np.sum(inds_this_roi & inds_this_roi2)
                if n_overlap>0:
                    print('    %d vox overlap with %s'%(n_overlap, self.roi_names[rr2]))


def load_roi_label_mapping(subject, verbose=False):
    """
    Load files (ctab) that describe the mapping from numerical labels to text labels for the ROIs.
    These correspond to the mask definitions of each type of ROI (either nii or mgz files).
    """
    
    filename_prf = os.path.join(default_paths.nsd_root,'nsddata','freesurfer',\
                                'subj%02d'%subject, 'label', 'prf-visualrois.mgz.ctab')
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

    filename_ret = os.path.join(default_paths.nsd_root,'nsddata','freesurfer',\
                                'subj%02d'%subject, 'label', 'Kastner2015.mgz.ctab')
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

    # kastner atlas and prf have same values/names for all shared elements, 
    # so can just use kastner going forward.
    assert(np.array_equal(prf_num_labels,ret_num_labels[0:len(prf_num_labels)]))
    assert(np.array_equal(prf_text_labels,ret_text_labels[0:len(prf_text_labels)]))

    filename_faces = os.path.join(default_paths.nsd_root,'nsddata','freesurfer',\
                                  'subj%02d'%subject, 'label', 'floc-faces.mgz.ctab')
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

    filename_places = os.path.join(default_paths.nsd_root,'nsddata','freesurfer',\
                                   'subj%02d'%subject, 'label', 'floc-places.mgz.ctab')
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
        
    filename_body = os.path.join(default_paths.nsd_root,'nsddata','freesurfer',\
                                 'subj%02d'%subject, 'label', 'floc-bodies.mgz.ctab')
    names = np.array(pd.read_csv(filename_body))
    names = [str(name) for name in names]
    body_num_labels = [int(name[2:np.char.find(name,' ')]) for name in names]
    body_text_labels=[]
    for name in names:
        if np.char.find(name,'\\')>-1:
            body_text_labels.append(name[np.char.find(name,' ')+1:np.char.find(name,'\\')])
        else:
            body_text_labels.append(name[np.char.find(name,' ')+1:np.char.find(name,"'",2)])
    if verbose:
        print(body_num_labels)
        print(body_text_labels)

    return [ret_num_labels, ret_text_labels], [faces_num_labels, faces_text_labels], \
            [places_num_labels, places_text_labels], [body_num_labels, body_text_labels]
                
     
def get_voxel_roi_info(subject, volume_space=True, include_all=False, verbose=True):

    """
    For a specified subject, load all definitions of all ROIs for this subject. 
    The ROIs included here are retinotopic visual regions (defined using a combination of Kastner 2015 atlas
    and pRF mapping data), and category-selective (face and place) ROIs.
    Will return four separate bricks of labels, for each method of definition (retino, category, etc.)
    These are partially overlapping, so can choose later which definition to use for the overlapping voxels.
    Can be done in either volume space (volume_space=True) or surface space (volume_space=False).
    If surface space, then each voxel is a "vertex" of mesh.
    """
    
     # First loading each ROI definitions file - lists nvoxels long, with diff numbers for each ROI.
    if volume_space:

        roi_path = os.path.join(default_paths.nsd_root, 'nsddata', 'ppdata', \
                                'subj%02d'%subject, 'func1pt8mm', 'roi')
       
        if verbose:
            print('\nVolume space: ROI defs are located at: %s\n'%roi_path)

        nsd_general_full = nsd_utils.load_from_nii(os.path.join(roi_path, 'nsdgeneral.nii.gz')).flatten()
            
        prf_labels_full  = nsd_utils.load_from_nii(os.path.join(roi_path, 'prf-visualrois.nii.gz'))
        # save the shape, so we can project back to volume space later.
        brain_nii_shape = np.array(prf_labels_full.shape)
        prf_labels_full = prf_labels_full.flatten()

        kast_labels_full = nsd_utils.load_from_nii(os.path.join(roi_path, 'Kastner2015.nii.gz')).flatten()
        face_labels_full = nsd_utils.load_from_nii(os.path.join(roi_path, 'floc-faces.nii.gz')).flatten()
        place_labels_full = nsd_utils.load_from_nii(os.path.join(roi_path, 'floc-places.nii.gz')).flatten()
        body_labels_full = nsd_utils.load_from_nii(os.path.join(roi_path, 'floc-bodies.nii.gz')).flatten()
        
        # Masks of ncsnr values for each voxel 
        ncsnr_full = nsd_utils.load_from_nii(os.path.join(default_paths.beta_root, \
                                              'subj%02d'%subject, 'func1pt8mm', \
                                              'betas_fithrf_GLMdenoise_RR', 'ncsnr.nii.gz')).flatten()

    else:
        
        roi_path = os.path.join(default_paths.nsd_root,'nsddata', 'freesurfer', 'subj%02d'%subject, 'label')

        if verbose:
            print('\nSurface space: ROI defs are located at: %s\n'%roi_path)
        
        # Surface space, concatenate the two hemispheres
        # always go left then right, to match the data which also gets concatenated same way
        prf_labs1 = nsd_utils.load_from_mgz(os.path.join(roi_path, 'lh.prf-visualrois.mgz'))[:,0,0]
        prf_labs2 = nsd_utils.load_from_mgz(os.path.join(roi_path, 'rh.prf-visualrois.mgz'))[:,0,0]
        prf_labels_full = np.concatenate((prf_labs1, prf_labs2), axis=0)

        kast_labs1 = nsd_utils.load_from_mgz(os.path.join(roi_path, 'lh.Kastner2015.mgz'))[:,0,0]
        kast_labs2 = nsd_utils.load_from_mgz(os.path.join(roi_path, 'rh.Kastner2015.mgz'))[:,0,0]
        kast_labels_full = np.concatenate((kast_labs1, kast_labs2), axis=0)

        face_labs1 = nsd_utils.load_from_mgz(os.path.join(roi_path, 'lh.floc-faces.mgz'))[:,0,0]
        face_labs2 = nsd_utils.load_from_mgz(os.path.join(roi_path, 'rh.floc-faces.mgz'))[:,0,0]
        face_labels_full = np.concatenate((face_labs1, face_labs2), axis=0)

        place_labs1 = nsd_utils.load_from_mgz(os.path.join(roi_path, 'lh.floc-places.mgz'))[:,0,0]
        place_labs2 = nsd_utils.load_from_mgz(os.path.join(roi_path, 'rh.floc-places.mgz'))[:,0,0]
        place_labels_full = np.concatenate((place_labs1, place_labs2), axis=0)
      
        body_labs1 = nsd_utils.load_from_mgz(os.path.join(roi_path, 'lh.floc-bodies.mgz'))[:,0,0]
        body_labs2 = nsd_utils.load_from_mgz(os.path.join(roi_path, 'rh.floc-bodies.mgz'))[:,0,0]
        body_labels_full = np.concatenate((body_labs1, body_labs2), axis=0)
      
        # Note this part hasn't been tested
        general_labs1 = nsd_utils.load_from_mgz(os.path.join(roi_path, 'lh.nsdgeneral.mgz'))[:,0,0]
        general_labs2 = nsd_utils.load_from_mgz(os.path.join(roi_path, 'rh.nsdgeneral.mgz'))[:,0,0]
        nsd_general_full = np.concatenate((general_labs1, general_labs2), axis=0)
  
        # Masks of ncsnr values for each voxel 
        n1 = nsd_utils.load_from_mgz(os.path.join(default_paths.beta_root, \
                                                  'subj%02d'%subject, 'nativesurface',\
                                                  'betas_fithrf_GLMdenoise_RR', 'lh.ncsnr.mgh')).flatten()
        n2 = nsd_utils.load_from_mgz(os.path.join(default_paths.beta_root, 'subj%02d'%subject, \
                                                  'nativesurface','betas_fithrf_GLMdenoise_RR',\
                                                  'rh.ncsnr.mgh')).flatten()
        ncsnr_full = np.concatenate((n1, n2), axis=0)
  
        brain_nii_shape = None

    # boolean masks of which voxels had definitions in each of these naming schemes
    has_general_label = (nsd_general_full>0).astype(bool)
    has_prf_label = (prf_labels_full>0).astype(bool)
    has_kast_label = (kast_labels_full>0).astype(bool)
    has_face_label = (face_labels_full>0).astype(bool)
    has_place_label = (place_labels_full>0).astype(bool)
    has_body_label = (body_labels_full>0).astype(bool)
    
    # To combine all regions, first starting with the kastner atlas for retinotopic ROIs.
    roi_labels_retino = np.copy(kast_labels_full)
    # Partially overwrite these defs with prf defs, which are more accurate when they exist.
    roi_labels_retino[has_prf_label] = prf_labels_full[has_prf_label]
    if verbose:
        print('%d voxels of overlap between kastner and prf definitions, using prf defs'\
              %np.sum(has_kast_label & has_prf_label))
        print('unique values in retino labels:')
        print(np.unique(roi_labels_retino))

    roi_labels_face = face_labels_full
    roi_labels_place = place_labels_full
    roi_labels_body = body_labels_full
 
    if verbose:
        print('unique values in face labels:')
        print(np.unique(roi_labels_face))
        print('unique values in place labels:')
        print(np.unique(roi_labels_place))
        print('unique values in body labels:')
        print(np.unique(roi_labels_body))
        # how much overlap between these sets of roi definitions?
        print('%d voxels are defined (differently) in both retinotopic areas and category areas'\
              %np.sum((has_kast_label | has_prf_label) & (has_face_label | has_place_label | has_body_label)))
        print('%d voxels are defined (differently) in both face areas and place areas'\
              %np.sum(has_face_label & has_place_label))    
        print('%d voxels are defined (differently) in both face areas and body areas'\
              %np.sum(has_face_label & has_body_label))    
        print('%d voxels are defined (differently) in both place areas and body areas'\
              %np.sum(has_place_label & has_body_label))    
        
    # Now masking out all voxels that have any definition, and using them for the analysis. 
    voxel_mask = (roi_labels_retino>0) | (roi_labels_face>0) | (roi_labels_place>0) | (roi_labels_body>0)
    if include_all:
        if verbose:
            print('Including all voxels that are defined within nsdgeneral mask, in addition to roi labels.')
        voxel_mask = np.logical_or(voxel_mask, has_general_label)
        
    voxel_idx = np.where(voxel_mask) # numerical indices into the big 3D array
    if verbose:
        print('\n%d voxels are defined across all areas, and will be used for analysis\n'%np.size(voxel_idx))

    if verbose:
        # Now going to print out some more information about these rois and their individual sizes...
        print('Loading numerical label/name mappings for all ROIs:')
    ret, face, place, body = load_roi_label_mapping(subject, verbose=verbose)

    if verbose:
        print('\nSizes of all defined ROIs in this subject:')
    
    # checking the retino grouping labels to make sure we have them correct 
    # (print which subregions go to which label)
    ret_vox_total = 0
    for gi, group in enumerate(ret_group_inds):
        n_this_region = np.sum(np.isin(roi_labels_retino, group))
        inds = np.where(np.isin(ret[0],group))[0]
        if verbose:
            print('Region %s has %d voxels. Includes subregions:'%(ret_group_names[gi],n_this_region))      
            print(list(np.array(ret[1])[inds]))
        ret_vox_total = ret_vox_total + n_this_region
    assert(np.sum(roi_labels_retino>0)==ret_vox_total)
    
    if verbose:
        for ii, name in enumerate(face[1]):
            print('Region %s has %d voxels'%(name,np.sum(roi_labels_face==(ii+1))))  
        for ii, name in enumerate(place[1]):
            print('Region %s has %d voxels'%(name,np.sum(roi_labels_place==(ii+1))))  
        for ii, name in enumerate(body[1]):
            print('Region %s has %d voxels'%(name,np.sum(roi_labels_body==(ii+1))))  

    return voxel_mask, voxel_idx, [roi_labels_retino, roi_labels_face, roi_labels_place, roi_labels_body], \
                ncsnr_full, brain_nii_shape


                
                  
def view_data(vol_shape, idx_mask, data_vol, order='C', save_to=None):
    view_vol = np.ones(np.prod(vol_shape), dtype=np.float32) * np.nan
    view_vol[idx_mask.astype('int').flatten()] = data_vol
    view_vol = view_vol.reshape(vol_shape, order=order)
    if save_to:
        nib.save(nib.Nifti1Image(view_vol, affine=np.eye(4)), save_to)
    return view_vol
