import sys, os
import numpy as np
import h5py
import time

code_dir = '/user_data/mmhender/imStat/code/'
sys.path.append(code_dir)
from utils import default_paths

def change_h5py_dtype(h5py_fn, dtype=np.float32, fieldname='features'):
    
    t=time.time()
    with h5py.File(h5py_fn, 'r') as data_set:
        data_orig = np.copy(data_set['/%s'%fieldname])
        print('original size:')
        print(data_orig.shape)
        print('first element:')
        print(data_orig[0,0,0])
        dtype_orig = data_orig.dtype
        print('original type:')
        print(dtype_orig)
        data_set.close() 
    elapsed = time.time() - t;
    print('took %.5f sec to load file'%elapsed)
    
    if dtype_orig==dtype:
        print('file %s is already in desired data type (%s), exiting'%(h5py_fn,dtype))
    else:
        print('changing dtype of %s to %s'%(h5py_fn, dtype))
        t=time.time()
        with h5py.File(h5py_fn, 'w') as data_set:
            dset = data_set.create_dataset(fieldname, np.shape(data_orig), dtype=dtype)
            data_set['/features'][:,:,:] = data_orig
            data_set.close()
        elapsed = time.time() - t;
        print('took %.5f sec to write new file'%elapsed)
        
        t=time.time()
        with h5py.File(h5py_fn, 'r') as data_set:
            print('new size:')
            print(data_set['/%s'%fieldname].shape)
            print('first element:')
            print(data_set['/%s'%fieldname][0,0,0])
            print('new type:')
            print(data_set['/%s'%fieldname].dtype)
            data_set.close() 
        elapsed = time.time() - t;
        print('took %.5f sec to load file'%elapsed)
        
if __name__ == '__main__':
       
    subs2do = [8]
#     subs2do = np.arange(1,9,1)
    features_folder = os.path.join(default_paths.root, 'features', 'pyramid_texture')
    for subject in subs2do:
        h5py_fn = os.path.join(features_folder,'S%d_features_each_prf_4ori_4sf_grid5.h5py')
        change_h5py_dtype(h5py_fn, dtype=np.float32, fieldname='features')

    features_folder = os.path.join(default_paths.root, 'features', 'sketch_tokens')
    for subject in subs2do:
        h5py_fn = os.path.join(features_folder,'S%d_features_each_prf_grid5.h5py')
        change_h5py_dtype(h5py_fn, dtype=np.float32, fieldname='features')

    features_folder = os.path.join(default_paths.root, 'features', 'gabor_texure')
    for subject in subs2do:
        h5py_fn = os.path.join(features_folder,'S%d_features_each_prf_12ori_8sf_gabor_solo_nonlin_grid5.h5py')
        change_h5py_dtype(h5py_fn, dtype=np.float32, fieldname='features')
