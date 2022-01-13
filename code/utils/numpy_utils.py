import numpy as np
import scipy.io as sio
from scipy.special import erf
import math
import sys
import scipy.stats

def zscore_in_groups(data, group_labels):
    """
    Apply z-scoring to data of several columns at a time - column groupings given by group_labels.
    """
    if len(group_labels.shape)>1:
        group_labels = np.squeeze(group_labels)
    assert(len(group_labels.shape)==1)
    zdata = np.zeros(shape=np.shape(data))
    ungroups = np.unique(group_labels)
    for gg in range(len(ungroups)):       
        d = data[:,group_labels==ungroups[gg]]        
        zd = np.reshape(scipy.stats.zscore(d.ravel()), np.shape(d))        
        zdata[:,group_labels==ungroups[gg]] = zd
        
    return zdata

def zscore_in_groups_trntest(trndata, tstdata, group_labels):
    """
    Apply z-scoring to data of several columns at a time - column groupings given by group_labels.
    Computing mean/std on training set only, and apply same parameters to test set.
    """
    if len(group_labels.shape)>1:
        group_labels = np.squeeze(group_labels)
    assert(len(group_labels.shape)==1)
    trn_zdata = np.zeros(shape=np.shape(trndata))
    tst_zdata = np.zeros(shape=np.shape(tstdata))
    ungroups = np.unique(group_labels)
    for gg in range(len(ungroups)):       
        trnd = trndata[:,group_labels==ungroups[gg]] 
        tstd = tstdata[:,group_labels==ungroups[gg]] 
        trnmean = np.mean(trnd.ravel())
        trnstd = np.std(trnd.ravel())
        trn_zdata[:,group_labels==ungroups[gg]] = (trnd-trnmean)/trnstd
        tst_zdata[:,group_labels==ungroups[gg]] = (tstd-trnmean)/trnstd
        
    return trn_zdata, tst_zdata

def unshuffle(shuffled_data, shuffle_order):
    """
    Take an array that has been shuffled according to shuffle_order, and re-create its original order.
    Assumes that first dim of data is what needs to be unshuffled.    
    """   
    unshuffle_order = np.zeros_like(shuffle_order);
    unshuffle_order[shuffle_order] = np.arange(shuffled_data.shape[0])
    unshuffled_data = shuffled_data[unshuffle_order] # Unshuffle the shuffled data
    return unshuffled_data

def invertible_sort(sequence):
    """
    Sort a sequence and store the order needed to reverse sort.
    Based on np.argsort.
    """
    order2sort = np.argsort(sequence)
    order2reverse = np.argsort(order2sort)
    
    return order2sort, order2reverse

def iterate_range(start, length, batchsize):
    batch_count = int(length // batchsize )
    residual = int(length % batchsize)
    for i in range(batch_count):
        yield range(start+i*batchsize, start+(i+1)*batchsize),batchsize
    if(residual>0):
        yield range(start+batch_count*batchsize,start+length),residual 


def select_along_axis(a, choice, run_axis=0, choice_axis=1):
    ''' run axis of lenght N
        choice axis of lenght M
        choice is a vector of lenght N with integer entries between 0 and M (exclusive).
        Equivalent to:
        >   for i in range(N):
        >       r[...,i] = a[...,i,...,choice[i],...]
        returns an array with the same shape as 'a' minus the choice_axis dimension
    '''
    assert len(choice)==a.shape[run_axis], "underspecified choice"
    final_pos = run_axis - (1 if choice_axis<run_axis else 0)
    val = np.moveaxis(a, source=[run_axis, choice_axis], destination=[0,1])
    il = list(val.shape)
    il.pop(1)
    r = np.ndarray(shape=tuple(il), dtype=a.dtype)
    for i in range(len(choice)):
        r[i] = val[i,choice[i]]
    return np.moveaxis(r, source=0, destination=final_pos)

def get_list_size_gib(list_object):
    
    try:
        fullsize = np.sum([list_object[ii].nbytes for ii in range(len(list_object))])
    except:
        fullsize = np.sum([sys.getsizeof(list_object[ii]) for ii in range(len(list_object))])
    gib, gb = bytes_to_gb(fullsize)
    
    return gib
    
def bytes_to_gb(bytes_size):

    gib = bytes_size/1024**3
    gb = bytes_size/1000**3
    
    return gib, gb