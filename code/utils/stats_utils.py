import numpy as np
import scipy.stats
import warnings

def numpy_corrcoef_warn(a,b):
    
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            cc = np.corrcoef(a,b)
        except RuntimeWarning as e:
            print('Warning: problem computing correlation coefficient')
            print('shape a: ',a.shape)
            print('shape b: ',b.shape)
            print('sum a: %.9f'%np.sum(a))
            print('sum b: %.9f'%np.sum(b))
            print('std a: %.9f'%np.std(a))
            print('std b: %.9f'%np.std(b))
            print(e)
            warnings.filterwarnings('ignore')
            cc = np.corrcoef(a,b)
            
    if np.any(np.isnan(cc)):
        print('There are nans in correlation coefficient')
    
    return cc

def get_r2(actual,predicted):
    """
    This computes the coefficient of determination (R2).
    Always goes along first dimension (i.e. the trials/samples dimension)
    MAKE SURE INPUTS ARE ACTUAL AND THEN PREDICTED, NOT FLIPPED
    """
    ssres = np.sum(np.power((predicted - actual),2), axis=0);
    sstot = np.sum(np.power((actual - np.mean(actual, axis=0)),2), axis=0);
    r2 = 1-(ssres/sstot)
    
    return r2

def get_corrcoef(actual,predicted,dtype=np.float32):
    """
    This computes the linear correlation coefficient.
    Always goes along first dimension (i.e. the trials/samples dimension)
    Assume input is 2D.
    """
    assert(len(actual.shape)==2)
    vals_cc = np.full(fill_value=0, shape=(actual.shape[1],), dtype=dtype)
    for vv in range(actual.shape[1]):
        vals_cc[vv] = numpy_corrcoef_warn(actual[:,vv], predicted[:,vv])[0,1] 
    return vals_cc
 
def get_dprime(predlabs,reallabs,un=None):
    """ 
    Calculate d' for predicted and actual values. Works for multiple classes.
    """

    predlabs==np.squeeze(predlabs)
    reallabs==np.squeeze(reallabs)
    if len(predlabs)!=len(reallabs):
        raise ValueError('real and predicted labels do not match')
    if len(predlabs.shape)>1 or len(reallabs.shape)>1:
        raise ValueError('need to have 1d inputs')
    if un is None:
        un = np.unique(reallabs)
    if not np.all(np.isin(np.unique(predlabs), un)):
        print('Warning: some labels in pred are not included in real labels! Will return nan')
        return np.nan
    
    hrz=np.zeros((len(un),1));
    fpz=np.zeros((len(un),1));

    n_trials = len(predlabs);

    #loop over class labels, get a hit rate and false pos for each (treating
    #any other category as non-hit)
    for ii in range(len(un)):

        if np.sum(reallabs==un[ii])==0 or np.sum(reallabs!=un[ii])==0:

            # if one of the categories is completely absent - this will return a
            # nan dprime value
            return np.nan

        else:

            hr = np.sum((predlabs==un[ii]) & (reallabs==un[ii]))/np.sum(reallabs==un[ii]);
            fp = np.sum((predlabs==un[ii]) & (reallabs!=un[ii]))/np.sum(reallabs!=un[ii]);    

            # make sure this never ends up infinite
            # correction from Macmillan & Creelman, use 1-1/2N or 1/2N in place
            # of 1 or 0 
            if hr==0:
                hr=1/(2*n_trials)
            if fp==0:
                fp=1/(2*n_trials)
            if hr==1:
                hr=1-1/(2*n_trials)
            if fp==1:
                fp=1-1/(2*n_trials);

        # convert to z score (this is like percentile - so 50% hr would be zscore=0)
        hrz[ii]=scipy.stats.norm.ppf(hr,0,1);
        fpz[ii]=scipy.stats.norm.ppf(fp,0,1);

    # dprime is the mean of individual dprimes (for two classes, they will be
    # same value)
    dprime = np.mean(hrz-fpz);

    return dprime


def get_shared_unique_var(combined, just_a, just_b):
    
    unique_a = combined - just_b
    unique_b = combined - just_a
    shared_ab = just_a + just_b - combined
   
    return shared_ab, unique_a, unique_b


def von_mises_deg(xx,mu,k,a=None,b=None,normalize=True,axis_size_deg = 180):
  
    """
    Make a von mises function over the range in xx.
    Input should be in 180 deg or 360 deg space, can specify space in degrees.
    mu = center
    k = concentration parameter
    a = amplitude (height)
    b = baseline
    if normalize=True, a and b are applied following normalization to range 0-1. 
    generally want to normalize first, otherwise height will vary with k.
    
    """ 
    if k<10**(-15):
        print('WARNING: k is too small, might get precision errors')
    
    xx_rad2pi = np.float128(xx/axis_size_deg*2*np.pi)
    mu_rad2pi = mu/axis_size_deg*2*np.pi
    yy = np.exp(k*(np.cos(xx_rad2pi-mu_rad2pi)-1))
    
    if normalize:
        # make the y values span from 0-1
        yy = yy-np.min(yy)
        yy = yy/np.max(yy)
    
    # then apply the entered amplitude and baseline.
    if a is not None:
        yy *= a
    if b is not None:
        yy += b

    return yy

def anova_oneway_warn(groups):
    
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            anova_out = scipy.stats.f_oneway(*groups)
        except RuntimeWarning as e:
            print('Warning: problem with one way anova. Means/vars/counts each group:')
            means = [np.mean(group) for group in groups]
            vrs = [np.var(group) for group in groups]
            counts = [len(group) for group in groups]
            print(means)
            print(vrs)
            print(counts)
            print(e)
            warnings.filterwarnings('ignore')
            anova_out = scipy.stats.f_oneway(*groups)
    
    if np.isnan(anova_out.statistic):
        print('nans in anova result')
           
    return anova_out