import numpy as np
import scipy.stats

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