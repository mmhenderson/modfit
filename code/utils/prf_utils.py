import struct
import numpy as np
import math
from scipy.special import erf

def make_polar_angle_grid(sigma_range=[0.04, 1], n_sigma_steps=12, \
                          eccen_range=[0, 1.4], n_eccen_steps=12, n_angle_steps=16):
    
    """
    Create a grid of pRF positions/sizes that are evenly spaced in polar angle.
    Sizes and eccentricities are logarithmically spaced.
    Units for size/sigma are relative to image aperture size (1.0 units)
    To convert to degrees, multiply these values by degrees of display (8.4 degrees for NSD expts)
    """

    sigma_vals = np.logspace(np.log10(sigma_range[0]), np.log10(sigma_range[1]), n_sigma_steps)
    # min eccen should usually be zero, accounting for this here
    eccen_vals = np.logspace(np.log10(eccen_range[0]+0.1), np.log10(eccen_range[1]+0.1), n_eccen_steps) - 0.1
    angle_step = 2*np.pi/n_angle_steps
    angle_vals = np.linspace(0,np.pi*2-angle_step, n_angle_steps)

    eccen_vals = eccen_vals.astype(np.float32)
    angle_vals = angle_vals.astype(np.float32)
    sigma_vals = sigma_vals.astype(np.float32)

    # First, create the grid of all possible combinations.
    x_vals = (eccen_vals[:,np.newaxis] * np.cos(angle_vals[np.newaxis,:]))
    y_vals = (eccen_vals[:,np.newaxis] * np.sin(angle_vals[np.newaxis,:]))

    x_vals = np.tile(np.reshape(x_vals, [n_angle_steps*n_eccen_steps, 1]), [n_sigma_steps,1])
    y_vals = np.tile(np.reshape(y_vals, [n_angle_steps*n_eccen_steps, 1]), [n_sigma_steps,1])

    sigma_vals = np.repeat(sigma_vals, n_angle_steps*n_eccen_steps)[:,np.newaxis]

    prf_params = np.concatenate([x_vals, y_vals, sigma_vals], axis=1)

    # Now removing a few of these that we don't want to actually use - some duplicates, 
    # and some that go entirely outside the image region.
    unrows, inds = np.unique(prf_params, axis=0, return_index=True)
    prf_params = prf_params[np.sort(inds),:]

    # what is the approx spatial extent of the pRF? Assume 1 standard deviation.
    n_std = 1
    left_extent = prf_params[:,0] - prf_params[:,2]*n_std
    right_extent = prf_params[:,0] + prf_params[:,2]*n_std
    top_extent = prf_params[:,1] + prf_params[:,2]*n_std
    bottom_extent = prf_params[:,1] - prf_params[:,2]*n_std

    out_of_bounds = (left_extent > 0.5) | (right_extent < -0.5) | (top_extent < -0.5) | (bottom_extent > 0.5)

    prf_params = prf_params[~out_of_bounds,:]

    return prf_params

class subdivision_1d(object):
    def __init__(self, n_div=1, dtype=np.float32):
        self.length = n_div
        self.dtype = dtype
        
    def __call__(self, center, width):
        '''	returns a list of point positions '''
        return [center] * self.length
    
class linspace(subdivision_1d):    
    def __init__(self, n_div, right_bound=False, dtype=np.float32, **kwargs):
        super(linspace, self).__init__(n_div, dtype=np.float32, **kwargs)
        self.__rb = right_bound
        
    def __call__(self, center, width):
        if self.length<=1:
            return [center]     
        if self.__rb:
            d = np.float32(width)/(self.length-1)
            vmin, vmax = center, center+width  
        else:
            d = np.float32(width)/self.length
            vmin, vmax = center+(d-width)/2, center+width/2 
        return np.arange(vmin, vmax+1e-12, d).astype(dtype=self.dtype)
    
class logspace(subdivision_1d):    
    def __init__(self, n_div, dtype=np.float32, **kwargs):
        super(logspace, self).__init__(n_div, dtype=np.float32, **kwargs)
               
    def __call__(self, start, stop):    
        if self.length <= 1:
            return [start]
        lstart = np.log(start+1e-12)
        lstop = np.log(stop+1e-12)
        dlog = (lstop-lstart)/(self.length-1)
        return np.exp(np.arange(lstart, lstop+1e-12, dlog)).astype(self.dtype)

def model_space_pyramid(sigmas, min_spacing, aperture):
    rf = []
    for s in sigmas:
        X, Y = np.meshgrid(np.linspace(-aperture/2, aperture/2, int(np.ceil(aperture/(s * min_spacing)))),
                           np.linspace(-aperture/2, aperture/2, int(np.ceil(aperture/(s * min_spacing)))))
        rf += [np.stack([X.flatten(), Y.flatten(), np.full(fill_value=s, shape=X.flatten().shape)], axis=1),]
    return np.concatenate(rf, axis=0)

def model_space_pyramid2(sigmas, min_spacing, aperture):
    rf = []
    n_grid_list = [21,15,11,9,7,7,7,7,7]
    for si, s in enumerate(sigmas):
        n_grid = n_grid_list[si]
        X, Y = np.meshgrid(np.linspace(-aperture/2, aperture/2, n_grid),
                           np.linspace(-aperture/2, aperture/2, n_grid))
        rf += [np.stack([X.flatten(), Y.flatten(), np.full(fill_value=s, shape=X.flatten().shape)], axis=1),]
    s = 10.0
    n_grid = 1;
    X, Y = X, Y = np.meshgrid(np.linspace(0,0, n_grid),
                           np.linspace(0,0, n_grid))
    rf += [np.stack([X.flatten(), Y.flatten(), np.full(fill_value=s, shape=X.flatten().shape)], axis=1),]
    return np.concatenate(rf, axis=0)

def gauss_2d(center, sd, patch_size, orient_deg=0, aperture=1.0, dtype=np.float32):
    """
     Making a little gaussian blob. Can be elongated in one direction relative to other.
     [sd] is the x and y standard devs, respectively. 
     center and size are scaled according to the patch size, so that the blob always 
     has the same center/size relative to image even when patch size is different.
     aperture defines the number of arbitrary "units" occupied by the whole image
     units occupied by each pixel = aperture/patch_size.
        
     """
    if (not hasattr(sd,'__len__')) or len(sd)==1:
        sd = np.array([sd, sd])
        
    aspect_ratio = sd[0] / sd[1]
    orient_rad = orient_deg/180*np.pi
    
    # first meshgrid over image space
    x,y = np.meshgrid(np.linspace(-aperture/2, aperture/2, patch_size), \
                      np.linspace(-aperture/2, aperture/2, patch_size))
    x_centered = x-center[0]
    center[1] = (-1)*center[1] # negate the y coord so the grid matches w my other code
    y_centered = y-center[1]
    
    # rotate the axes to match desired orientation (if orient=0, this is just regular x and y)
    x_prime = x_centered * np.cos(orient_rad) + y_centered * np.sin(orient_rad)
    y_prime = y_centered * np.cos(orient_rad) - x_centered * np.sin(orient_rad)

    # make my gaussian w the desired size/eccentricity
    gauss = np.exp(-((x_prime)**2 + aspect_ratio**2 * (y_prime)**2)/(2*sd[0]**2))
    
    # normalize so it will sum to 1
    gauss = gauss/np.sum(gauss)
    
    gauss = gauss.astype(dtype)
    
    return gauss


def gaussian_mass(xi, yi, dx, dy, x, y, sigma):
    return 0.25*(erf((xi-x+dx/2)/(np.sqrt(2)*sigma)) - erf((xi-x-dx/2)/(np.sqrt(2)*sigma)))*(erf((yi-y+dy/2)/(np.sqrt(2)*sigma)) - erf((yi-y-dy/2)/(np.sqrt(2)*sigma)))
    
def make_gaussian_mass(x, y, sigma, n_pix, size=None, dtype=np.float32):
    deg = dtype(n_pix) if size==None else size
    dpix = dtype(deg) / n_pix
    pix_min = -deg/2. + 0.5 * dpix
    pix_max = deg/2.
    [Xm, Ym] = np.meshgrid(np.arange(pix_min,pix_max,dpix), np.arange(pix_min,pix_max,dpix));
    if sigma<=0:
        Zm = np.zeros_like(Xm)
    elif sigma<dpix:
        g_mass = np.vectorize(lambda a, b: gaussian_mass(a, b, dpix, dpix, x, y, sigma)) 
        Zm = g_mass(Xm, -Ym)        
    else:
        d = (2*dtype(sigma)**2)
        A = dtype(1. / (d*np.pi))
        Zm = dpix**2 * A * np.exp(-((Xm-x)**2 + (-Ym-y)**2) / d)
    return Xm, -Ym, Zm.astype(dtype)   
    
def make_gaussian_mass_stack(xs, ys, sigmas, n_pix, size=None, dtype=np.float32):
    stack_size = min(len(xs), len(ys), len(sigmas))
    assert stack_size>0
    Z = np.ndarray(shape=(stack_size, n_pix, n_pix), dtype=dtype)
    X,Y,Z[0,:,:] = make_gaussian_mass(xs[0], ys[0], sigmas[0], n_pix, size=size, dtype=dtype)
    for i in range(1,stack_size):
        _,_,Z[i,:,:] = make_gaussian_mass(xs[i], ys[i], sigmas[i], n_pix, size=size, dtype=dtype)
    return X, Y, Z
