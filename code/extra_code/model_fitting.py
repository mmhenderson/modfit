""" 
OLD code - may be useful in future
General code related to fitting encoding models.
"""

import numpy as np
import torch 
from torch.utils.data import Dataset
from torch import nn


    
def get_r2(actual,predicted):
  
    # calculate r2 for this fit.
    ssres = np.sum(np.power((predicted - actual),2));
#     print(ssres)
    sstot = np.sum(np.power((actual - np.mean(actual)),2));
#     print(sstot)
    r2 = 1-(ssres/sstot)
    
    return r2

    
class fwrf_encoding_model(nn.Module):
    """
    Definition of a simple linear encoding model that can be fit with grad descent.
    nFeats would be e.g. number of feature maps. 
    """
    def __init__(self, nFeats):
    
        super(encoding_model, self).__init__()
        self.weights = nn.Parameter(torch.randn(nFeats, 1) / np.sqrt(nFeats))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        
        pred_resp = torch.matmul(x, self.weights) + self.bias
        return pred_resp
    

    
    
class ridge_regression_model():
    """
    Ridge regression model, solving with least squares.
    Input x is e.g. [nTrials x nFeatureMaps]
    Output y is e.g. [nTrials x nVoxels]
    """
    def __init__(self, lamb):
    
        self.lamb = lamb
        
    def fit(self, x, y):
        
        assert(x.shape[0] == y.shape[0])
        # adding extra column of ones for the feature matrix
        x = torch.cat([x, torch.ones(x.shape[0], 1)], dim = 1)
        # Solving X*w = y with Normal equations:
        # X^{T}*X*w = X^{T}*y 
        lhs = x.T @ x 
        rhs = x.T @ y
        
        ridge = self.lamb*torch.eye(lhs.shape[0])
        self.w, _ = torch.lstsq(rhs, lhs + ridge)
        
    def predict(self, x):
        
        # adding extra column of ones for the feature matrix
        x = torch.cat([x, torch.ones(x.shape[0], 1)], dim = 1)
        y = x @ self.w
        
        return y
    
    
    
def loss_sse(pred_resp, real_resp):
    """ 
    Calculate loss based on sum of squared error.
    If input is a matrix, assume [nTrials x nVoxels] and returns [nVoxels]
    """
    pred_resp = torch.tensor(pred_resp)
    real_resp = torch.tensor(real_resp)
    if len(torch.squeeze(pred_resp).shape)==1 & len(torch.squeeze(real_resp).shape)==1:
        pred_resp = torch.squeeze(pred_resp)
        real_resp = torch.squeeze(real_resp)       
        error = torch.sum(torch.square(pred_resp - real_resp))
    else:       
        assert(np.all(pred_resp.shape==real_resp.shape))
        error = torch.sum(torch.square(pred_resp - real_resp), axis=0)
    
    return error