import ruptures as rpt 
from bottleneck import move_std
import numpy as np 
import pywt
from scipy.interpolate import CubicSpline
import antropy as ant
import tsfresh 

from .utils import reject_outliers

def get_AE_feats(sig): 
    pass 

def get_cwt_feats(sig, fs, wavelet):    
    scales = range(fs)
    coeff, _ = pywt.cwt(sig, scales, wavelet, 1)
    return coeff

def get_time_series_feats(sig):
    '''
    Given a time series signal of shape (NxD), return an array of interpretable features (NxM)
    '''
    mob, comp = ant.hjorth_params(sig)
    zc = ant.num_zerocross(sig) 
    svd_ent = ant.svd_entropy(sig, order=2)
    pass 

# For a given time series input, get the segment indices 
def get_segment_indices(sig, window, min_samples, kernel='linear', num_points=4):
    sig_var = move_std(sig, window)
    algo = rpt.KernelCPD(kernel=kernel, jump=0.1, min_size=min_samples).fit(sig_var)
    points = algo.predict(n_bkps=num_points)
    
    return points[0], points[-2]

# Given a MIMO time series gesture array, return the segment indices 
def get_gesture_indices(sig, num_channels, fs, min_samples, kernel='linear', num_points=4):
    startPoints = np.zeros((num_channels, ))
    endPoints = np.zeros((num_channels, ))
    for i in range(num_channels):
        startPoints[i], endPoints[i] = get_segment_indices(sig[i, :], round(fs/10), min_samples, kernel=kernel, num_points=num_points)
    
    # Now get the most coincident points 
    startPoints = reject_outliers(startPoints)
    endPoints = reject_outliers(endPoints)
    
    return round(np.average(startPoints)), round(np.average(endPoints))

def segment_gesture(sig, num_channels=16, kernel='linear', min_size=0.4, num_points=4, fs=1000):
    # TODO: Find better algorithm to extract gestures
    seg_length = fs 
    startIdx, endIdx = get_gesture_indices(sig, num_channels, fs, min_size * fs, kernel=kernel, num_points=num_points)
    
    sig_seg = np.zeros((num_channels, seg_length))
    
    # Segment all channels 
    for i in range(num_channels):
        foo = sig[i, startIdx:endIdx] # Get segment
        x_old = np.linspace(0, len(foo), len(foo))
        x_new = np.linspace(0, len(foo), seg_length) # Interpolate to given size 
    
        spl = CubicSpline(x_old, foo)
        
        sig_seg[i, :] = spl(x_new)
        
    return sig_seg