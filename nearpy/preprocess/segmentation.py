import numpy as np 
import pandas as pd

import numpy as np 
from bottleneck import move_std
import ruptures as rpt 
from scipy.interpolate import CubicSpline

from .utils import reject_outliers

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

def make_action_segmented_dataset(dataset, num_channels=16, fs=1000):
    # Segment length is chosen to be same as fs
    action_data = np.zeros((len(dataset), num_channels*fs))
    
    for i in range(len(dataset)):
        dat = np.reshape(dataset.iloc[i]['mag'], (num_channels, -1))
        xseg = segment_gesture(dat, num_channels=num_channels, fs=fs)
        action_data[i, :] = np.reshape(xseg, -1)
    
    seg_dataset = pd.DataFrame({'subject': dataset['subject'].squeeze(),
                                'routine': dataset['routine'].squeeze(),
                                'gesture': dataset['gesture'].squeeze(), 
                                'mag': action_data.tolist()})
    
    return seg_dataset
