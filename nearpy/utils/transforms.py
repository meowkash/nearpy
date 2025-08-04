import numpy as np 
import numpy as np 
from scipy import signal 
from sklearn.preprocessing import minmax_scale 

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return data[s<m]

def align_and_normalize(sig, ref):
    # Given a reference signal, align polarity
    if np.corrcoef(sig, ref) < 0:
        return minmax_scale(max(sig)-sig)
    else:
        return minmax_scale(sig)

def normalize(x): 
    return (x-np.min(x))/(np.max(x)-np.min(x))  

def xcorr(x,y):
    """
    Perform Cross-Correlation on x and y (with the same API as MATLAB)
    x    : 1st signal
    y    : 2nd signal

    returns
    lags : lags of correlation
    corr : coefficients of correlation
    """
    corr = signal.correlate(x, y, mode="full")
    lags = signal.correlation_lags(len(x), len(y), mode="full")
    return lags, corr

def get_sig_power(sig): 
    return np.sum(np.abs(sig)**2)/len(sig)

def resample_indices(indices: list,  source: list, target: list): 
    ''' 
    Given indices from source, provide equivalent indices in target. 
    This essentialy performs resampling using minimized distances 
    '''
    assert (source[0] == target[0]) and (source[-1] == target[-1]), 'Indexing axes must have the same start and end values'
    start_val = source[indices[0]]
    end_val = source[indices[-1]]
    
    tgt_start = np.argmin(np.abs(target-start_val))
    tgt_end = np.argmin(np.abs(target-end_val))
    
    return list(range(tgt_start, tgt_end))