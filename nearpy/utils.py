import numpy as np 
import pandas as pd
from scipy.fft import fft
from sklearn.preprocessing import minmax_scale 
from scipy import signal 
import os 

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return data[s<m]

def get_small_dataframe(df, map_dict=None):
    if map_dict is None: 
        return df  
    
    subset_df = df
    for k, v in map_dict.items(): 
        subset_df = subset_df.loc[subset_df[k] == v]

    return subset_df

def _ncs_fft(sig, fs):
    Y = fft(sig)
    L = len(sig)
    L = L - L%2 # Make it even  
    P2 = abs(Y/L) 
    P = P2[0:L//2]
    P[1:-2] = 2*P[1:-2]
    f = fs*np.arange(0, L/2)/L

    return f, P 

def ncs_fft(sig, fs, plot=False, range=None):
    f, P = _ncs_fft(sig, fs)
    
    if range is not None: 
        idx = (range[0] < f) & (f < range[1])
        f = f[idx]
        P = P[idx]
        
    if plot:
        import matplotlib.pyplot as plt
        plt.ion()
        plt.plot(f, P)
        plt.show()
        
    return f, P

def get_peak_harmonic(sig, fs, range=None): 
    f, P = ncs_fft(sig, fs, range=range)
    idx = np.argmax(P)
    
    return f[idx]
                
def align_and_normalize(sig, ref):
    # Given a reference signal, align polarity
    if np.corrcoef(sig, ref) < 0:
        return minmax_scale(max(sig)-sig)
    else:
        return minmax_scale(sig)
    
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

def get_accuracy(cm):
    if type(cm) is dict:
        num_classes = cm[list(cm.keys())[0]].shape[0]
        return _get_dict_accuracy(cm, num_classes)
    else:
        num_classes = cm.shape[0]
        return _get_accuracy(cm, num_classes)    

def _get_accuracy(cm, num_classes):
    return sum([cm[i, i] for i in range(num_classes)])/np.concatenate(cm).sum()

def _get_dict_accuracy(cm, num_classes):
    return np.average([_get_accuracy(cc, num_classes) for _, cc in cm.items()])

def get_total_steps(df, subjects):
    total_steps = 0
    for sub in subjects:
        dft = df.loc[df['subject'] == sub]
        routines = set(dft['routine'])
        total_steps += len(routines) 
    return total_steps

def get_subject_path(base_path, subject_num): 
    DIRNAMES = 1
    get_sub_num = lambda x: int(x[x.index(' ')+1:]) 
    dPath = [os.path.join(base_path, x) for x in next(os.walk(base_path))[DIRNAMES] if get_sub_num(x)==subject_num]
    return dPath[0]