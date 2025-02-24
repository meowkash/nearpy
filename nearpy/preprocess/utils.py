import numpy as np 

from scipy import signal 
from scipy.fft import fft
from sklearn.preprocessing import minmax_scale 

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

