# This will maintain a collection of filters used throughout 
# Refer: https://tomverbeure.github.io/2020/10/11/Designing-Generic-FIR-Filters-with-pyFDA-and-Numpy.html
# Filters were made using MATLAB and PyFDA
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import remez, freqz, filtfilt
import numpy as np
import matplotlib.pyplot as plt 

from ..utils import logprint

def get_gesture_filter(f_s=15, fs=100, visualize=False, logger=None,):
    ''' References: 
        [1]: https://stackoverflow.com/questions/24460718/how-to-use-pythons-scipy-signal-remez-output-for-scipy-signal-lfilter
    '''
    logprint(logger, 'debug', f'Remez (equi-ripple) band-pass filter with pass band {[0.15, f_s]}')
        
    # Band-Pass Filter between 0.1 Hz and 15 Hz
    taps = remez(1415, [0, 0.05 , 0.2, f_s, f_s + 0.2, 0.5*fs], [0, 1, 0], fs=fs)
    w, h = freqz(taps, fs=fs)
    
    if visualize: 
        plot_response(w, h)
        
    return taps

def filter_and_normalize(sig, filter, axis=0):
    # Given a multi-variate signal array, filter and normalize each variable using z-score normalization 
    filt = lambda x: filtfilt(filter[0], filter[1], x)
    scaler = MinMaxScaler() 
    norm = lambda x: np.transpose(scaler.fit_transform(np.transpose(x)))
    
    return norm(np.apply_along_axis(filt, axis, sig))
    
# TODO: Jitter Removal - Use median filtering to remove small jitters 
# TODO: Spike Removal - Use differential integral technique to remove sudden jumps due to ADC - Tweak Thresholding to ensure this does not impact the morphology of acquired signal 

def ncs_filt(sig, n_taps, f_p=0.1, f_s=15, fs=1000, ftype = 'bandpass'):
    if ftype == 'lowpass':
        band = [0, f_p, f_s, 0.5*fs]
        gain = [1, 0]
    elif ftype == 'bandass':
        band = [0, f_p/2 , f_p, f_s, f_s + f_p, 0.5*fs]
        gain = [0, 1, 0]
    elif ftype == 'highpass':
        band = [0, f_p, f_s, 0.5*fs]
        gain = [0, 1]
    
    if n_taps is None: 
        return None 

    b = remez(n_taps, band, gain, fs=fs)
    return filtfilt(b, 1, sig)
    
def detrend(sig, deg=3, logger=None):
    '''
    Detrend a given signal using a n-degree polynomial. By default, n is chosen to be 3 as it provides the best empirical results
    '''
    logprint(logger, 'debug', f'Detrending signal with degree {deg} polynomial fit')
    
    t = np.linspace(1, len(sig), len(sig))
    pfit = np.polynomial.Polynomial.fit(t, sig, deg=deg)
    return sig - pfit(t)
    
def spike_filter(sig):
    # Remove small spikes using medfilt1
    
    return sig

def dB20(array):
    with np.errstate(divide='ignore'):
        return 20 * np.log10(array)
    
def plot_response(w, h, title='Filter Response'):
    "Utility function to plot response functions"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(w, 20*np.log10(np.abs(h)))
    ax.grid(True)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Gain (dB)')
    
    ax.set_title(title)