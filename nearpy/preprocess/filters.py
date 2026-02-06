# This will maintain a collection of filters used throughout 
# Refer: https://tomverbeure.github.io/2020/10/11/Designing-Generic-FIR-Filters-with-pyFDA-and-Numpy.html

# Filters were made using MATLAB and PyFDA
from scipy.signal import remez, freqz, filtfilt
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import median_filter
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path

from nearpy.io import log_print

def get_gesture_filter(f_s=15, fs=100, visualize=False, logger=None,):
    ''' References: 
    [1]: https://stackoverflow.com/questions/24460718/how-to-use-pythons-scipy-signal-remez-output-for-scipy-signal-lfilter
    '''
    log_print(logger, 'debug', f'Remez (equi-ripple) band-pass filter with pass band {[0.15, f_s]}')
        
    # Band-Pass Filter between 0.1 Hz and 15 Hz
    taps = remez(1415, [0, 0.05 , 0.2, f_s, f_s + 0.2, 0.5*fs], [0, 1, 0], fs=fs)
    w, h = freqz(taps, fs=fs)
    
    if visualize: 
        plot_filter_response(taps, fs=fs)
        
    return taps

def load_filter(filename: str):
    # Check if file location is already a valid path
    if Path(filename).exists() and filename.endswith('.npz'): 
        f_path = Path(filename)
    else:
        f_path = Path(__file__).parent / 'saved_filters' / f'{filename}.npz'
    
    fobj = np.load(f_path)
    
    return fobj['ba']

def filter_and_normalize(sig, filter, axis=0):
    # Given a multi-variate signal array, filter and normalize each variable using z-score normalization 
    filt = lambda x: filtfilt(filter[0], filter[1], x)
    scaler = MinMaxScaler() 
    norm = lambda x: np.transpose(scaler.fit_transform(np.transpose(x)))
    
    return norm(np.apply_along_axis(filt, axis, sig))

def spike_removal_filter(sig, fs, window_size=11):
    '''
    Median filters signal (removes small spikes) as well as first derivative of signal (removes large spikes)
    '''
    # med_sig = median_filter(sig, size=window_size)
    diff_sig = np.diff(sig, prepend=sig[0]) # Compute 1st order derivative while preserving shape
    med_diff = median_filter(diff_sig, size=window_size)
    xx = np.linspace(0, len(med_diff)/fs, len(med_diff))
    med_sig = cumulative_trapezoid(med_diff, xx)
    
    return med_sig

def ncs_filt(sig, n_taps, f_p=0.1, f_s=15, fs=1000, ftype = 'bandpass'):
    # Helper function to maintain compatibility with prior MATLAB scripts
    if ftype == 'lowpass':
        band = [0, f_p, f_s, 0.5*fs]
        gain = [1, 0]
    elif ftype == 'bandpass':
        band = [0, f_p/2 , f_p, f_s, f_s + f_p, 0.5*fs]
        gain = [0, 1, 0]
    elif ftype == 'highpass':
        band = [0, f_s, f_p, 0.5*fs]
        gain = [0, 1]
    else:
        return
    
    if n_taps is None: 
        return None 

    b = remez(n_taps, band, gain, fs=fs)
    return filtfilt(b, 1, sig)
    
def detrend(sig, deg=3, logger=None):
    '''
    Detrend a given signal using a n-degree polynomial. 
    By default, n is chosen to be 3 as it provides the best empirical results.
    '''
    log_print(logger, 'debug', f'Detrending signal with degree {deg} polynomial fit')
    
    t = np.linspace(1, len(sig), len(sig))
    pfit = np.polynomial.Polynomial.fit(t, sig, deg=deg)
    return sig - pfit(t)
    
def plot_filter_response(ba, fs=None):
    # Helper function to provide similar functionality as prior MATLAB scripts
    if fs is not None: 
        w, h = freqz(ba[0], ba[1], fs=fs)
    else: 
        w, h = freqz(ba[0], ba[1])
    
    "Utility function to plot response functions"
    h_mag = 20*np.log10(np.abs(h))
    h_phase = np.unwrap(np.arctan2(np.imag(h), np.real(h)))
    
    fig, ax = plt.subplots(2, 1, figsize=(5, 5), dpi=300)
    # Magnitude response 
    ax[0].plot(w, h_mag, 
               linewidth=1.5,
               color='#3171ad', 
               label='Magnitude Response')
    # ax[0].set_xticks([])
    ax[0].set_ylabel('Gain (dB)', fontsize=12)
    ax[0].legend()
    # Phase response
    ax[1].plot(w, h_phase, 
               linewidth=1.5,
               color='#cc7e4e',
               label='Phase Response')
    ax[1].set_xlabel('Frequency', fontsize=12)
    ax[1].set_ylabel('Phase (rad)', fontsize=12)
    ax[1].legend()
    
    ax[1].sharex(ax[0])

    fig.tight_layout() 
    plt.show() 
