import numpy as np 
from scipy import signal 
from scipy.stats import ecdf 

def get_snr(sig, fs, sig_band, noise_band, nperseg=256, logarithmic=True): 
    '''
    Input Arguments:
        sig = Time-series signal 
        fs = Sampling frequency (Hz)
        sig_band: [lower, upper] = Bounds where the signal is supposed to be 
        noise_band: [lower, upper] = Bounds where the noise is supposed to be 
    
    Optional Arguments: 
        nperseg = Number of segments for STFT
    '''
    sig = sig / np.max(np.abs(sig))
        
    # STFT parameters
    noverlap = nperseg // 2  # 50% overlap between windows

    # Compute STFT
    f, t, Zxx = signal.stft(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Compute power spectrogram (magnitude squared)
    power = np.abs(Zxx) ** 2

    # Find indices for our frequency bands
    noise_band_indices = np.logical_and(f >= np.min(noise_band), f <= np.max(noise_band))
    sig_band_indices = np.logical_and(f >= np.min(sig_band), f <= np.max(sig_band))

    # Calculate power in each band for each time segment
    noise_band_power = np.sum(power[noise_band_indices, :], axis=0)
    sig_band_power = np.sum(power[sig_band_indices, :], axis=0)

    # Calculate ratio (add small constant to avoid division by zero)
    snr = sig_band_power / (noise_band_power + 1e-3)
    
    if logarithmic: 
        snr = 10*np.log10(snr)
        
    return t, snr

# If harmonic = 2, this is the classic linearity metric
def get_harmonic_ratio(sig, fs, sig_band, harmonic=2, nperseg=256, logarithmic=True): 
    '''
    Returns: H1/H[n] power 
    
    Input Arguments:
        sig = Time-series signal 
        fs = Sampling frequency (Hz)
        sig_band: [lower, upper] = Bounds where the signal is supposed to be 
        harmonic: n-th harmonic 
    
    Optional Arguments: 
        nperseg = Number of segments for STFT
    '''
    
    return get_snr(sig=sig, 
                   fs=fs, 
                   sig_band=sig_band, 
                   noise_band=sig_band*harmonic, 
                   nperseg=nperseg, 
                   logarithmic=logarithmic
                )
    
def get_threshold_value(sig, prob_thresh=0.95): 
    assert (prob_thresh>=0 & prob_thresh <=1), f'Probability threshold must be between 0 and 1. Got {prob_thresh} instead'
    
    cdf = ecdf(sig).cdf
    vals, probs = cdf.quantiles, cdf.probabilities
    thresh = vals[np.where(probs > prob_thresh)[0][0]]
    
    return thresh
