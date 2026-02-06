import numpy as np 
from scipy.signal import detrend 
from scipy.stats import ecdf 

def get_snr(ncs_sig, fs, sig_band, noise_band, use_db: bool = True):
    """
    Parameters:
    ncs_sig (np.ndarray): The input signal (Non-Contact Sensing/Speech signal)
    fs (int/float): Sampling frequency in Hz
    f_range (tuple/list): [low_freq, high_freq] for the peak search
    
    Returns:
    float: SNR score in dB
    """
    l_sig = len(ncs_sig)
    y_sig = np.fft.fft(detrend(ncs_sig))
    
    # Compute Two-sided Spectrum (P2) and Single-sided (P1)
    p2 = np.abs(y_sig / l_sig)
    n_half = l_sig // 2 + 1
    p1 = p2[:n_half].copy()
    
    # Energy conservation: Double the magnitudes of non-DC/non-Nyquist bins
    p1[1:-1] = 2 * p1[1:-1]
    f = np.linspace(0, fs / 2, n_half)
    
    # Extract peak magnitude in signal and noise bands 
    mask_peak = (f > sig_band[0]) & (f < sig_band[1])
    if not np.any(mask_peak):
        print(f'Invalid signal band provided: {sig_band}')
        return None
    mag_1_peak = np.max(p1[mask_peak])

    mask_interf = (f > noise_band[0]) & (f < noise_band[1])
    if not np.any(mask_interf):
        interf1 = 1e-12 # Prevent division by zero
    else:
        interf1 = np.median(p1[mask_interf])
    
    score = mag_1_peak / interf1
    
    if use_db: 
        return 10 * np.log10(score)
    else: 
        return score

# If harmonic = 2, this is the classic linearity metric
def get_harmonic_ratio(sig, fs, sig_band, harmonic=2, use_db: bool =True): 
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
    
    return get_snr(
        sig=sig, 
        fs=fs, 
        sig_band=sig_band, 
        noise_band=sig_band*harmonic, 
        use_db=use_db
    )
    
def get_adaptive_threshold(sig, prob_thresh=0.95): 
    assert (prob_thresh>=0 & prob_thresh <=1), f'Probability threshold must be between 0 and 1. Got {prob_thresh} instead'
    
    cdf = ecdf(sig).cdf
    vals, probs = cdf.quantiles, cdf.probabilities
    thresh = vals[np.where(probs > prob_thresh)[0][0]]
    
    return thresh
