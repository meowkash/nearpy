import numpy as np 
from bottleneck import move_std, move_mean
import ruptures as rpt 
from scipy.interpolate import CubicSpline
from scipy.stats import ecdf 

from .utils import normalize
from .quality import get_snr

def get_adaptive_segment_indices(sig, 
                                 fs: int, 
                                 method: str, 
                                 prob_thresh: float = 0.9, 
                                 sig_band: list =None, 
                                 noise_band: list =None, 
                                 win_size: int =10, 
                                 logarithmic: bool =False): 
    '''
    Depending upon provided input method, return points for segmentation chosen adaptively (CDF > Thresholded value)
    '''
    timeAx = np.linspace(0, len(sig)/fs, len(sig))
    
    if method == 'Abs': 
        proc_sig = np.abs(sig)
    elif method == 'Square': 
        proc_sig = sig**2
    elif method == 'Movstd':
        # Using min_count = 1 ensures that we do not have NaNs in our data
        proc_sig = normalize(move_std(move_mean(sig, win_size, min_count=1), win_size, min_count=1))
    elif method == 'SNR': 
        t, power_ratio = get_snr(sig, fs, 
                                 sig_band=sig_band, 
                                 noise_band=noise_band, 
                                 logarithmic=logarithmic)
        timeAx = np.linspace(0, t[-1], len(sig))
        cs = CubicSpline(t, power_ratio)
        proc_sig = cs(timeAx)
    else: 
        return
    
    cdf = ecdf(proc_sig).cdf
    vals, probs = cdf.quantiles, cdf.probabilities
    thresh = vals[np.where(probs > prob_thresh)[0][0]]
    vals = normalize(vals)
    
    # Plot using thresholded value  
    idx = np.where(proc_sig > thresh)[0]
    marker_indices = timeAx[idx] 
    marker_values = normalize(sig)[idx]
    
    return marker_indices, marker_values, vals, probs

# For a given time series input, get the segment indices 
def get_segment_indices(sig, window, min_samples, kernel='linear', num_points=4):
    sig_var = move_std(sig, window)
    algo = rpt.KernelCPD(kernel=kernel, jump=0.1, min_size=min_samples).fit(sig_var)
    points = algo.predict(n_bkps=num_points)
    
    return points[0], points[-2]