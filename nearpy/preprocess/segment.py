import numpy as np 
from bottleneck import move_std, move_mean
import ruptures as rpt 
from scipy.interpolate import CubicSpline
from scipy.stats import ecdf 

from .utils import normalize
from .quality import get_snr

# Not really needed but kept as reference implementation
def split_timewise(sig, fs, start_time, end_time, num_segs): 
    return np.array_split(sig[start_time*fs:end_time*fs], num_segs)

def get_adaptive_segment_indices(sig, 
                                 timeAx, 
                                 fs: int, 
                                 method: str, 
                                 prob_thresh: float = 0.9, 
                                 sig_band: list = None, 
                                 noise_band: list = None, 
                                 win_size: int = 10, 
                                 logarithmic: bool = False,
                                 max_gap: int = 10,
                                 padding: int = 0.05): 
    '''
    Depending upon provided input method, return points for segmentation chosen adaptively (CDF > Thresholded value)
    '''
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
    
    # Return markers for plotting
    idx = np.where(proc_sig > thresh)[0]
    
    # Only return the largest contiguous block 
    if len(idx) == 0:
        return np.array([]), vals, probs
    
    # Find continuous blocks with maximum allowed gap
    blocks = []
    current_block = [idx[0]]
    
    for i in range(1, len(idx)):
        if idx[i] - idx[i-1] <= max_gap:
            # This index is within max_gap of the previous one
            current_block.append(idx[i])
        else:
            # Gap is larger than max_gap, start a new block
            blocks.append(current_block)
            current_block = [idx[i]]
    
    # Add the last block
    if current_block:
        blocks.append(current_block)
    
    # Find the largest block
    largest_block = max(blocks, key=len)
    
    # Add a little padding if specified
    if padding is not None:
        idx[0] = max(idx[0] - padding*fs, 0)
        idx[-1] = min(idx[-1] + padding*fs, len(sig)-1)
        
    # Fill in any gaps within the block
    idx = np.array(largest_block)
    if len(idx) > 1:
        # Create a fully continuous block by filling gaps
        start = idx[0]
        end = idx[-1]
        idx = np.arange(start, end + 1)
        
    return idx, vals, probs

# For a given time series input, get the segment indices 
def get_segment_indices(sig, window, min_samples, kernel='linear', num_points=4):
    sig_var = move_std(sig, window)
    algo = rpt.KernelCPD(kernel=kernel, jump=0.1, min_size=min_samples).fit(sig_var)
    points = algo.predict(n_bkps=num_points)
    
    return points[0], points[-2]