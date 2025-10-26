import numpy as np 
import neurokit2 as nk 

from bottleneck import move_std, move_mean
from tslearn.barycenters import softdtw_barycenter as DBA
from scipy.interpolate import CubicSpline
from scipy.stats import ecdf 

from typing import Dict, List

from nearpy.utils import normalize
from nearpy.io import log_print
from .quality import get_snr

def get_segments_template(segments, method: str): 
    '''
    Assuming a list of segments, return a template. This will work with multivariable signals as well.

    Inputs: 
        - segments: np.ndarray with shape (num_segs, num_variables, seg_len)
        - method: str = ['mean', 'dba']

    Output: 
        - template: np.ndarray with shape (num_variables, seg_len)
    '''
    template = None 

    if method == 'mean': 
        template = np.mean(segments, axis=0)
    elif method == 'dba': 
        template = DBA(segments)
    else: 
        print(f'Template method {method} not currently implemented.')
    
    return template

def segment_multibeat_timeseries(
    sensor_data: Dict[str, Dict[str, np.ndarray]], 
    reference_sensor: str,
    reference_locs: np.ndarray,
    num_beats: int = 1
) -> Dict[str, List[Dict[str, np.ndarray]]]:
    """
    Segment time series data from multiple sensors into multi-beat segments.
    
    Parameters:
    -----------
    sensor_data : dict
        Dictionary with sensor names as keys, each containing:
        {'time': np.array, 'data': np.array}
    reference_sensor : str
        Name of the reference sensor used to define segment locations
    reference_locs : np.ndarray
        Array of indices in the reference sensor where beats occur
    num_beats : int, default=1
        Number of beats per segment
    return_time : bool, default=True
        Whether to return time arrays along with data segments
    
    Returns:
    --------
    list
        Dictionary with sensor names as keys, each containing a list of segments.
    
    Example:
    --------
    >>> # Sample data
    >>> sensor_data = {
    ...     'ecg': {'time': np.linspace(0, 10, 1000), 'data': np.sin(2*np.pi*np.linspace(0, 10, 1000))},
    ...     'ppg': {'time': np.linspace(0, 10, 1200), 'data': np.cos(2*np.pi*np.linspace(0, 10, 1200))}
    ... }
    >>> reference_locs = np.array([100, 200, 300, 400, 500])
    >>> segments = segment_multibeat_timeseries(sensor_data, 'ecg', reference_locs, num_beats=2)
    """
    
    if reference_sensor not in sensor_data:
        raise ValueError(f"Reference sensor '{reference_sensor}' not found in sensor_data")
    
    if len(reference_locs) < num_beats + 1:
        raise ValueError(f"Need at least {num_beats + 1} reference locations for {num_beats}-beat segments")
    
    # Get reference time array
    ref_time = sensor_data[reference_sensor]['time']
    
    # Find aligned locations for all sensors
    aligned_locs = {}
    
    for sensor_name, sensor_info in sensor_data.items():
        if sensor_name == reference_sensor:
            # Use original locations for reference sensor
            aligned_locs[sensor_name] = reference_locs
        else:
            # Find closest time points in this sensor
            sensor_time = sensor_info['time']
            ref_times_at_locs = ref_time[reference_locs]
            
            # For each reference time, find the closest time in current sensor
            aligned_locs[sensor_name] = np.array([
                np.argmin(np.abs(sensor_time - ref_t)) 
                for ref_t in ref_times_at_locs
            ])
    
    # Generate segments for all sensors
    all_segments = {}
    
    for sensor_name, sensor_info in sensor_data.items():
        sensor_time = sensor_info['time']
        sensor_data_array = sensor_info['data']
        locs = aligned_locs[sensor_name]
        
        segments = []
        
        # Create multi-beat segments
        for i in range(len(locs) - num_beats):
            start_idx = locs[i]
            end_idx = locs[i + num_beats]
            
            # Extract segment
            segment_data = sensor_data_array[start_idx:end_idx]
            
            segments.append(segment_data)
        
        all_segments[sensor_name] = segments
    
    return all_segments
    
# Prepares n-beat datasets from raw waveforms, not pre-segmented waveforms
def get_ecg_segment_locs(t_ecg, ecg, plot: bool = False): 
    # Get all ECG segmented data 
    signals, info = nk.ecg_process(ecg, sampling_rate=1/(t_ecg[1]-t_ecg[0]))
    
    if plot: 
        import matplotlib.pyplot as plt 
        nk.ecg_plot(signals, info)
        fig = plt.gcf()
        fig.set_dpi(192)
        fig.set_size_inches(10, 8, forward=True)
        
    r_peak_locs = np.where(signals['ECG_R_Peaks'] == 1) 
    
    return np.squeeze(r_peak_locs)

def segment_data(data: Dict, samp_rate: Dict, seg_len: float, num_segs: int, seg_type: str = 'time'): 
    ''' 
    General wrapper function which segments all data present in the dictionary into equal sized chunks 
    (assuming their sampling frequencies are provided as well)  

    Inputs: 
    - data: Data (with labels expected to be the same as for samp_rate)
    - samp_rate: Sampling frequencies (for each sensor type)
    - seg_len: Length of each individual segment (in seconds) 
    - num_segs: Number of segments
    '''
    segments = {}

    for key, datastream in data.items(): 
        seg_len = int(seg_len * samp_rate[key])
        if seg_type == 'time':
            segments[key] = get_time_based_segments(datastream, seg_len, num_segs)
        else: 
            log_print(f'Segmentation method "{seg_type}" is not currently supported.')
            continue
        
    return segments

def get_time_based_segments(signal, seg_len, num_seg: int = None):
    # Given a signal of some length, divide it into chunks of length seg_len.
    # This is basically the same as np.split but with some nice-to-haves.
    if num_seg is None:
        num_seg = signal // seg_len

    # Since seg_len may be a float, ensure we round it to the nearest integer
    cropped_signal = np.take(signal, range(int(seg_len*num_seg)), axis=-1)

    return np.split(cropped_signal, num_seg, axis=-1)

def get_adaptive_segment_indices(
        sig, 
        timeAx, 
        fs: int, 
        method: str, 
        prob_thresh: float = 0.9, 
        sig_band: list = None, 
        noise_band: list = None, 
        win_size: int = 10, 
        logarithmic: bool = False,
        max_gap: int = 10,
        padding: int = 0.05
): 
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