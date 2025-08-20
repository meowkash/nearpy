''' 
Implements common decomposition methods that may not be available using external libraries
'''
import numpy as np 
from scipy.signal import hilbert, savgol_filter, find_peaks, butter, sosfilt
from scipy.fft import fft, fftfreq

def lcd_decomposition(data, max_scales=10, tol=1e-6):
    """
    Local Characteristic Scale Decomposition
    """
    def characteristic_scale(x):
        # Simple characteristic scale based on zero crossings
        zero_crossings = np.where(np.diff(np.sign(x)))[0]
        if len(zero_crossings) < 2:
            return len(x)
        return np.mean(np.diff(zero_crossings)) * 2
    
    isfs = []  # Intrinsic Scale Functions
    residue = data.copy()
    
    for scale_idx in range(max_scales):
        if np.std(residue) < tol:
            break
        
        # Calculate characteristic scale
        char_scale = int(characteristic_scale(residue))
        char_scale = max(3, min(char_scale, len(residue)//4))
        
        # Smooth the signal
        if char_scale >= len(residue):
            isfs.append(residue)
            break
            
        # Moving average filter
        smoothed = np.convolve(residue, np.ones(char_scale)/char_scale, mode='same')
        
        # Extract ISF
        isf = residue - smoothed
        isfs.append(isf)
        
        # Update residue
        residue = smoothed
        
        if np.std(isf) < tol:
            break
    
    isfs.append(residue)  # Final residue
    return isfs

def lmd_decomposition(data, max_pf=10, tol=1e-6, max_iter=100):
    """
    Local Mean Decomposition - decomposes into Product Functions (PFs)
    """
    def find_extrema(x):
        from scipy.signal import find_peaks
        maxima, _ = find_peaks(x)
        minima, _ = find_peaks(-x)
        return maxima, minima
    
    pfs = []
    residue = data.copy()
    
    for pf_idx in range(max_pf):
        if np.std(residue) < tol:
            break
            
        h = residue.copy()
        
        for iter_count in range(max_iter):
            maxima, minima = find_extrema(h)
            
            if len(maxima) < 2 or len(minima) < 2:
                break
            
            # Calculate local mean
            max_env = np.interp(range(len(h)), maxima, h[maxima])
            min_env = np.interp(range(len(h)), minima, h[minima])
            local_mean = (max_env + min_env) / 2
            
            # Calculate local magnitude  
            local_mag = (max_env - min_env) / 2
            local_mag[local_mag == 0] = 1e-10  # Avoid division by zero
            
            # Update h
            h_new = (h - local_mean) / local_mag
            
            if np.mean(np.abs(h_new - h)) < tol:
                break
            h = h_new
        
        pfs.append(h)
        
        # Calculate instantaneous amplitude
        max_env = np.interp(range(len(residue)), maxima, residue[maxima]) if len(maxima) > 1 else np.ones_like(residue)
        min_env = np.interp(range(len(residue)), minima, residue[minima]) if len(minima) > 1 else np.ones_like(residue)
        inst_amp = (max_env - min_env) / 2
        
        # Update residue
        residue = residue - h * inst_amp
    
    pfs.append(residue)  # Final residue
    return pfs

def hvd_decomposition(data, fs: int, num_components=6, freq_range=None):
    """
    Hilbert Vibration Decomposition based on instantaneous frequency
    """
    if freq_range is None:
        freq_range = (1, fs/4)
    
    # Get instantaneous frequency using Hilbert transform
    analytic_signal = hilbert(data)
    inst_phase = np.unwrap(np.angle(analytic_signal))
    inst_freq = np.diff(inst_phase) / (2*np.pi) * fs
    inst_freq = np.append(inst_freq, inst_freq[-1])  # Keep same length
    
    # Create frequency bands
    freq_bands = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), num_components+1)
    
    components = []
    
    for i in range(num_components):
        # Design bandpass filter for this frequency band
        low_freq = freq_bands[i]
        high_freq = freq_bands[i+1]
        
        # Butterworth bandpass filter
        sos = butter(4, [low_freq, high_freq], btype='band', fs=fs, output='sos')
        filtered = sosfilt(sos, data)
        
        components.append(filtered)
    
    # Add residual (very low frequency component)
    sos_low = butter(4, freq_bands[0], btype='low', fs=fs, output='sos')
    residual = sosfilt(sos_low, data)
    components.append(residual)
    
    return components

def ewt_decomposition(data: np.ndarray, fs: int, num_modes=6):
    """
    Empirical Wavelet Transform - adaptive wavelet construction
    """
    # FFT of the signal
    N = len(data)
    freq_axis = fftfreq(N, 1/fs)[:N//2]
    fft_signal = np.abs(fft(data))[:N//2]
    
    # Smooth the spectrum
    fft_smooth = savgol_filter(fft_signal, window_length=min(51, N//10), polyorder=3)
    
    # Find local maxima in the spectrum
    peaks, _ = find_peaks(fft_smooth, height=np.max(fft_smooth)*0.1)
    
    if len(peaks) == 0:
        peaks = np.array([N//4])  # Default peak if none found
    
    # Select the most significant peaks
    peak_heights = fft_smooth[peaks]
    sorted_indices = np.argsort(peak_heights)[::-1]
    num_selected = min(num_modes-1, len(peaks))
    selected_peaks = peaks[sorted_indices[:num_selected]]
    selected_peaks = np.sort(selected_peaks)
    
    # Create frequency boundaries
    boundaries = [0]
    for peak in selected_peaks:
        freq_val = freq_axis[peak]
        if 0 < freq_val < fs/2:  # Valid frequency range
            boundaries.append(freq_val)
    boundaries.append(fs/2 - 1)  # Ensure below Nyquist
    boundaries = np.unique(boundaries)
    boundaries = np.sort(boundaries)
    
    # Ensure minimum separation between boundaries
    min_sep = fs / (10 * len(boundaries))
    for i in range(1, len(boundaries)):
        if boundaries[i] - boundaries[i-1] < min_sep:
            boundaries[i] = boundaries[i-1] + min_sep
    
    # Create modes using bandpass filtering
    modes = []
    
    for i in range(len(boundaries)-1):
        low_freq = boundaries[i]
        high_freq = boundaries[i+1]
        
        # Ensure valid frequency range
        if high_freq >= fs/2:
            high_freq = fs/2 - 1
        if low_freq >= high_freq:
            continue
            
        try:
            if low_freq <= 1:  # Very low frequency, use highpass instead
                sos = butter(4, high_freq, btype='low', fs=fs, output='sos')
            elif high_freq >= fs/2 - 1:  # Very high frequency, use lowpass  
                sos = butter(4, low_freq, btype='high', fs=fs, output='sos')
            else:
                # Band-pass filter
                sos = butter(4, [low_freq, high_freq], btype='band', fs=fs, output='sos')
            
            filtered = sosfilt(sos, data)
            modes.append(filtered)
        except ValueError:
            # Skip invalid frequency ranges
            continue
    
    # If no valid modes created, return original signal
    if len(modes) == 0:
        modes = [data]
    
    return modes