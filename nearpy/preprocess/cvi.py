# Complex-Vector Injection Implementation
import numpy as np 
from scipy.fft import fft, fftfreq
from scipy.spatial.distance import pdist 

def cvi(mag_data, phase_data, vec_offset=0, method='snr', llim=0.1, ulim=0.9, mag_step=3, lplim=0, uplim=2*np.pi, ph_step=200, **kwargs):
    # Generate complex data
    c = np.empty(mag_data.shape, dtype=np.complex128)
    c.real = mag_data * np.cos(phase_data * np.pi/180)
    c.imag = phase_data * np.sin(phase_data * np.pi/180)

    # Re-center
    c = c - complex(min(c.real, c.imag)) 
    # Linear regression to find complex angle
    ang = np.arctan(np.linalg.lstsq(c.real, c.imag))
    # Remove original angle and add desired angle to normalize
    amat = np.empty(c.shape, dtype=np.complex128)
    amat.real = np.cos(ang)
    amat.imag = - np.sin(ang)
    c = c * amat
    # Add offset
    c = c + vec_offset

    # Bulk-CVI code: Find vector to inject 
    inj_vec = get_optimal_injection(c, method, llim, ulim, mag_step, lplim, uplim, ph_step, kwargs)

    # Add injected vector to base vector and extract real signals
    c_opt = c + inj_vec
    
    return abs(c_opt), np.angle(c_opt)

def get_optimal_injection(data, method, llim, ulim, mag_step, lplim, uplim, ph_step, **kwargs):   
    mag_scan = np.linspace(llim, ulim, mag_step)
    ph_scan = np.linspace(lplim, uplim, ph_step)
    
    scores = np.zeros(mag_step, ph_step)

    for midx in range(mag_scan):
        for pidx in range(ph_scan):
            # Get vector to add
            m = mag_scan[midx]
            p = ph_scan[pidx]
            ivec = complex(m*np.cos(p), m*np.sin(p))
            # Inject data
            modd = data + ivec
            # Get score
            scores[midx, pidx] = get_cvi_score(modd, method, kwargs)

    # Return vector with maximum score
    opt_idx = np.unravel_index(np.argmax(scores, axis=None), scores.shape)            
    return mag_scan[opt_idx[0]], ph_scan[opt_idx[1]]

def get_cvi_score(data, method, **kwargs):
    match method:
        case 'snr':
            return snr(data, kwargs.get('fs', 1000), kwargs.get('f_range'), kwargs.get('nf_range', [10, 50])) 
        case 'sim':
            return sim(data, kwargs.get('seg_len', 1000))
        case 'temp':
            return sim(data, kwargs.get('seg_len', 1000), kwargs.get('temp'))
        case _:
            return 0
    
def sim(data, seg_len, temp=None):
    segs = np.split(data, seg_len)
    if temp is not None: 
        return -np.median([(dat - temp)**2 for dat in segs])
    else:
        return -np.median(pdist(segs, 'euclidean'))    

def snr(data, fs, f_range, nf_range):
    l_sig = len(data)
    fft_signal = fft(data) # Take FFT of magnitude data
    fft_norm = np.abs(fft_signal/l_sig)[0:round(l_sig/2)]
    freq = fftfreq(l_sig, 1/fs)[0:round(l_sig/2)] # Get the frequency bin centers

    # Compute SNR
    max_fund = fft_norm[(f_range[0] <= freq) & (f_range[1] >= freq)].max()
    avg_noise = fft_norm[(nf_range[0] <= freq) & (nf_range[1] >= freq)].mean()

    return 10 * np.log10(max_fund/avg_noise)