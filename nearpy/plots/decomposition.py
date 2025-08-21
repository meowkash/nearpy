'''
Using a combination of decomposition methods, display scalograms corresponding to each IMF in a standardized format
'''

import pywt
import numpy as np
import matplotlib.pyplot as plt

from PyEMD import EMD
from vmdpy import VMD

from nearpy.preprocess import lcd_decomposition, lmd_decomposition, hvd_decomposition, ewt_decomposition

# ========== EMPIRICAL MODE DECOMPOSITION (EMD) ==========
def emd_cwt_analysis(signal: np.ndarray, fs: int, wavelet: str = 'cmor1.5-1.0'):
    # EMD decomposition
    emd = EMD()
    imfs = emd.emd(signal)
    
    # CWT of each IMF
    scales = np.logspace(0.5, 3, 80)
    fig, axes = plt.subplots(len(imfs), 1, figsize=(12, 2*len(imfs)))
    
    for i, imf in enumerate(imfs):
        coeffs, freqs = pywt.cwt(imf, scales, wavelet, sampling_period=1/fs)
        power = np.abs(coeffs)**2
        power_norm = power / np.max(power)
        
        t = np.linspace(0, len(signal)/fs, len(signal))
        axes[i].imshow(power_norm, aspect='auto', cmap='turbo',
                      extent=[0, len(signal)/fs, freqs[-1], freqs[0]])
        axes[i].set_yscale('log')
        axes[i].set_title(f'EMD IMF {i+1}')
        axes[i].set_ylabel('Freq (Hz)')
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    return imfs

# ========== VARIATIONAL MODE DECOMPOSITION (VMD) ==========
def vmd_cwt_analysis(signal, fs: int, wavelet: str = 'cmor1.5-1.0', K: int = 6,):
    # VMD decomposition
    u, _, _ = VMD(signal, alpha=2000, tau=0, K=K, DC=0, init=1, tol=1e-7)
    
    # CWT of each mode
    scales = np.logspace(0.5, 3, 80)
    fig, axes = plt.subplots(K, 1, figsize=(12, 2*K))
    
    for i, mode in enumerate(u):
        coeffs, freqs = pywt.cwt(mode, scales, wavelet, sampling_period=1/fs)
        power = np.abs(coeffs)**2
        power_norm = power / np.max(power)
        
        axes[i].imshow(power_norm, aspect='auto', cmap='turbo',
                      extent=[0, len(signal)/fs, freqs[-1], freqs[0]])
        axes[i].set_yscale('log')
        axes[i].set_title(f'VMD Mode {i+1}')
        axes[i].set_ylabel('Freq (Hz)')
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    return u

# ========== WAVELET DECOMPOSITION ==========
def wavedec_cwt_analysis(signal, fs:int, wavelet: str = 'db4', levels: int = 6):
    # Wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=levels)
    
    # Reconstruct components
    components = []
    labels = []
    
    # Approximation (lowest frequency)
    approx_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    approx = pywt.waverec(approx_coeffs, wavelet)[:len(signal)]
    components.append(approx)
    labels.append(f'A{levels}')
    
    # Details (high to low frequency)
    for i in range(1, len(coeffs)):
        detail_coeffs = [np.zeros_like(coeffs[0])] + [np.zeros_like(c) for c in coeffs[1:]]
        detail_coeffs[i] = coeffs[i]
        detail = pywt.waverec(detail_coeffs, wavelet)[:len(signal)]
        components.append(detail)
        labels.append(f'D{levels-i+1}')
    
    # CWT of each component
    scales = np.logspace(0.5, 3, 80)
    fig, axes = plt.subplots(len(components), 1, figsize=(12, 2*len(components)))
    
    for i, (comp, label) in enumerate(zip(components, labels)):
        coeffs, freqs = pywt.cwt(comp, scales, 'cmor1.5-1.0', sampling_period=1/fs)
        power = np.abs(coeffs)**2
        power_norm = power / np.max(power)
        
        axes[i].imshow(power_norm, aspect='auto', cmap='turbo',
                      extent=[0, len(signal)/fs, freqs[-1], freqs[0]])
        axes[i].set_yscale('log')
        axes[i].set_title(f'Wavelet {label}')
        axes[i].set_ylabel('Freq (Hz)')
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    return components, labels

# ========== LOCAL MEAN DECOMPOSITION (LMD) ==========
def lmd_cwt_analysis(signal, fs: int, wavelet: str = 'cmor1.5-1.0', max_pf=8):
    # LMD decomposition
    pfs = lmd_decomposition(signal, max_pf=max_pf)
    
    # CWT of each PF
    scales = np.logspace(0.5, 3, 80)
    fig, axes = plt.subplots(len(pfs), 1, figsize=(12, 2*len(pfs)))
    if len(pfs) == 1:
        axes = [axes]
    
    for i, pf in enumerate(pfs):
        coeffs, freqs = pywt.cwt(pf, scales, wavelet, sampling_period=1/fs)
        power = np.abs(coeffs)**2
        power_norm = power / np.max(power) if np.max(power) > 0 else power
        
        axes[i].imshow(power_norm, aspect='auto', cmap='turbo',
                      extent=[0, len(signal)/fs, freqs[-1], freqs[0]])
        axes[i].set_yscale('log')
        axes[i].set_title(f'LMD PF {i+1}')
        axes[i].set_ylabel('Freq (Hz)')
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    return pfs

# ========== LOCAL CHARACTERISTIC SCALE DECOMPOSITION (LCD) ==========
def lcd_cwt_analysis(signal, fs: int, wavelet: str = 'cmor1.5-1.0', max_scales=8):
    # LCD decomposition
    isfs = lcd_decomposition(signal, max_scales=max_scales)
    
    # CWT of each ISF
    scales = np.logspace(0.5, 3, 80)
    fig, axes = plt.subplots(len(isfs), 1, figsize=(12, 2*len(isfs)))
    if len(isfs) == 1:
        axes = [axes]
    
    for i, isf in enumerate(isfs):
        coeffs, freqs = pywt.cwt(isf, scales, wavelet, sampling_period=1/fs)
        power = np.abs(coeffs)**2
        power_norm = power / np.max(power) if np.max(power) > 0 else power
        
        axes[i].imshow(power_norm, aspect='auto', cmap='turbo',
                      extent=[0, len(signal)/fs, freqs[-1], freqs[0]])
        axes[i].set_yscale('log')
        axes[i].set_title(f'LCD ISF {i+1}')
        axes[i].set_ylabel('Freq (Hz)')
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    return isfs

# ========== HILBERT VIBRATION DECOMPOSITION (HVD) ==========
def hvd_cwt_analysis(signal, fs: int, wavelet: str = 'cmor1.5-1.0', num_components: int = 6):
    # HVD decomposition
    components = hvd_decomposition(signal, num_components=num_components, fs=fs)
    
    # CWT of each component
    scales = np.logspace(0.5, 3, 80)
    fig, axes = plt.subplots(len(components), 1, figsize=(12, 2*len(components)))
    if len(components) == 1:
        axes = [axes]
    
    for i, comp in enumerate(components):
        coeffs, freqs = pywt.cwt(comp, scales, wavelet, sampling_period=1/fs)
        power = np.abs(coeffs)**2
        power_norm = power / np.max(power) if np.max(power) > 0 else power
        
        axes[i].imshow(power_norm, aspect='auto', cmap='turbo',
                      extent=[0, len(signal)/fs, freqs[-1], freqs[0]])
        axes[i].set_yscale('log')
        label = f'HVD Comp {i+1}' if i < len(components)-1 else 'HVD Residual'
        axes[i].set_title(label)
        axes[i].set_ylabel('Freq (Hz)')
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    return components

# ========== EMPIRICAL WAVELET TRANSFORM (EWT) ==========
def ewt_cwt_analysis(signal, fs: int, wavelet: str = 'cmor1.5-1.0', num_modes: int = 6):
    # EWT decomposition
    modes = ewt_decomposition(signal, num_modes=num_modes, fs=fs)
    
    # CWT of each mode
    scales = np.logspace(0.5, 3, 80)
    fig, axes = plt.subplots(len(modes), 1, figsize=(12, 2*len(modes)))
    if len(modes) == 1:
        axes = [axes]
    
    for i, mode in enumerate(modes):
        coeffs, freqs = pywt.cwt(mode, scales, wavelet, sampling_period=1/fs)
        power = np.abs(coeffs)**2
        power_norm = power / np.max(power) if np.max(power) > 0 else power
        
        axes[i].imshow(power_norm, aspect='auto', cmap='turbo',
                      extent=[0, len(signal)/fs, freqs[-1], freqs[0]])
        axes[i].set_yscale('log')
        axes[i].set_title(f'EWT Mode {i+1}')
        axes[i].set_ylabel('Freq (Hz)')
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    return modes