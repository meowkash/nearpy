import pywt
import numpy as np
from ssqueezepy import ssq_cwt
import matplotlib.pyplot as plt 
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming

from typing import Optional, Tuple

def plot_spectrogram(self, data: np.ndarray, fs: float, 
                        nperseg: Optional[int] = None, noverlap: Optional[int] = None,
                        title: str = "Spectrogram", figsize: Tuple[int, int] = (12, 6)):
    """Plot STFT spectrogram"""
    if nperseg is None:
        nperseg = min(256, len(data) // 8)
    if noverlap is None:
        noverlap = nperseg // 2
        
    win = hamming(nperseg)
    nfft = max(256, nperseg)
    
    SFT = ShortTimeFFT(
        win=win, hop=nperseg-noverlap, fs=fs,
        fft_mode='onesided', mfft=nfft,
        scale_to='magnitude'
    )
    
    specgram = SFT.spectrogram(data)
    t_lo, t_hi = SFT.extent(len(data))[:2]
    
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    Sx_dB = 10 * np.log10(np.fmax(specgram, 1e-10))
    im = ax.imshow(Sx_dB, origin='lower', aspect='auto',
                    extent=SFT.extent(len(data)), cmap=self.cmap)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    ax.set_xlim(t_lo, t_hi)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Power Spectral Density (dB)')
    
    plt.tight_layout()
    return fig, ax

def plot_scalogram(self, data: np.ndarray, fs: float, 
                      wavelet: str = 'cmor1.5-1.0', scales: Optional[np.ndarray] = None,
                      title: str = "Scalogram", figsize: Tuple[int, int] = (12, 6)):
    """Plot CWT scalogram"""
    if scales is None:
        scales = np.arange(1, min(128, len(data)//4))
        
    coeffs, freqs = pywt.cwt(data, scales, wavelet, 1/fs)
    t = np.arange(len(data)) / fs
    
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    # Convert to dB scale
    coeffs_dB = 20 * np.log10(np.abs(coeffs) + 1e-10)
    
    im = ax.imshow(coeffs_dB, aspect='auto', cmap=self.cmap,
                    extent=[0, t[-1], freqs[-1], freqs[0]])
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Magnitude (dB)')
    
    plt.tight_layout()
    return fig, ax
    
def plot_sst(self, data: np.ndarray, fs: float,
            wavelet: str = 'morlet', gamma: float = 5.0,
            title: str = "Synchrosqueezed Transform", figsize: Tuple[int, int] = (12, 6)):
    """Plot Synchrosqueezed Transform"""
    # Compute SST
    Tx, Wx, *_ = ssq_cwt(data, wavelet=wavelet, gamma=gamma, fs=fs)
    
    # Create frequency and time axes
    scales = np.arange(1, min(128, len(data)//4))
    freqs = pywt.scale2frequency(wavelet, scales) * fs
    t = np.arange(len(data)) / fs
    
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    # Convert to dB scale
    Tx_dB = 20 * np.log10(np.abs(Tx) + 1e-10)
    
    im = ax.imshow(Tx_dB, aspect='auto', cmap=self.cmap,
                    extent=[0, t[-1], freqs[-1], freqs[0]])
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Magnitude (dB)')
    
    plt.tight_layout()
    return fig, ax