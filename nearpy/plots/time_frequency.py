import pywt
import numpy as np
from pathlib import Path
from ssqueezepy import ssq_cwt
import matplotlib.pyplot as plt 
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming

from typing import Optional, Tuple

def plot_spectrogram( 
    data: np.ndarray, 
    fs: float, 
    nperseg: Optional[int] = None, 
    noverlap: Optional[int] = None,
    export: bool = False, 
    export_dir: str = '',
    figsize = (4, 4),
    dpi = 300 
):
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
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    
    Sx_dB = 10 * np.log10(np.fmax(specgram, 1e-10))
    img = ax.imshow(Sx_dB, origin='lower', aspect='auto',
            extent=SFT.extent(len(data)), cmap='turbo')
    
    plt.tight_layout()    
    if export: 
        ax.set_xticks([])
        ax.set_yticks([])
        # Ensure path always exists
        Path(export_dir).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(export_dir, bbox_inches="tight")
    else: 
        fig.colorbar(img, ax=ax, label='Power Spectral Density (dB)')
        ax.set_xlabel('Time (s)')    
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Spectrogram')
        plt.show(block=False)

    plt.close()

def plot_scalogram(
    data, 
    fs, 
    wavelet='cmor1.5-1.0', 
    scales=None, 
    use_log_scale = False, 
    export=False, 
    export_path: str = '',
    fig = None, 
    ax = None, 
    figsize = (8, 4),
    dpi = 150, 
):
    '''
    Convenience function to plot CWT of provided data. 
    Possible cwt families can be found by using ```pywt.wavelist(kind='continuous')```
    '''
    if scales is None:
        scales = np.arange(1, min(128, len(data)//4))
        
    coeffs, freqs = pywt.cwt(data, scales, wavelet, 1/fs)
    t = np.arange(len(data)) / fs
    
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if use_log_scale:
        coeffs = 20 * np.log10(np.abs(coeffs) + 1e-10)    
    else:
        coeffs = np.abs(coeffs)    
    
    img = ax.imshow(coeffs, aspect='auto', cmap='turbo', extent=[0, t[-1], freqs[-1], freqs[0]])
    # plt.tight_layout()

    if export: 
        ax.set_xticks([])
        ax.set_yticks([])
        # Ensure path always exists
        Path(export_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(export_path, bbox_inches="tight")
    else: 
        if use_log_scale: 
            fig.colorbar(img, ax=ax, label='Magnitude (dB)')
        else:
            fig.colorbar(img, ax=ax, label='Magnitude')

        ax.set_xlabel('Time (s)')    
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'Scalogram, {wavelet}')
        
        plt.show(block=False)

    plt.close()
    
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