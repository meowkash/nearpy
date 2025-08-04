import numpy as np
import matplotlib.pyplot as plt 
from scipy.fft import fft, fftfreq

from typing import List 

def plot_spectrum(data: np.ndarray, fs: List):
    ''' 
    Given a number of time series signal(s), plot respective FFT
    '''
    assert data.shape[0] == len(fs), f'Each input signal must have a corresponding sampling frequency. \nSignals found: {data.shape[0]}, Sampling Frequencies found: {len(fs)}'

    num_sigs = len(fs)
    fig, ax = plt.subplots(num_sigs, 1, figsize=(2*num_sigs, 1.25*num_sigs))

    for idx in range(num_sigs): 
        sig = data[idx, :]
        N_fft = len(sig)
        Y = fft(sig)
        freqs = fftfreq(N_fft, 1 / fs[idx])[:N_fft // 2]

        ax[idx].semilogy(freqs, np.abs(Y[:N_fft // 2]), 'b-', linewidth=1)
        ax[idx].set_ylabel('Magnitude')
        
        if idx == 0:
            ax[idx].set_title('Spectrum')
        
        if idx == num_sigs - 1: 
            ax[idx].set_xlabel('Frequency (Hz)')
        ax[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=False)