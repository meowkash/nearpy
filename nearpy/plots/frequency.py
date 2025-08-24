import numpy as np
import matplotlib.pyplot as plt 
from scipy.fft import fft, fftfreq

from typing import Dict 

def plot_spectrum(data_dict: Dict):
    ''' 
    Given a data dictionary formatted as - 
    signal_name: {
        'data': time series signal,
        'fs': sampling frequency
    } 
    plot respective FFT for each 
    '''
    num_sigs = len(data_dict)
    fig, ax = plt.subplots(num_sigs, 1, figsize=(2*num_sigs, 1.25*num_sigs))

    idx = 0

    for signal_type, plot_data in data_dict.items():
        sig = plot_data['data']
        fs = plot_data['fs']

        N_fft = len(sig)
        Y = fft(sig)
        freqs = fftfreq(N_fft, 1 / fs)[:N_fft // 2]
        
        ax[idx].semilogy(freqs, np.abs(Y[:N_fft // 2]), 'b-', linewidth=1)
        ax[idx].set_ylabel('Magnitude')
        
        if idx == 0:
            ax[idx].set_title(f'{signal_type} Spectrum')
        
        if idx == num_sigs - 1: 
            ax[idx].set_xlabel('Frequency (Hz)')
        ax[idx].grid(True, alpha=0.3)

        idx += 1
    plt.tight_layout()
    plt.show(block=False)