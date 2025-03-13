import numpy as np 
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming
import matplotlib.pyplot as plt 

def get_spectrogram(sig, fs, seg_frac=4.5, perc_overlap=0.5, visualize=False): 
    # We use the same defaults that matlab uses for consistency across code-bases 
    N = len(sig) 
    nperseg = np.floor(N/seg_frac) # Divide signal into segments of fixed length
    noverlap = np.floor(perc_overlap*nperseg) # Define overlap between contiguous segments 
    win = hamming(nperseg)
    nfft = max(256, 2**np.log2(nperseg)) # Compute number of points to take FFT on 
    
    # Define STFT object to get STFT and Spectrogram
    SFT = ShortTimeFFT.from_window(win, fs, nperseg, noverlap, fft_mode='centered',
                               mfft=nfft, scale_to='magnitude', phase_shift=None)
    specgram = SFT.spectrogram(sig)

    if visualize: 
        fig, ax = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit
        t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
        
        ax.set_title('Spectrogram')
        ax.set(xlabel='Time (s)', ylabel='Frequency (Hz)', 
                xlim=(t_lo, t_hi))
        
        Sx_dB = 10 * np.log10(np.fmax(specgram, 1e-4))  # limit range to -40 dB
        im = ax.imshow(Sx_dB, origin='lower', aspect='auto',
                        extent=SFT.extent(N), cmap='turbo')
        fig.colorbar(im, label='Power Spectral Density (dB)')
        
        # Shade areas where window slices stick out to the side:
        for t0_, t1_ in [(t_lo, SFT.lower_border_end[0] * SFT.T),
                        (SFT.upper_border_begin(N)[0] * SFT.T, t_hi)]:
            ax.axvspan(t0_, t1_, color='w', linewidth=0, alpha=.3)
        for t_ in [0, N * SFT.T]:  # mark signal borders with vertical line
            ax.axvline(t_, color='c', linestyle='--', alpha=0.5)
        
        fig.tight_layout()
        plt.show()
        
    return specgram
