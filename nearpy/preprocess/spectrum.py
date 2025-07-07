import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming
import matplotlib.pyplot as plt 
from scipy.fft import fft
from librosa.feature import melspectrogram
from librosa.display import specshow

def _ncs_fft(sig, fs):
    Y = fft(sig)
    L = len(sig)
    L = L - L%2 # Make it even  
    P2 = abs(Y/L) 
    P = P2[0:L//2]
    P[1:-2] = 2*P[1:-2]
    f = fs*np.arange(0, L/2)/L

    return f, P 

def ncs_fft(sig, fs, plot=False, range=None):
    f, P = _ncs_fft(sig, fs)
    
    if range is not None: 
        idx = (range[0] < f) & (f < range[1])
        f = f[idx]
        P = P[idx]
        
    if plot:
        import matplotlib.pyplot as plt
        plt.ion()
        plt.plot(f, P)
        plt.show()
        
    return f, P

def get_peak_harmonic(sig, fs, range=None): 
    f, P = ncs_fft(sig, fs, range=range)
    idx = np.argmax(P)
    
    return f[idx]

def get_spectrogram(sig, fs, seg_frac=20, perc_overlap=0.5, visualize=False): 
    # We use the same defaults that matlab uses for consistency across code-bases 
    N = len(sig) 
    nperseg = int(np.floor(N/seg_frac)) # Divide signal into segments of fixed length
    noverlap = int(np.floor(perc_overlap*nperseg)) # Define overlap between contiguous segments 
    win = hamming(int(nperseg))
    nfft = max(256, nperseg) # Compute number of points to take FFT on 
    
    # Define STFT object to get STFT and Spectrogram
    SFT = ShortTimeFFT(win=win, 
                       hop=noverlap, 
                       fs=fs, 
                       fft_mode='onesided',
                       mfft=nfft, 
                       scale_to='magnitude', 
                       phase_shift=None)
    
    specgram = SFT.spectrogram(sig)

    if visualize: 
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot

        ax.set_title('Spectrogram')
        ax.set(xlabel='Time (s)', ylabel='Frequency (Hz)',
                xlim=(t_lo, t_hi))

        Sx_dB = 10 * np.log10(np.fmax(specgram, 1e-4))  # limit range to -40 dB
        im = ax.imshow(Sx_dB, origin='lower', aspect='auto',
                        extent=SFT.extent(N), cmap='jet')
        fig.colorbar(im, label='Power Spectral Density (dB)')

        # Shade areas where window slices stick out to the side:
        for t0_, t1_ in [(t_lo, SFT.lower_border_end[0] * SFT.T),
                        (SFT.upper_border_begin(N)[0] * SFT.T, t_hi)]:
            ax.axvspan(t0_, t1_, color='w', linewidth=0.1, alpha=.3)
        for t_ in [0, N * SFT.T]:  # mark signal borders with vertical line
            ax.axvline(t_, color='y', linestyle='--', alpha=0.5)

        fig.tight_layout()
        plt.show()
        
    return specgram

def get_mel_spectrogram(sig, fs, visualize=False): 
    # This is primarily useful for understanding and analyzing sound data using the Mel-Frequency scale
    specgram = melspectrogram(sig, sr=fs)
    Sx_dB = 10 * np.log10(np.fmax(specgram, 1e-4))

    if visualize: 
    # Plot spectrogram
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        img = specshow(Sx_dB, x_axis='time', y_axis='mel', ax = ax)
        fig.colorbar(img, ax = ax, format='%+2.0f dB')
        ax.set_title('Mel-Scaled Spectrogram')
        fig.show()