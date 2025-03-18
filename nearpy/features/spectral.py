import pywt
import numpy as np 
from librosa.feature import mfcc 

def get_cwt_feats(df, f_low=0.1, f_high=15, fs=100, num_levels=30, wavelet='cgaul'): 
    scales = np.linspace(f_low, f_high, num_levels)
    widths = np.round(fs/scales)
    
    # Assuming that each column is a channel
    get_feats = lambda x: np.abs(np.transpose(pywt.cwt(x, widths, wavelet=wavelet)))
    cwt_extractor = lambda x: np.reshape(get_feats(x), (-1))
    
    return df.apply(cwt_extractor)

def get_mel_feats(df, fs, n_feats=10):
    # Assuming that each column is a channel of data 
    mel_extractor = lambda x: mfcc(y=x, sr=fs, n_mfcc=n_feats)
    
    return df.apply(mel_extractor)
