import pywt
import numpy as np 
import librosa 

# TODO: Rewrite to remove dataframe dependency 
def get_cwt_feats(df, f_low=0.1, f_high=15, fs=100, num_levels=30, wavelet='cgaul'): 
    scales = np.linspace(f_low, f_high, num_levels)
    widths = np.round(fs/scales)
    
    # Assuming that each column is a channel
    get_feats = lambda x: np.abs(np.transpose(pywt.cwt(x, widths, wavelet=wavelet)))
    cwt_extractor = lambda x: np.reshape(get_feats(x), (-1))
    
    return df.apply(cwt_extractor)

def get_mfcc_feats(audio_data, sample_rate=22050, n_mfcc=13, 
                n_fft=2048, hop_length=512, fmin=0, fmax=None):
    '''
    Extract MFCC features from an audio waveform.
    
    Parameters:
    -----------
    audio_data : numpy.ndarray
        The audio waveform (already silence-removed)
    sample_rate : int
        Sampling rate of the audio
    n_mfcc : int
        Number of MFCC coefficients to extract
    n_fft : int
        Length of the FFT window
    hop_length : int
        Number of samples between successive frames
    fmin : int
        Minimum frequency for mel filterbank
    fmax : int or None
        Maximum frequency for mel filterbank (None uses sample_rate/2)
        
    Returns:
    --------
    mfccs : numpy.ndarray
        MFCC features with shape (n_mfcc, n_frames)
    '''
    
    mfccs = librosa.feature.mfcc(
        y=audio_data, 
        sr=sample_rate, 
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax
    )
    
    # Normalize MFCCs to improve performance 
    mfccs_normalized = librosa.util.normalize(mfccs, axis=1)
    
    return mfccs_normalized

def get_mfcc_delta_feats(audio_data, sample_rate=22050, n_mfcc=13, 
                                n_fft=2048, hop_length=512):
    """
    Extract MFCC features with delta and delta-delta coefficients.
    
    Returns:
    --------
    feature_vector : numpy.ndarray
        Combined MFCC features with derivatives
    """
    # Get base MFCCs
    mfccs = extract_mfcc(audio_data, sample_rate, n_mfcc, n_fft, hop_length)
    
    # Calculate delta features (first-order derivatives)
    delta_mfccs = librosa.feature.delta(mfccs)
    
    # Calculate delta-delta features (second-order derivatives)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    # Stack all features
    feature_vector = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
    
    return feature_vector