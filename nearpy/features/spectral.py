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

def get_mfcc_feats(
    audio, 
    sample_rate: int = 22050, 
    n_mfcc: int = 13, 
    n_fft: int = 2048, 
    hop_length: int = 512, 
    fmin: float = 0, 
    fmax: float = None,
    diffs: bool = True,
    normalize: bool = True
):
    '''
    Helper function to extract MFCC features for an input utterance
    
    Inputs:
    -----------
    audio_data : numpy.ndarray -> Input utterance
    sample_rate : int -> Sampling rate of the audio
    n_mfcc : int -> Number of MFCC coefficients per frame
    n_fft : int -> Number of samples per FFT window
    hop_length : int -> Number of samples between successive frames
    fmin : int -> Minimum frequency for mel filterbank
    fmax : int or None -> Maximum frequency for mel filterbank
        
    Output:
    --------
    mfccs : numpy.ndarray -> MFCC features with shape (n_mfcc, n_frames)
    '''
    audio = np.array(audio, dtype=float)
    
    mfccs = librosa.feature.mfcc(
        y=audio, 
        sr=sample_rate, 
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax
    )
    
    # Normalize MFCCs to improve performance 
    mfcc_feats = librosa.util.normalize(mfccs, axis=1)
    
    if diffs: 
        delta_mfccs = librosa.feature.delta(mfcc_feats)
        delta2_mfccs = librosa.feature.delta(mfcc_feats, order=2)
        mfcc_feats = np.vstack([mfcc_feats, delta_mfccs, delta2_mfccs])
        
    # Normalize (Cepstral Mean Variance Normalization - CMVN)
    if normalize: 
        scaler = StandardScaler()
        mfcc_feats = scaler.fit_transform(mfcc_feats)

    return mfcc_feats