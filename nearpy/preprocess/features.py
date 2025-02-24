import numpy as np 
import pywt
import torch 
import torch.nn as nn 

import tsfresh.feature_extraction.feature_calculators as fc

from ..ai import TimeSeriesAutoencoder, GestureTimeDataset, get_dataloaders, train_and_evaluate
    
''' Given an input dataframe with specified column(s) for data, generate feature vectors for each column and concat
'''
def generate_feature_vector(dataframe, method, cols=None): 
    methods = { 
        'ae': _get_AE_feats, 
        'ts': _get_time_series_feats, 
        'dwt': _get_dwt_feats
    }
    assert method in methods.keys(), f'Feature extractor must be one of {methods.keys()}. Got {method} instead'
    
    method_fcn = methods['method']
    
    if method == 'ae': 
        return method_fcn(dataframe[cols])

    feature_df = dataframe[cols].apply(method_fcn, axis=0)

    return feature_df.apply(lambda x: x.tolist(), axis=1)

def get_cwt_feats(df, f_low=0.1, f_high=15, fs=100, num_levels=30, wavelet='cgaul'): 
    scales = np.linspace(f_low, f_high, num_levels)
    widths = np.round(fs/scales)
    
    # Assuming that each column is a channel
    get_feats = lambda x: np.abs(np.transpose(pywt.cwt(x, widths, wavelet=wavelet)))
    cwt_extractor = lambda x: np.reshape(get_feats(x), (-1))
    
    return df.apply(cwt_extractor)

def _get_AE_feats(df): 
    torch.set_float32_matmul_precision('medium')
    data = GestureTimeDataset(df)
    train_loader, val_loader = get_dataloaders(data)

    # Define model - all segments are treated independently
    input_size = data[0][0].shape[0] # Len x 1
    val_loss = [None] * 9
    enc_sizes = [None] * 9
    models = [None] * 9 
    
    for i in range(1, 10):
        enc_size = 5*i
        enc_sizes[i-1] = enc_size
        models[i-1] = TimeSeriesAutoencoder(input_size=input_size, encoding_size=enc_size)

        # Train using AE
        _, _, val_loss[i-1] = train_and_evaluate(models[i-1], train_loader, val_loader, max_epochs=50, loss=nn.functional.mse_loss, task='AE', name=f'AE_Feat_{str(enc_size)}')
    
    # Pseudo AIC 
    pseudo_aic = -2 * np.log(val_loss) + 2*enc_size
    selected_model = models[np.argmin(pseudo_aic)]
    
    feat_extractor = lambda x: selected_model(torch.Tensor(x)).encoded
    
    # Generate features 
    return df.apply(feat_extractor)

def _get_dwt_feats(sig, fs=100, wavelet='morlet'):    
    pass 

def _get_time_series_feats(sig):
    '''
    Given a time series signal of shape (NxD), return an array of interpretable features (NxM)
    '''
    mob, comp = _get_hjorth_params(sig)
    zc = fc.number_crossing_m(sig, 0) # Get zero-crossing
    svd_ent = fc.fourier_entropy(sig, bins=10)
    skew = fc.skewness(sig)
    enrg = fc.abs_energy(sig)
    cid = fc.cid_ce(sig, True)
    kurt = fc.kurtosis(sig)
    med = fc.median(sig)
    pks = fc.number_peaks(sig, 10)
    cwt_pks = fc.number_cwt_peaks(sig, 10)
    var = fc.variance(sig)

    return np.vstack((mob, comp, zc, svd_ent, skew, enrg, cid, kurt, med, pks, cwt_pks, var))

def _get_hjorth_params(sig): 
    ''' Returns mobility and complexity, calculated using the same method as antropy. A re-implementation is performed to remove dependency on antropy, which relies upon numba (that does not work with NumPy 2). 

    mobility: sqrt(var(dy/dt)/var(y))
    complexity: mobility(dy/dt)/mobility(y)
    '''
    epsilon = 1e-10 # Added for numerical stability 

    mobility = lambda x: np.sqrt((np.var(np.diff(x)) + epsilon)/(np.var(x) + epsilon))
    complexity = lambda x: (mobility(np.diff(x)) + epsilon)/(mobility(x) + epsilon)

    return mobility(sig), complexity(sig)