import numpy as np 
import tsfresh.feature_extraction.feature_calculators as fc

def get_temporal_feats(sig, keys=None):
    '''
    Given a time series signal of shape (NxD), return an array of interpretable features (NxM)
    '''
    mob, comp = get_hjorth_params(sig)
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

    result_dict = {
        'Mobility': mob, 
        'Complexity': comp,
        'Zero-Cross': zc,
        'Fourier Entropy': svd_ent, 
        'Skewness': skew,
        'Energy': enrg, 
        'Complexity': cid,
        'Kurtosis': kurt, 
        'Median': med,
        'N-Peaks': pks,
        'CWT-Peaks': cwt_pks, 
        'Activity': var
    }
    
    if keys is not None: 
        return result_dict[keys]
    else: 
        return result_dict

def get_hjorth_params(sig): 
    ''' Returns mobility and complexity, calculated using the same method as antropy. A re-implementation is performed to remove dependency on antropy, which relies upon numba (that does not work with NumPy 2). 

    mobility: sqrt(var(dy/dt)/var(y))
    complexity: mobility(dy/dt)/mobility(y)
    '''
    epsilon = 1e-10 # Added for numerical stability 

    mobility = lambda x: np.sqrt((np.var(np.diff(x)) + epsilon)/(np.var(x) + epsilon))
    complexity = lambda x: (mobility(np.diff(x)) + epsilon)/(mobility(x) + epsilon)

    return mobility(sig), complexity(sig)