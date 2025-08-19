import numpy as np 
import tsfresh.feature_extraction.feature_calculators as fc

def get_feat_fn(feat):
    callbacks = {
        'Mobility': get_mobility,
        'Complexity': get_complexity,
        'Zero-Cross': lambda  x: fc.number_crossing_m(x, 0),
        'Fourier Entropy': lambda x: fc.fourier_entropy(x, bins=10),
        'Skewness': fc.skewness,
        'Energy': fc.abs_energy,
        'CID-CE': lambda x: fc.cid_ce(x, True),
        'Kurtosis': fc.kurtosis,
        'Median': fc.median,
        'N-Peaks': lambda x: fc.number_peaks(x, 10),
        'CWT-Peaks': lambda x: fc.number_cwt_peaks(x, 10),
        'Activity': fc.variance
    }

    return callbacks[feat]

def get_mobility(sig):
    epsilon = 1e-10  # Added for numerical stability
    mobility = lambda x: np.sqrt((np.var(np.diff(x)) + epsilon) / (np.var(x) + epsilon))

    return mobility(sig)

def get_complexity(sig):
    epsilon = 1e-10
    complexity = lambda x: (get_mobility(np.diff(x)) + epsilon) / (get_mobility(x) + epsilon)

    return complexity(sig)

def get_temporal_feats(sig, exclude=[]):
    '''
    Given a time series signal of shape (NxD), return an array of interpretable features (NxM)
    '''
    FEATS = ['Mobility', 'Complexity', 'Zero-Cross', 
            'Fourier Entropy', 'Skewness', 'Energy', 
            'Complexity', 'Kurtosis', 'Median', 
            'N-Peaks', 'CWT-Peaks', 'Activity'
    ]
    result_dict = {}

    for feat in FEATS:
        if feat in exclude:
            continue

        fn = get_feat_fn(feat)
        result_dict[feat] = fn(sig)

    return result_dict

def get_hjorth_params(sig): 
    ''' Returns mobility and complexity, calculated using the same method as antropy. A re-implementation is performed to remove dependency on antropy, which relies upon numba (that does not work with NumPy 2).
    mobility: sqrt(var(dy/dt)/var(y))
    complexity: mobility(dy/dt)/mobility(y)
    '''

    return get_mobility(sig), get_complexity(sig)