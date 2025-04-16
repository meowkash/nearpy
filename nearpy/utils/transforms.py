import numpy as np 

def resample_indices(indices: list,  source: list, target: list): 
    ''' 
    Given indices from source, provide equivalent indices in target. 
    This essentialy performs resampling using minimized distances 
    '''
    assert (source[0] == target[0]) and (source[-1] == target[-1]), 'Indexing axes must have the same start and end values'
    start_val = source[indices[0]]
    end_val = source[indices[-1]]
    
    tgt_start = np.argmin(np.abs(target-start_val))
    tgt_end = np.argmin(np.abs(target-end_val))
    
    return list(range(tgt_start, tgt_end))