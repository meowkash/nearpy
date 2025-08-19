from scipy.signal import decimate

def dec_and_trunc(inp, truncate_start, truncate_end, downsample_factor):
    decInp = decimate(inp, downsample_factor)
    return decInp[truncate_start:-truncate_end]