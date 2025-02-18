import numpy as np
from nptdms import TdmsFile
from .utils import dec_and_trunc, TxRx

def read_tdms(fPath, num_channels, bio_channel=None, n_mimo=4,
              downsample_factor=10, truncate_start=0, truncate_end=1):
    # Read, truncate, downsample and save
    with TdmsFile.open(fPath) as tdms_file:
        # All TDMS groups are Untitled
        tmp = dec_and_trunc(tdms_file['Untitled'][TxRx(0, n_mimo)][:], truncate_start, truncate_end, downsample_factor)
        alen = len(tmp)
        bio = None
        
        # Bring all data elements to the same shape
        if bio_channel is not None:
            bio = tdms_file['Untitled']['BIOPAC_CH' + str(bio_channel)]
            alen = min(bio.shape, alen)
            bio = bio[truncate_start:alen-truncate_end] 
        
        mag = np.zeros((num_channels, alen))
        ph = np.zeros((num_channels, alen))

        for ch in range(num_channels):
            mag[ch] = dec_and_trunc(tdms_file['Untitled'][TxRx(ch, n_mimo)][:], truncate_start, truncate_end, downsample_factor)
            ph[ch] = dec_and_trunc(tdms_file['Untitled'][TxRx(ch, n_mimo) + '_Phase'][:], truncate_start, truncate_end, downsample_factor)

        # Properties can be read using the following command
        # tdms_file.properties

        return mag, ph, bio    
    
def read_mat(fPath, legacy=False):
    if legacy:
        # Compatibility for matfiles stored with version 7 instead of the recent 7.3
        from scipy.io import loadmat
        matfile = loadmat(fPath)
    else:
        # By default, we work with v7.3
        from mat73 import loadmat
        matfile = loadmat(fPath)
  
    return matfile