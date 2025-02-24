import os
import math 
import numpy as np
import pandas as pd 

from nptdms import TdmsFile

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

def tdms_to_csv(fPath):
    sub = os.path.split(fPath)[-1]
    idx = sub.rfind('.tdms')
    save_name = sub[0:idx] + '.csv'
    sub = os.path.split(fPath)[0] # get folder path 
    save_name = os.path.join(sub, save_name)

    with TdmsFile.open(fPath) as tdms_file:
        data_dict = {}
    
        group = tdms_file['Untitled']
        
        for ch in range(16):
            if ch==0:
                data_dict[TxRx(ch)] = group['Untitled'][:]
            else:
                data_dict[TxRx(ch)] = group['Untitled ' + str(ch)][:]
        
        df = pd.DataFrame(data_dict)
        df.to_csv(save_name, encoding='utf-8')

def dec_and_trunc(inp, truncate_start, truncate_end, downsample_factor):
    from scipy.signal import decimate
    decInp = decimate(inp, downsample_factor)
    return decInp[truncate_start:-truncate_end]

def TxRx(x, n_mimo=4):
    return 'Tx' + str((x)%n_mimo +1) + 'Rx' + str(math.ceil((x+1)/n_mimo))
