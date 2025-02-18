from nptdms import TdmsFile
import os 
import pandas as pd     
import math 

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
