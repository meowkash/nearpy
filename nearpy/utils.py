import math 
import os 
import time 
import numpy as np 
import pandas as pd
from pathlib import Path 
from nptdms import TdmsFile
from scipy.signal import decimate

def fn_timer(func, *args, **kwargs): 
    st_time = time.time()
    func_result = func(*args, **kwargs)
    en_time = time.time()
    return func_result, en_time - st_time 
    
def get_accuracy(cm):
    if type(cm) is dict:
        num_classes = cm[list(cm.keys())[0]].shape[0]
        return _get_dict_accuracy(cm, num_classes)
    else:
        num_classes = cm.shape[0]
        return _get_accuracy(cm, num_classes)    

def _get_accuracy(cm, num_classes):
    return sum([cm[i, i] for i in range(num_classes)])/np.concatenate(cm).sum()

def _get_dict_accuracy(cm, num_classes):
    return np.average([_get_accuracy(cc, num_classes) for _, cc in cm.items()])

def TxRx(x, n_mimo=4):
    return 'Tx' + str((x)%n_mimo +1) + 'Rx' + str(math.ceil((x+1)/n_mimo))

def get_mimo_channels(n_mimo, use_phase=False):
    # Magnitude channels
    channels = [TxRx(ch, n_mimo) for ch in range(n_mimo**2)]
    if use_phase: 
        phase_channels = [f'{TxRx(ch, n_mimo)}_Phase' for ch in range(n_mimo**2)] 
        channels.extend(phase_channels)
    
    return channels

def _prettify_channel_names(tdms_channels): 
    clear_name = lambda x: x.split('/')[-1].strip("'>")
    return [clear_name(str(ch)) for ch in list(tdms_channels)]
    
def _separate_channel_types(channel_list, excluded_channels=None, include_biopac=False): 
    bio_channels = [x for x in channel_list if x.startswith('BIOPAC')]
    rf_channels = list(set(channel_list) - set(bio_channels))
     
    remove_excluded = lambda x: list(set(x) - set(excluded_channels))
    if excluded_channels is not None: 
        bio_channels = remove_excluded(bio_channels)
        rf_channels = remove_excluded(rf_channels)
    
    if not include_biopac: 
        bio_channels = []
    
    return bio_channels, rf_channels
    
# DEPRECATION WARNING
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

# New version: Plan to move to this. 
def read_tdms_v2(f_path, ds_ratio=10, truncate=[0, 1], get_bio=False, 
                 exclude=None, logger=None, *args, **kwargs):
    '''
    This function loads TDMS files and loads variables into dictionaries which may be easily converted into Dataframes. By default, all channels present in the TDMS file are loaded. 
    
    Input Arguments: 
        f_path: str or pathlib.Path object representing file location
        get_bio: bool, specifies if BIOPAC channels are to be returned or not
        ds_ratio: int, specifies amount by which raw file must be downsampled
        truncate: [int, int], specifies time to truncate from the start and end of recording
        exclude: [strs], specifies channels to be excluded
        logger: None or logging.Logger, for logging messages 
    '''
    
    f_path = Path(f_path)
    
    with TdmsFile.open(f_path) as tdm:
        # Get TDMS group
        tdmg = tdm['Untitled']
        # List available channels 
        tdm_channels = _prettify_channel_names(tdmg.channels())
        logprint(logger, 'debug', f'Available Channels: {tdm_channels}')
        
        if len(tdm_channels) == 0:
            raise ValueError('TDMS file has no available channels')
        
        # Compute dimensions of input 
        tmp = dec_and_trunc(tdmg[tdm_channels[0]][:], truncate[0], truncate[1], ds_ratio)
        alen = len(tmp)
        if get_bio: 
            tmp = tdmg[bio_channels[0]]
            alen = min(tmp.shape, alen)
            
        # Compute available channels  
        bio_channels, rf_channels = _separate_channel_types(tdm_channels, exclude, get_bio)
        logprint(logger, 'info', f'Selected Channels\n BIOPAC:{bio_channels}\n RF:{rf_channels}')
        
        # Load data, ensuring all data elements have the same shape
        rf, bio = {}, {} 
        for ch in rf_channels: 
            rf[ch] = dec_and_trunc(tdmg[ch][:], truncate[0], truncate[1], ds_ratio)
        for ch in bio_channels: 
            bio[ch] = tdmg[ch][truncate[0]:alen-truncate[1]]
                        
        # Properties can be read using the following command
        props = tdm.properties

    return rf, bio, props 
    
def read_mat(fPath, legacy=False):
    if legacy:
        # Compatibility for matfiles stored with version 7 instead of the recent 7.3
        from scipy.io import loadmat
        matfile = loadmat(fPath, squeeze_me=True, simplify_cells=True)
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
    decInp = decimate(inp, downsample_factor)
    return decInp[truncate_start:-truncate_end]

# Log if logger available, else print
def logprint(logger, level, message, *args, **kwargs):
    if logger is not None:
        log_method = getattr(logger, level, None)
        if log_method:
            log_method(message, *args, **kwargs)
    else:
        print(message)
            