import os 
import pandas as pd
from pathlib import Path 
from nptdms import TdmsFile
from typing import Dict, List

from scipy.signal import decimate

from .logs import log_print
from .mimo import TxRx, get_channels_from_df, split_channels_by_type
    
# Loads a TDMS file into a dictionary 
def read_tdms(f_path, 
              ds_ratio=10, 
              truncate=[0, 1], 
              get_bio=False, 
              exclude=None, 
              logger=None
):
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
        tdm_channels = get_channels_from_df(tdmg.channels())
        log_print(logger, 'debug', f'Available Channels: {tdm_channels}')
        
        if len(tdm_channels) == 0:
            raise ValueError('TDMS file has no available channels')
        
        # Compute dimensions of input 
        tmp = dec_and_trunc(tdmg[tdm_channels[0]][:], truncate[0], truncate[1], ds_ratio)
        alen = len(tmp)
        if get_bio: 
            tmp = tdmg[bio_channels[0]]
            alen = min(tmp.shape, alen)
            
        # Compute available channels  
        bio_channels, rf_channels = split_channels_by_type(tdm_channels, exclude, get_bio)
        log_print(logger, 'info', f'Selected Channels\n BIOPAC:{bio_channels}\n RF:{rf_channels}')
        
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

def find_files(base_path: Path, params: Dict, extensions: List[str]) -> Dict:
    '''
    Given a bunch of key-value pairs, find files in the base_dir that match the given pattern and extensions. 
    The number of extensions are left unconstrained to be able to find any files  
    '''
    pattern = '' 
    for k, v in params.items(): 
        pattern += f'*{k}*{v}'
    
    file_paths = {} 

    for extension in extensions:  
        try:    
            file_path = next(base_path.glob(f'{pattern}.{extension}'))
            file_paths[extension] = file_path
        except StopIteration:
            continue 

    return file_paths