from nptdms import TdmsFile
from pathlib import Path
from typing import List
import numpy as np 
from nearpy.utils import (
    dec_and_trunc, 
    get_channels_from_df, 
    split_channels_by_type
)

from .console import log_print 

def read_tdms_v2(
    f_path: Path, 
    rf_ds_ratio: int = 1, 
    truncate: List = None,
    rf_sr: int = None, 
    bio_sr: int = None,  
    get_bio: bool = False,
    exclude_rf_channels: List = None, 
    logger = None
):
    ''' [Future Projects Should Use This]
    Read TDMS files, parse them into appropriate channel types and return as 
    dictionary allowing for easy working with Dataframes. 
    
    Input Arguments: 
        f_path: Path -> File location
        
        rf_ds_ratio: int -> Amount by which raw RF data must be downsampled (default = 1)
        truncate: List -> Time (in seconds) to truncate from the start and end of data. 
            Note: If truncation is specified, sample rates must be provided. 
        rf_sr: int -> Sample rate for RF data (only needed for truncation)
        bio_sr: int -> Sample rate for BIOPAC data (only needed for truncation)
        
        get_bio: bool -> Flag for whether BIOPAC data should be returned or not
        exclude_rf_channels: List -> Specifies RF channels to be excluded
        
        logger: None or logging.Logger -> Logging messages 
    '''
    if truncate is not None: 
        assert isinstance(truncate, (list, tuple)), f"'truncate' must be a list of times (in seconds), got {type(truncate)} instead"

        assert rf_sr is not None, "Since RF data is being truncated, 'rf_sr' must not be None"
        if get_bio: 
            assert bio_sr is not None, "Since BIOPAC data is being truncated, 'bio_sr' must not be None"

    with TdmsFile.open(f_path) as tdm:
        # Get TDMS group
        tdmg = tdm['Untitled']
        # List available channels 
        tdm_channels = get_channels_from_df(tdmg.channels())
        log_print(logger, 'debug', f'Available Channels: {tdm_channels}')
        
        if len(tdm_channels) == 0:
            raise ValueError('TDMS file has no available channels')
        
        # Get all channels 
        bio_channels, rf_channels = split_channels_by_type(
            channel_list = tdm_channels, 
            excluded_channels = exclude_rf_channels, 
            include_biopac = get_bio
        )
        log_print(logger, 'info', f'Selected Channels\n BIOPAC:{bio_channels}\n RF:{rf_channels}')

        # Downsample and truncate RF data
        rf, bio = {}, {} 
        for ch in rf_channels: 
            rf[ch] = dec_and_trunc(tdmg[ch][:], truncate[0] * rf_sr, max(truncate[1] * rf_sr, 1), rf_ds_ratio)
        
        # Process properties 
        props = {
            'Timestamp': tdmg[ch].properties['NI_ExpTimeStamp']
        }

        # Align RF and BIOPAC length
        if get_bio: 
            bio_start = int(truncate[0] * bio_sr)
            bio_end = bio_start + int(bio_sr * len(rf[ch])/rf_sr) + 1 
            for ch in bio_channels: 
                bio[ch] = tdmg[ch][bio_start:bio_end]
        
    return rf, bio, props


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
            
        # Compute available channels  
        bio_channels, rf_channels = split_channels_by_type(tdm_channels, exclude, get_bio)
        log_print(logger, 'info', f'Selected Channels\n BIOPAC:{bio_channels}\n RF:{rf_channels}')
        
        if get_bio: 
            tmp = tdmg[bio_channels[0]]
            alen = min(len(tmp), alen)
        
        # Load data, ensuring all data elements have the same shape
        rf, bio = {}, {} 
        for ch in rf_channels: 
            rf[ch] = dec_and_trunc(tdmg[ch][:], truncate[0], truncate[1], ds_ratio)
        for ch in bio_channels: 
            bio[ch] = tdmg[ch][truncate[0]:-truncate[1]]
                        
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