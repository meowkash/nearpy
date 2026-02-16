from nptdms import TdmsFile
from pathlib import Path

from nearpy.utils import dec_and_trunc, get_channels_from_df, split_channels_by_type

from .console import log_print 

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