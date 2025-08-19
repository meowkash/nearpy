import pandas as pd
from nptdms import TdmsFile
from pathlib import Path

from nearpy.utils import get_channels_from_df

from .console import log_print

def tdms_to_csv(file_path: Path, logger = None):
    save_path = file_path.parent / file_path.name.replace('.tdms', '.csv')

    with TdmsFile.open(file_path) as tdms_file:
        data_dict = {}
    
        group = tdms_file['Untitled']
        # List available channels 
        channels = get_channels_from_df(group.channels())

        log_print(logger, 'debug', f'Available Channels: {channels}')
        
        if len(channels) == 0:
            raise ValueError('TDMS file has no available channels')
        
        for ch in channels:
            data_dict[ch] = group[ch][:]
        
        df = pd.DataFrame(data_dict)
        df.to_csv(save_path, encoding='utf-8')

        log_print(logger, 'info', f'Successfully saved CSV file at {str(save_path)}')