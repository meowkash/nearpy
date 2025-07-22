import math 

def TxRx(x, n_mimo=4):
    return 'Tx' + str((x)%n_mimo +1) + 'Rx' + str(math.ceil((x+1)/n_mimo))

def get_mimo_channels(n_mimo, use_phase=False):
    # Magnitude channels
    channels = [TxRx(ch, n_mimo) for ch in range(n_mimo**2)]
    if use_phase: 
        phase_channels = [f'{TxRx(ch, n_mimo)}_Phase' for ch in range(n_mimo**2)] 
        channels.extend(phase_channels)
    
    return channels

def get_channels_from_df(tdms_channels): 
    clear_name = lambda x: x.split('/')[-1].strip("'>")
    return [clear_name(str(ch)) for ch in list(tdms_channels)]
    
def split_channels_by_type(channel_list, excluded_channels=None, include_biopac=False): 
    bio_channels = [x for x in channel_list if x.startswith('BIOPAC')]
    rf_channels = list(set(channel_list) - set(bio_channels))
     
    remove_excluded = lambda x: list(set(x) - set(excluded_channels))
    if excluded_channels is not None: 
        bio_channels = remove_excluded(bio_channels)
        rf_channels = remove_excluded(rf_channels)
    
    if not include_biopac: 
        bio_channels = []
    
    return bio_channels, rf_channels