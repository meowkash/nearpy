#%%
from pathlib import Path

data_path = Path('G:/Pig DL Model Data/Processed DL Data')

# There will be different years in the data path - Need to keep the structure for book-keeping purposes
years = [x for x in data_path.iterdir() if x.is_dir()]

#%% 
import numpy as np 
import pandas as pd 
from nearpy import read_mat, TxRx

# Ensures we only select desired files
INTERVENTION_MAPPING = {
    'Phenylephrine': ['phenylephrine'],
    'Dobutamine': ['dobutamine'],
    'Atropine': ['atropine'],
    'Dexmed': ['dexmed'],
    'Xylazine': ['xylazine'],
    'Dopamine': ['dopamine'],
    'Embolism': ['pulmonary_embolism', 'pulmonaryembolism'],
    'Caval Occlusion': ['vena_cava_occlusion', 'venacavaocclusion', 'cavalocclusion'], 
    'Hypotension': ['hypotension'],
    'Hypoxia': ['hypoxia'],
    'Hypertension': ['hypertension', 'pulmonaryhypertension']
}

# Mapping for all keys in stored MAT files 
CAT_DATA_MAPPING = {
    'LVp': 'allLVPresWaveforms', 
    'LVv': 'allLVVolWaveforms', 
    'RVp': 'allRVPresWaveforms', 
    'RVv': 'allRVVolWaveforms', 
    'ECG': 'allECGWaveforms'
}

def get_intervention_name(inter_name): 
    for key, vals in INTERVENTION_MAPPING.items(): 
        for val in vals: 
            if val in inter_name: 
                return key
            
    return ''

def get_catheter_data(matdata): 
    cath_dict = {} 
    for key, val in CAT_DATA_MAPPING.items(): 
        try:
            # Ensure data shape is (n_segs, seg_len)
            cath_dict[key] = np.transpose(matdata[val].item())
        except ValueError:
            # Said key does not exist for this run, no worries
            continue

    return cath_dict

def get_nfrf_data(matdata): 
    nfrf_dict = {} 
    # For good measure, squeeze
    for ch_idx, ch in enumerate(matdata.squeeze()): 
        # Ensure data shape is (n_segs, seg_len)
        nfrf_dict[TxRx(ch_idx, 4)] = np.transpose(ch)

    return nfrf_dict

def make_dataframe_from_dict(data_dict):
    # Filter out empty arrays
    non_empty_dict = {k: v for k, v in data_dict.items() if v.shape[0] > 0}
    
    # If all arrays are empty, return an empty DataFrame
    if not non_empty_dict:
        return pd.DataFrame()
    
    # Determine max number of rows from non-empty arrays
    max_rows = max(array.shape[0] for array in non_empty_dict.values())
    
    # Initialize the DataFrame with only non-empty columns
    df = pd.DataFrame(index=range(max_rows), columns=non_empty_dict.keys())
    
    # Fill the DataFrame
    for key, array in non_empty_dict.items():
        for i in range(min(max_rows, array.shape[0])):
            df.at[i, key] = array[i, :]
    
    return df

for year in years: 
    if year.name == '2023':
        continue 
        # For some reason 2023 files do not have BPFiltWaveforms
          
    # Make dataframe per year and then append it into one if needed
    data_files = list(year.glob('*.mat'))
    year_df = pd.DataFrame()
    
    for idx, dfile in enumerate(data_files):
        # Get file path
        f_path = year/dfile.name
        # Extract metadata
        file_date, file_name = dfile.name.split(' ')
        inter_name = get_intervention_name(file_name)
        # Only valid interventions should be allowed
        if inter_name == '': 
            continue
        print(f'Date: {file_date}, Intervention: {inter_name}')
        
        # For each file, read all segments and add them to the dataframe 
        matdata = read_mat(f_path, legacy=True)
        
        cath_data_dict = get_catheter_data(matdata['bio'])
        file_df = make_dataframe_from_dict(cath_data_dict)
        
        try:
            nfrf_data_dict = get_nfrf_data(matdata['ncs']['allBPWaveforms'].item())
            nfrf_df = make_dataframe_from_dict(nfrf_data_dict)
            file_df = pd.concat([file_df, nfrf_df], axis=1)
        except:
            print(f'{dfile} has no NFRF data')
        
        # Annotate overall dataframe with metadata        
        file_df['Intervention'] = inter_name
        file_df['Date'] = file_date
        
        print(f'File dataframe summary: {file_df.info()}')
        
        # Concat to year dataframe
        year_df = pd.concat([year_df, file_df], axis=0)
        print(f'Overall dataframe summary: {year_df.info()}')
    
    # Save year dataframe
    year_df.to_hdf(f'{year.name}.h5', 'table')