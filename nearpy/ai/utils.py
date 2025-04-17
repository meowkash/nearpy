import numpy as np 
import contextlib
import logging 
from pathlib import Path

import numpy as np 
import pandas as pd

from ..utils import read_tdms
from ..utils.logs import log_print
from ..preprocess import get_gesture_filter, filter_and_normalize

from .features import generate_feature_df

def make_dataset(data_path: Path, 
                 gestures: dict, 
                 num_reps: int, 
                 seg_time: float,
                 fs: int, 
                 num_channels: int, 
                 ds_ratio: int = 10, 
                 f_s=15, 
                 visualize: bool = False, 
                 refresh: bool = False, 
                 logger: logging.Logger = None):
    
    """Loads TDMS files, processes signals, and saves datasets."""
    data_path = Path(data_path)
    dataset_file = data_path / 'dataset.pkl'
    filtered_dataset_file = data_path / 'filtered_dataset.pkl'
    
    if refresh:
        with contextlib.suppress(FileNotFoundError):
            dataset_file.unlink()
            filtered_dataset_file.unlink()
            log_print(logger, 'debug', 'Old dataset files removed for refresh.')
    
    if dataset_file.exists() and filtered_dataset_file.exists():
        log_print(logger, 'debug', 'Loading existing datasets.')
        return pd.read_pickle(dataset_file), pd.read_pickle(filtered_dataset_file)
    
    log_print(logger, 'debug', 'Creating new datasets.')
    num_gestures = len(gestures)
    ges_sub = num_gestures * num_reps
    stime = int(seg_time * fs)
    files = list(data_path.glob('*/*.tdms'))
    num_files = len(files)
    num_segments = num_files * ges_sub
    
    # Preallocate arrays
    shape = (num_segments, num_channels * stime)
    mag_data, filt_mag = np.zeros(shape), np.zeros(shape)
    phase, filt_phase = np.zeros(shape), np.zeros(shape)
    sub_label, rout_label, ges_label = np.zeros((num_segments, 1), dtype=int), np.zeros((num_segments, 1), dtype=int), np.zeros((num_segments, 1), dtype=int)
    
    filt_num = get_gesture_filter(f_s, fs, visualize)
    
    for f_idx, file in enumerate(files):
        base_idx = f_idx * ges_sub
        sub = int(file.parent.name.split()[-1])
        rt = int(file.stem.split('_')[0][-1])
        
        tdmsData = read_tdms(file, num_channels, downsample_factor=ds_ratio)
        mag, ph = tdmsData[0], tdmsData[1]
        
        sub_label[base_idx:base_idx + ges_sub] = sub
        rout_label[base_idx:base_idx + ges_sub] = rt
        
        start_idx = mag.shape[1] - ges_sub * stime
        # This is necessary to prevent issues arising from initial jumps
        fm = filter_and_normalize(mag[:, start_idx:], (filt_num, 1), axis=1)
        fp = filter_and_normalize(ph[:, start_idx:], (filt_num, 1), axis=1)
        mag = mag[:, start_idx: ]
        ph = ph[:, start_idx: ]
        
        start_idx = 0
        
        for i in range(ges_sub):
            end_idx = start_idx + stime
            mag_data[base_idx + i] = mag[:, start_idx:end_idx].ravel()
            phase[base_idx + i] = ph[:, start_idx:end_idx].ravel()
            filt_mag[base_idx + i] = fm[:, start_idx:end_idx].ravel()
            filt_phase[base_idx + i] = fp[:, start_idx:end_idx].ravel()
            ges_label[base_idx + i] = i // num_reps
            start_idx = end_idx
        
        # Save raw files 
        save_path = file.parent / f'Routine {rt}.npz'
        np.savez(save_path, mag=mag, phase=ph, filt_mag=fm, filt_phase=fp)    
    
    dataset = pd.DataFrame({
        'subject': sub_label.ravel(), 
        'routine': rout_label.ravel(), 
        'gesture': ges_label.ravel(),
        'mag': mag_data.tolist(), 
        'phase': phase.tolist()
    })
    filt_dataset = pd.DataFrame({
        'subject': sub_label.ravel(), 
        'routine': rout_label.ravel(), 
        'gesture': ges_label.ravel(),
        'mag': filt_mag.tolist(), 
        'phase': filt_phase.tolist()
    })
    
    dataset.to_pickle(dataset_file)
    filt_dataset.to_pickle(filtered_dataset_file)
    log_print(logger, 'debug', 'Datasets saved successfully.')
    
    return dataset, filt_dataset

def make_feature_dataset(dataframe: pd.DataFrame, 
                         method, 
                         base_path, 
                         num_vars=16, 
                         data_key='mag', 
                         subject_key='subject', 
                         routine_key='routine',
                         label_key='gesture', 
                         refresh=False, 
                         logger=None): 
    dataset_file = base_path / f'{method}_feat_dataset.pkl'
    
    if refresh:
        with contextlib.suppress(FileNotFoundError):
            dataset_file.unlink()
            log_print(logger, 'debug', 'Old dataset files removed for refresh.')

    if dataset_file.exists():
        log_print(logger, 'debug', f'Loading existing {method} feature dataset.')
        feat_dataset = pd.read_pickle(open(dataset_file, 'rb'))
    else:
        feat_dataset = generate_feature_df(dataframe, method=method, num_vars=num_vars,
                                           data_key=data_key, label_key=label_key, 
                                           subject_key=subject_key, routine_key=routine_key,
                                           refresh=refresh)
        feat_dataset.to_pickle(dataset_file)
    
    return feat_dataset

def load_dataset(base_path, gestures, num_channels=16, 
                   num_reps=5, f_ncs=10000, ds_ratio=100,
                   visualize=True, refresh=False, logger=None):
    """Loads or creates datasets with improved structure."""
    base_path = Path(base_path)
    data_path, long_data_path = base_path / 'Data', base_path / 'Longitudinal Data'
    
    fs = f_ncs / ds_ratio
    rep_time = 2.997
    
    df, filt_df = make_dataset(data_path, gestures, num_reps, 
                               rep_time, fs, num_channels, ds_ratio, 
                               visualize=visualize, refresh=refresh)
    log_print(logger, 'debug', 'Loaded dataset')
        
    long_df, long_filt_df = make_dataset(long_data_path, gestures, num_reps, 
                                         rep_time, fs, num_channels, ds_ratio, visualize=visualize, refresh=refresh)
    log_print(logger, 'debug', 'Loaded longitudinal dataset')
    
    # Make feature datasets 
    log_print(logger, 'debug', 'Loading feature datasets')
    ts_feat_df = make_feature_dataset(filt_df, method='ts', base_path=data_path, 
                                      num_vars=num_channels, refresh=refresh)
    ae_feat_df = make_feature_dataset(filt_df, method='ae', base_path=data_path, 
                                      num_vars=num_channels, refresh=refresh)
    # Longitudinal
    ts_long_feat_df = make_feature_dataset(long_filt_df, method='ts', base_path=long_data_path,
                                           num_vars=num_channels, refresh=refresh)
    ae_long_feat_df = make_feature_dataset(long_filt_df, method='ae', base_path=long_data_path,
                                           num_vars=num_channels, refresh=refresh)
    
    num_subjects = len(set(df['subject']))
    
    log_print(logger, 'info', f'Dataset contains {num_subjects} subjects. Longitudinal Subjects: {set(long_df["subject"])}')
    
    return df, filt_df, long_df, long_filt_df, ts_feat_df, ae_feat_df, ts_long_feat_df, ae_long_feat_df

# tslearn dataset spec  is (num_cases, time_steps, num_channels)
def adapt_dataset_to_tslearn(data: pd.DataFrame, 
                             num_vars: int = 16,
                             subject_num: int = None,
                             class_key: str = 'gesture', 
                             data_key: str = 'mag', 
                             routine_key: str = 'routine',
                             subject_key: str = 'subject', 
                            ):
    subset_map = {
        subject_key: subject_num
    }
    dft = get_dataframe_subset(data, subset_map)
    
    data = np.array([np.transpose(np.reshape(dft.iloc[i][data_key], (num_vars, -1))) for i in range(len(dft))])
    label = dft[class_key].to_numpy()
    routine = dft[routine_key].to_numpy()
    
    return data, label, routine 

def get_dataframe_subset(df, map_dict=None):
    if map_dict is None: 
        return df  
    
    subset_df = df
    for k, v in map_dict.items(): 
        subset_df = subset_df.loc[subset_df[k] == v]

    return subset_df