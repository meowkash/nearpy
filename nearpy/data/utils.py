import contextlib
from pathlib import Path

import numpy as np 
import pandas as pd
from scipy.signal import filtfilt
from torch.utils.data import DataLoader, random_split

from .read_files import read_tdms
from ..preprocess import get_gesture_filter

def get_dataloaders(dataset, split=0.3, train_batch=32, val_batch=32):
    val_size = round(split * len(dataset))
    train_size = len(dataset) - val_size
    
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=train_batch, shuffle=True, 
                              num_workers=16, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=val_batch, shuffle=False, 
                            num_workers=8, persistent_workers=True)
    return train_loader, val_loader

def make_dataset(data_path, gestures, num_reps, seg_time, fs, 
                   num_channels, ds_ratio, f_s=15, visualize=False, refresh=False):
    """Loads TDMS files, processes signals, and saves datasets."""
    data_path = Path(data_path)
    dataset_file = data_path / 'dataset.pkl'
    filtered_dataset_file = data_path / 'filtered_dataset.pkl'
    
    if refresh:
        with contextlib.suppress(FileNotFoundError):
            dataset_file.unlink()
            filtered_dataset_file.unlink()
            print('Old dataset files removed for refresh.')
    
    if dataset_file.exists() and filtered_dataset_file.exists():
        print('Loading existing datasets.')
        return pd.read_pickle(dataset_file), pd.read_pickle(filtered_dataset_file)
    
    print("Creating new datasets.")
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
        fm = np.array([filtfilt(filt_num, 1, mag[ch, start_idx:]) for ch in range(num_channels)])
        fp = np.array([filtfilt(filt_num, 1, ph[ch, start_idx:]) for ch in range(num_channels)])
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
    print("Datasets saved successfully.")
    
    return dataset, filt_dataset

def load_dataset(base_path, gestures, num_channels=16, 
                   num_reps=5, f_ncs=10000, ds_ratio=100,
                   visualize=True, refresh=False):
    """Loads or creates datasets with improved structure."""
    base_path = Path(base_path)
    data_path, long_data_path = base_path / 'Data', base_path / 'Longitudinal Data'
    
    fs = f_ncs / ds_ratio
    rep_time = 2.997
    
    df, filt_df = make_dataset(data_path, gestures, num_reps, 
                               rep_time, fs, num_channels, ds_ratio, 
                               visualize=visualize, refresh=refresh)
    print('Loaded dataset')
    long_df, long_filt_df = make_dataset(long_data_path, gestures, num_reps, 
                                         rep_time, fs, num_channels, ds_ratio, visualize=visualize, refresh=refresh)
    print('Loaded longitudinal dataset')
    
    num_subjects = len(set(df['subject']))
    print(f'Dataset contains {num_subjects} subjects. Longitudinal Subjects: {set(long_df["subject"])}')
    
    return df, filt_df, long_df, long_filt_df