from pathlib import Path
import contextlib
import pandas as pd
import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import minmax_scale as normalize
from scipy.signal import filtfilt
from pywt import cwt 

from .data import read_tdms
from .features import segment_gesture
from .preprocess import get_gesture_filter

class GestureTimeDataset(Dataset):
    def __init__(self, df, sub=None, num_channels=16, transform=None):
        # Generate dataset using provided dataframe and subject 
        self.sub = sub
        
        # Handle situation for subject independent models 
        if sub is None: 
            self.data = df
        else: 
            self.data = df.loc[df['subject'] == self.sub] 
        
        self.num_channels = num_channels
        self.length = len(self.data)
        self.transform = transform

    def __len__(self):
        return self.length*self.num_channels

    def __getitem__(self, idx):
        # Process dataframe to be loaded according to gesture/subject
        ch = idx//self.length
        sub = idx % self.length
        datum = torch.reshape(torch.Tensor(self.data.iloc[sub]['mag']), (self.num_channels, -1))
        elem = datum[ch, :]
        
        if self.transform is not None:
            elem = self.transform(elem)    
            # tsai transforms to use: TSVerticalFlip, TSRandomShift, TSHorizontalFlip, TSRandomTrenda 

        label = self.data.iloc[sub]['gesture']
                    
        return elem, label

class CWTDataset(Dataset): 
    def __init__(self, df, sub=1, num_channels=16, wavelet='morl'):
        # Generate dataset using provided dataframe and subject 
        self.sub = sub
        
        if sub is None: 
            self.data = df
        else: 
            self.data = df.loc[df['subject'] == self.sub] 
        
        self.num_channels = num_channels
        self.wavelet = wavelet

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, ch_first=False, key='mag'):
        # Process dataframe to be loaded according to gesture/subject
        datum = torch.reshape(self.data.iloc[idx][key], (self.num_channels, -1))
        elem = torch.zeros((datum.shape[1], datum.shape[1], self.num_channels))
        scales = range(1, datum.shape[1]+1)
        
        for ch in range(self.num_channels): 
            elem[:, :, ch] = cwt(datum[ch, :], scales, self.wavelet, 1)
            
        label = self.data.iloc[idx]['gesture']
        
        if ch_first:
            elem = torch.reshape(elem, (self.num_channels, -1))
        else:
            elem = torch.reshape(elem, (-1, self.num_channels))
            
        return elem, label
    
def get_dataloaders(dataset, split=0.3, train_batch=32, val_batch=32):
    val_size = round(split * len(dataset))
    train_size = len(dataset) - val_size
    
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=train_batch, shuffle=True, 
                              num_workers=16, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=val_batch, shuffle=False, 
                            num_workers=8, persistent_workers=True)
    return train_loader, val_loader

# tslearn dataset spec  is (num_cases, time_steps, num_channels)
def make_tslearn_dataset(dataset, num_channels=16, 
                         data_key='mag', label_key='gesture', 
                         subset_key='subject', subset_val=None):
    if subset_val is not None:
        dft = dataset.loc[dataset[subset_key] == subset_val] 
    else:
        dft = dataset
    
    data = np.array([np.transpose(np.reshape(dft.iloc[i][data_key], (num_channels, -1))) for i in range(len(dft))])
    label = dft[label_key].to_numpy()
    routine = dft['routine'].to_numpy()
    
    return data, label, routine 

# tslearn dataset spec  is (num_cases, num_channels, time_steps)
def make_tsai_dataset(dataset, num_channels=16, 
                      data_key='mag', label_key='gesture', 
                      subset_key='subject', subset_val=None):
    if subset_val is not None:
        dft = dataset.loc[dataset[subset_key] == subset_val] 
    else:
        dft = dataset
    
    data = np.array([np.reshape(dft.iloc[i][data_key], (num_channels, -1)) for i in range(len(dft))])
    label = dft[label_key].to_numpy()
    
    return data, label 
    
def make_action_segmented_dataset(dataset, num_channels=16, fs=1000):
    # Segment length is chosen to be same as fs
    action_data = np.zeros((len(dataset), num_channels*fs))
    
    for i in range(len(dataset)):
        dat = np.reshape(dataset.iloc[i]['mag'], (num_channels, -1))
        xseg = segment_gesture(dat, num_channels=num_channels, fs=fs)
        action_data[i, :] = np.reshape(xseg, -1)
    
    seg_dataset = pd.DataFrame({'subject': dataset['subject'].squeeze(),
                                'routine': dataset['routine'].squeeze(),
                                'gesture': dataset['gesture'].squeeze(), 
                                'mag': action_data.tolist()})
    
    return seg_dataset

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

        for i in range(ges_sub):
            start_idx = 0
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
