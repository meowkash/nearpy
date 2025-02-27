import torch
from torch.utils.data import Dataset
from pywt import cwt
from torch.utils.data import DataLoader, random_split

class GestureTimeDataset(Dataset):
    def __init__(self, df, sub=None,
                 data_key='mag', label_key='gesture',
                 num_vars=16, transform=None):
        # Generate dataset using provided dataframe and subject
        self.sub = sub

        # Handle situation for subject independent models
        if sub is None:
            self.data = df[data_key]
            self.labels = df[label_key]
        else:
            self.data = df.loc[df['subject'] == self.sub][data_key]
            self.labels = df.loc[df['subject'] == self.sub][label_key]

        self.num_vars = num_vars
        self.length = len(self.data)
        self.transform = transform

    def __len__(self):
        return self.length * self.num_vars

    def __getitem__(self, idx):
        # Process dataframe to be loaded according to gesture/subject
        ch = idx % self.num_vars
        num = idx // self.num_vars

        datum = torch.reshape(torch.Tensor(self.data.iloc[num]), (self.num_vars, -1))
        elem = datum[ch, :].squeeze()  # Make 1-D

        if self.transform is not None:
            elem = self.transform(elem)
            # tsai transforms to use: TSVerticalFlip, TSRandomShift, TSHorizontalFlip, TSRandomTrends, TSWarp

        label = self.labels.iloc[num]

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
