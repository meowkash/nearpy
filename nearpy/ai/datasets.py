import torch
from torch.utils.data import Dataset
from pywt import cwt 

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