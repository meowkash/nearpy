from pathlib import Path 

import torch 
import torch.nn as nn 
import numpy as np 
import pandas as pd
import librosa 

import tsfresh.feature_extraction.feature_calculators as fc

from .models import TimeAutoEncoder, AEWrapper
from .datasets import GestureTimeDataset, get_dataloaders
from .trainer import train_and_evaluate
from ..features import get_temporal_feats

''' Given an input dataframe with specified column(s) for data, generate feature vectors for each column and concat
'''
def generate_feature_df(dataframe, 
                        method, 
                        base_path="", 
                        num_vars=16, 
                        data_key='mag', 
                        subject_key='subject', 
                        routine_key='routine',
                        label_key='gesture', 
                        refresh=False
                    ):
    methods = { 
        'ae': _get_ae_feats,
        'ts': _get_time_series_feats, 
    }
    assert method in methods.keys(), f'Feature extractor must be one of {methods.keys()}. Got {method} instead'
    assert label_key is not None, f'A valid label key is needed'
    
    method_fcn = methods[method]
    
    if base_path == "": 
        base_path = Path.cwd() 
    
    # We need to ensure we reshape and then apply the method
    n_elems = len(dataframe)
    subset = np.reshape(list(dataframe[data_key]), (n_elems, num_vars, -1))
    
    if method == 'ae': 
        # Pre-train AE 
        if refresh:
            input_size, ae_size = _pretrain_ae(dataframe, 
                                               data_key=data_key, 
                                               label_key=label_key, 
                                               base_path=base_path, 
                                               num_vars=num_vars
                                            )
        else:
            input_size = 299
            ae_size = 8
        # Load corresponding AE model into wrapper
        wrapped_model = AEWrapper(input_size=input_size, encoding_size=ae_size)
        ckpt_path = base_path / 'ckpts' / f'AE_Feat_{str(ae_size)}.ckpt'
        wrapped_model.load_state_dict(torch.load(ckpt_path)['state_dict'])
        feat_subset = np.apply_along_axis(method_fcn, 2, subset, 
                                          model=wrapped_model)
    else:
        # Extract features
        feat_subset = np.apply_along_axis(method_fcn, 2, subset)
    
    feat_subset = feat_subset.reshape((n_elems, -1))
    
    print(routine_key, set(dataframe[routine_key]))
    
    # Create a complete dataset
    feat_df = pd.DataFrame({
        label_key: list(dataframe[label_key]),
        subject_key: list(dataframe[subject_key]),
        routine_key: list(dataframe[routine_key]),
        data_key: list(feat_subset)
    })
    
    return feat_df

def _pretrain_ae(df, data_key, label_key, base_path, 
                 num_vars=16, num_ae_steps=10, step_feats=4): 
    torch.set_float32_matmul_precision('medium')
    data = GestureTimeDataset(df, data_key=data_key, num_vars=num_vars, label_key=label_key)
    train_loader, val_loader = get_dataloaders(data)

    # Define model - all segments are treated independently
    input_size = data[0][0].shape[0] # len x 1

    val_loss = np.zeros((num_ae_steps, 1)).squeeze()
    enc_sizes = np.zeros((num_ae_steps, 1)).squeeze()
    
    for stp in range(num_ae_steps):
        enc_size = (stp + 1) * step_feats
       
        model = TimeAutoEncoder(input_size=input_size, encoding_size=enc_size)
        enc_sizes[stp] = enc_size

        # Train using AE
        _, _, val_loss[stp] = train_and_evaluate(model, train_loader, val_loader,
                                                 base_path=base_path, max_epochs=50, 
                                                 loss=nn.functional.mse_loss, task='AE', name=f'AE_Feat_{str(enc_size)}')
        
    # Pseudo AIC to get best feat size
    pseudo_aic = -2 * np.log(val_loss) + 2*enc_size
    return input_size, step_feats * (np.argmin(pseudo_aic) + 1)
    
def _get_ae_feats(sig, model):
    # Generate features 
    with torch.no_grad():
        return model(torch.Tensor(sig)).numpy()

def _get_time_series_feats(sig):
    '''
    Given a time series signal of shape (NxD), return an array of interpretable features (NxM)
    '''
    feat_dict = get_temporal_feats(sig)
    return np.array(list(feat_dict.values()))