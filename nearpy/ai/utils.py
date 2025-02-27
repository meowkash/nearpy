import numpy as np 

# tslearn dataset spec  is (num_cases, time_steps, num_channels)
def adapt_dataset_to_tslearn(dataset, num_channels=16, 
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

def get_dataframe_subset(df, map_dict=None):
    if map_dict is None: 
        return df  
    
    subset_df = df
    for k, v in map_dict.items(): 
        subset_df = subset_df.loc[subset_df[k] == v]

    return subset_df