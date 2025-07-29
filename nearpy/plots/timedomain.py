import matplotlib.pyplot as plt
import seaborn as sns 

import numpy as np 
from tslearn.barycenters import softdtw_barycenter as DBA
from pathlib import Path 
import pandas as pd 

from nearpy.utils import TxRx

from typing import Tuple
from lets_plot import *

# Show per-routine averages, both longitudinal and otherwise, to glean insights from data. 
def plot_routine_template(df, title="", num_channels=16, show_individual=True, dtw_avg=False): 
    sns.set_theme(style="whitegrid")
    colmap = sns.color_palette('husl', 16)
    
    # Assuming this is a subject data-frame
    for rt in set(df['routine']):
        elems = df.loc[df['routine'] == rt]['mag']
        stacked_elems = np.vstack(elems).reshape(len(elems), num_channels, -1)
        if dtw_avg: 
            channel_averages = DBA(stacked_elems)
        else:
            channel_averages = np.mean(stacked_elems, axis=0)
        
        # Declare figure     
        fig, axes = plt.subplots(4, 4, figsize=(12, 10), sharex=True)
        fig.suptitle(f'{title}, Routine: {rt}', fontsize=16, fontweight='bold')
        
        for i, ax in enumerate(axes.flat):
            # Plot individual time series in gray 
            if show_individual:
                for j in range(len(elems)): 
                    ax.plot(stacked_elems[j, i, :], color='gainsboro')
                    
            # Plot template
            ax.plot(channel_averages[i], label=TxRx(i, 4), linewidth=2, color=colmap[i])
            ax.legend(fontsize=10)
            ax.set_ylabel("Value", fontsize=12)
            ax.tick_params(axis='both', labelsize=10)
            ax.set_ylim([0, 1])
        
        plt.xlabel("Time", fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def plot_time_series(
        data: np.ndarray, 
        fs: float, 
        title: str = "Time Series", 
        figsize: Tuple[int, int] = (10, 4)
):
    t = np.arange(len(data)) / fs
    
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.plot(t, data, linewidth=0.8, color='#1f77b4')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax

def plot_time_series_from_df(subject, routine, base_path, fs, start_time, 
                     channels=[0, 5, 10, 15], data_key='filt_mag'):
    # Load data from folder
    LetsPlot.setup_html()
    num_vars = len(channels)
    data_path = Path(base_path)
    file_path = data_path / f'Subject {subject}' / f'Routine {routine}.npz'
    
    if not file_path.exists():
        print(f'File {file_path} not found. Re-check')
        return
    
    file_data = np.load(file_path)
    data = file_data[data_key]
    
    st_idx = round(start_time * fs)
    num_pts = data.shape[1] - st_idx
    timeAx = np.linspace(1, num_pts / fs, num=num_pts)
    labels = [TxRx(i, 4) for i in channels]

    # Create data structure for plotting
    plot_df = pd.DataFrame(np.transpose(data[channels, st_idx:]), columns=labels)
    plot_df['time'] = timeAx
    
    plot_data = plot_df.melt(id_vars='time', value_vars=labels)
    
    p = ggplot(plot_data, aes(x='time', y='value', color='variable')) + \
            geom_line() + facet_wrap(facets='variable', ncol=1, scales='free_y') + \
            scale_x_continuous() + scale_color_hue() + ggtb() + ggsize(1200, 1000)

    p.show()
