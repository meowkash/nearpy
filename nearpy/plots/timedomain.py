from typing import Dict

import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 

from nearpy.utils import TxRx
from nearpy.preprocess import get_segments_template

from .utils import get_grid_layout


# Show per-routine averages, both longitudinal and otherwise, to glean insights from data. 
def plot_routine_template(df, title="", num_channels=16, show_individual=True, dtw_avg=False): 
    sns.set_theme(style="whitegrid")
    colmap = sns.color_palette('husl', 16)
    
    # Assuming this is a subject data-frame
    for rt in set(df['routine']):
        elems = df.loc[df['routine'] == rt]['mag']
        stacked_elems = np.vstack(elems).reshape(len(elems), num_channels, -1)

        if dtw_avg: 
            channel_averages = get_segments_template(stacked_elems, 'dba')
        else:
            channel_averages = get_segments_template(stacked_elems, 'mean')
        
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

def plot_time_series(data: Dict):
    '''
    Generate publication ready multi-grid subplots given data specified using the following format - 
    data = {
        'subplot_title': {
            'series1_label': [time-series],
            'series2_label': [time-series],
            ...
            'seriesK_label': [time-series],
            'time_axis': [time-series],
            'xlabel': (optional)
            'ylabel': (optional)
        }
    }
    '''
    # Set publication-ready styling with SciencePlots
    plt.style.use(['science', 'ieee'])
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    n_subplots = len(data)
    
    # Calculate grid layout
    nrows, ncols = get_grid_layout(n_subplots)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), 
                    constrained_layout=True)
    
    if n_subplots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_subplots > 1 else [axes]
    
    colors = plt.cm.Set1(np.linspace(0, 1, 10))
    
    for idx, (subplot_title, subplot_data) in enumerate(data.items()):
        ax = axes[idx]
        
        time_axis = subplot_data.get('time_axis', None)
        xlabel = subplot_data.get('xlabel', 'Time')
        ylabel = subplot_data.get('ylabel', 'Value')
        
        color_idx = 0
        for series_label, series_values in subplot_data.items():
            if series_label in ['time_axis', 'xlabel', 'ylabel']:
                continue
                
            x_data = time_axis if time_axis is not None else range(len(series_values))
            
            ax.plot(x_data, series_values, label=series_label, 
                   color=colors[color_idx % len(colors)], alpha=0.8)
            color_idx += 1
        
        ax.set_title(subplot_title, fontweight='bold', pad=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)
        
        # Publication formatting
        ax.tick_params(direction='in', which='both', top=True, right=True)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
    
    # Hide empty subplots
    for idx in range(n_subplots, len(axes)):
        axes[idx].set_visible(False)
    
    return fig, axes