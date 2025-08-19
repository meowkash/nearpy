import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import auc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from nearpy.utils import normalize
from nearpy.preprocess import get_adaptive_segment_indices

# Highlight what segmentation method works 
def plot_segmentation_results(sig, fs=10000, thr=0.9): 
    methods = ['Abs', 'Square', 'Movstd', 'SNR']
    
    # Make base line plot 
    timeAx = np.linspace(0, len(sig)/fs, len(sig))
    
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), dpi=300)
    
    cmap = sns.color_palette('pastel', len(methods))
    cmap_deep = sns.color_palette('deep', len(methods))
    
    for i, method in enumerate(methods): 
        axs[i//2, i%2].plot(timeAx, normalize(sig), color='grey')
        idx, vals, probs = get_adaptive_segment_indices(sig=sig,
                                                        timeAx=timeAx, 
                                                        fs=fs, 
                                                        method=method, 
                                                        prob_thresh=thr, 
                                                        sig_band=[80, 5000],
                                                        noise_band=[0, 80])
        marker_indices = timeAx[idx] 
        marker_values = normalize(sig)[idx]
        
        auc_val = np.round(auc(vals, probs), 3)
        # Plot AUC curve as an inset 
        axins = inset_axes(axs[i//2, i%2], 
                           width='45%', 
                           height='30%', 
                           loc='lower left')
        axins.plot(normalize(vals), probs, 
                   color=cmap_deep[i],
                   label=f'AUC: {auc_val}')
        axins.set_xticks([])
        axins.legend(fontsize=10, loc='center right')
        axins.set_yticks([])
        
        # Plot markers on top of the line
        axs[i//2, i%2].scatter(marker_indices, marker_values, color=cmap[i], s=30, 
                marker='o', edgecolor=cmap[i], linewidth=1.5)
        axs[i//2, i%2].set_xlabel('Time (s)')
        axs[i//2, i%2].set_ylabel('Amplitude (a.u.)')
        axs[i//2, i%2].set_title(f'{method}')
    
    fig.tight_layout()
    fig.show()