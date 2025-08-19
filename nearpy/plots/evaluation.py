import os 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from nearpy.utils import get_accuracy

def plot_pretty_confusion_matrix(cmat, 
                                 gestures, 
                                 cmap=sns.light_palette("brown", as_cmap=True), 
                                 sub_id=None, 
                                 save=False, 
                                 save_path=None,
                                 figsize=(10, 10), 
                                dpi=300):
    # Store overall confusion matrix over all subjects 
    cc = np.zeros((len(gestures), len(gestures)))

    spath = None
    if save_path is None:
        save_path = os.getcwd()
    elif not(os.path.isdir(save_path)):
        os.mkdir(save_path)
    
    if isinstance(sub_id, int):
        plot_title = f'Classification Accuracy for Subject {sub_id}'
        if save:
            spath = os.path.join(save_path, f'confusion_matrix_sub_{sub_id}')
        _plot_pretty_confusion_matrix(cmat, gestures, plot_title, cmap, save, spath, figsize, dpi)
    else:        
        # Plot confusion matrices for each subject
        for sub, cm in cmat.items():
            cc += cm
            if sub_id == 'All':
                plot_title = f'Classification Accuracy for Subject {sub}'
                if save:
                    spath = os.path.join(save_path, f'confusion_matrix_sub_{sub}')
                _plot_pretty_confusion_matrix(cm, gestures, plot_title, cmap, save, spath, figsize, dpi)
        
        # Plot overall confusion matrix
        plot_title = 'Classification Accuracy'
        if save:
            spath = os.path.join(save_path, f'overall_confusion_matrix')
        _plot_pretty_confusion_matrix(cc, gestures, plot_title, cmap, save, spath, figsize, dpi)

def _plot_pretty_confusion_matrix(cm, 
                                  gestures, 
                                  plot_title, 
                                  cmap, 
                                  save=False, 
                                  save_path=None,
                                  figsize=(10, 10), 
                                  dpi=300):
    acc = get_accuracy(cm)

    cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)
    mask = (cm == 0)
    sns.set_style('white')
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    })
    
    sns.heatmap(
        cm, 
        annot=True, 
        xticklabels=gestures, 
        yticklabels=gestures, 
        cmap=cmap,
        fmt='',
        linewidths=0.5,
        linecolor='#DEDEDE',
        mask=mask,
        cbar=True,
        cbar_kws={
            "shrink": 0.5, 
            "label": "", 
            "drawedges": False,
            "ticks": [0, 0.25, 0.5, 0.75, 1.0]
        },
        annot_kws={
            'weight': 'normal',  
            'fontsize': 16
        }, 
        square=True
    )
    
    ax.set_ylabel('Actual', fontsize=22)
    ax.set_xlabel('Predicted', fontsize=22)
    
    ax.set_title(f'{plot_title}: {round(acc*100, 2)}%', 
                 fontsize=24, pad=20) 
    
    # Set ticks on both sides of axes
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticklabels(gestures, rotation=45,
                       ha='right', fontsize=18, wrap=True)
    ax.set_yticklabels(gestures, rotation=0, fontsize=18, wrap=True)

    # Adjust layout
    plt.tight_layout()
    
    if save:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        fig.show()