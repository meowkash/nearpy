import os 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from ..utils import get_accuracy

def plot_pretty_confusion_matrix(cmat, gestures, cmap='Greens', sub_id=None, save=False, save_path=None):
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
        _plot_pretty_confusion_matrix(cmat, gestures, plot_title, cmap, save, spath)
    else:        
        # Plot confusion matrices for each subject
        for sub, cm in cmat.items():
            cc += cm
            if sub_id == 'All':
                plot_title = f'Classification Accuracy for Subject {sub}'
                if save:
                    spath = os.path.join(save_path, f'confusion_matrix_sub_{sub}')
                _plot_pretty_confusion_matrix(cm, gestures, plot_title, cmap, save, spath)
        
        # Plot overall confusion matrix
        plot_title = 'Overall classification accuracy'
        if save:
            spath = os.path.join(save_path, f'overall_confusion_matrix')
        _plot_pretty_confusion_matrix(cc, gestures, plot_title, cmap, save, spath)

def _plot_pretty_confusion_matrix(cm, gestures, plot_title, cmap, save=False, save_path=None):
    acc = get_accuracy(cm)

    cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)
    mask = (cm == 0)
    
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    
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
        linewidths=0.5,
        mask=mask,
        cbar_kws={
            "shrink": 0.8, 
            "label": "Proportion", 
            "drawedges": False,
            "ticks": [0, 0.25, 0.5, 0.75, 1.0]
        },
        annot_kws={
            'weight': 'medium',  
            'fontsize': 14
        }, 
        square=True
    )
    
    ax.set_ylabel('Actual', fontsize=18)
    ax.set_xlabel('Predicted', fontsize=18)
    
    ax.set_title(f'{plot_title}: {round(acc*100, 2)}%', fontsize=18, fontweight='bold') 
    
    # Set ticks on both sides of axes
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticklabels(gestures, rotation=45, ha='right')
    ax.set_yticklabels(gestures, rotation=0)

    # Adjust layout
    plt.tight_layout()
    
    if save:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        fig.show()