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
    # acc = sum([cm[i, i] for i in range(num_gestures)])/np.concatenate(cm).sum()
    acc = get_accuracy(cm)

    cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)
    mask = (cm == 0)
    
    fig = plt.figure(dpi=200)
    sns.heatmap(cm, annot=True, xticklabels=gestures, 
                yticklabels=gestures, cmap=cmap, mask=mask,
                annot_kws={"fontname":"serif", "fontsize":11}, square=True)
    plt.ylabel('Actual', fontsize=14, fontname='serif')
    plt.xlabel('Predicted',fontsize=14, fontname='serif')
    plt.title(f'{plot_title}: {round(acc*100, 2)}%', fontsize=14, fontname='serif') 
    
    if save:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        fig.show()