import math 
import numpy as np 

def get_accuracy(cm):
    if type(cm) is dict:
        num_classes = cm[list(cm.keys())[0]].shape[0]
        return _get_dict_accuracy(cm, num_classes)
    else:
        num_classes = cm.shape[0]
        return _get_accuracy(cm, num_classes)    

def _get_accuracy(cm, num_classes):
    return sum([cm[i, i] for i in range(num_classes)])/np.concatenate(cm).sum()

def _get_dict_accuracy(cm, num_classes):
    return np.average([_get_accuracy(cc, num_classes) for _, cc in cm.items()])

def TxRx(x, n_mimo=4):
    return 'Tx' + str((x)%n_mimo +1) + 'Rx' + str(math.ceil((x+1)/n_mimo))