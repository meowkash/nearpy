import numpy as np 

# All functions below assume a confusion matrix is provided
def get_accuracy(confmat):
    if type(confmat) is dict:
        num_classes = confmat[list(confmat.keys())[0]].shape[0]
        return _get_dict_accuracy(confmat, num_classes)
    else:
        num_classes = confmat.shape[0]
        return _get_accuracy(confmat, num_classes)    

def get_class_accuracy(confmat): 
    # TODO: Make more compact 
    # Given a confusion matrix, find individual class accuracy
    accs = [None] * np.shape(confmat)[0]
    for idx, row in enumerate(confmat): 
        accs[idx] = row[idx]/np.sum(row)
    return accs

def _get_accuracy(confmat, num_classes):
    return sum([confmat[i, i] for i in range(num_classes)])/np.concatenate(confmat).sum()

def _get_dict_accuracy(confmat, num_classes):
    return np.average([_get_accuracy(cc, num_classes) for _, cc in confmat.items()])