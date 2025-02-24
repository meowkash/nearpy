import pickle 
from tqdm import tqdm
from pathlib import Path 
import numpy as np

from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.svm import TimeSeriesSVC
from tslearn.clustering import TimeSeriesKMeans

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold

from .utils import make_tslearn_dataset
from ..utils import get_accuracy, get_small_dataframe
from ..plots import plot_pretty_confusion_matrix
    
def distance_classify_gestures(data, base_path, 
                 subject_key='subject', routine_key='routines', 
                 class_key='gesture', metric='dtw',
                 exp_name='kfold', exp_type='kfcv', 
                 classifier='nn', n_splits=4, visualize=True):
    print(f'Experiment Details: \n{classifier} classifier with {metric} metric')
    
    res_fname = Path(base_path) / f'results_{exp_name}.pkl'
    if res_fname.exists(): 
        print('Loading pre-existing results')
        cmat, acc = pickle.load(open(res_fname, 'rb'))
    else:
        print('Creating new confusion matrix')
        cmat, acc = {}, {}

    # Set up tqdm 
    total_steps = _get_total_steps(data, subject_key,
                                  routine_key, exp_type,
                                  n_splits=n_splits)
    pbar = tqdm(total=total_steps, desc='Classification Progress')
    
    classes = list(set(data[class_key]))
    num_classes = len(classes)
    subjects = list(set(data[subject_key]))
    
    for sub in subjects:
        # Get classifier object 
        clf = _get_classifier(classifier, metric)
        print(f'Processing Subject {sub}')    

        if exp_type == 'loro':
            cmat[sub], acc[sub] = _distance_classify_loro(clf, data, num_classes, sub, routine_key, subject_key, pbar=pbar)
        else: 
            cmat[sub], acc[sub] = _distance_classify_kfcv(clf, data, num_classes, subject_num=sub, n_splits=n_splits, pbar=pbar)
        
        print(f'Subject {sub} Accuracy: {acc[sub]}')
        pickle.dump([cmat, acc], open(res_fname, 'wb'))
        
    pbar.close()

    if visualize:
        plot_pretty_confusion_matrix(cmat, classes, save=True, save_path=base_path)
        
def _distance_classify_loro(clf, data, num_classes, 
                            subject_num, routine_key, 
                            subject_key, random_state=42, 
                            pbar=None, debug=False): 
    print(f'Performing leave-one routine out classification')
    
    subset_map = { subject_key: subject_num }
    routines = list(set(get_small_dataframe(data, subset_map)[routine_key]))
    
    X, y, routs = make_tslearn_dataset(data, subset_val=subject_num)
    cm = np.zeros((num_classes, num_classes))
        
    for rt in routines:
        print(f'Excluding routine {rt}')
        test_idx = (routs == rt)     
        # Train/Val/Test Split with Test being new routine
        X_train, X_val, y_train, y_val = train_test_split(X[~test_idx], y[~test_idx], test_size=0.3, random_state=random_state)
        X_test = X[test_idx]
        y_test = y[test_idx]
        
        if debug: 
            print(f'Train: {X_train.shape, y_train.shape} \nVal: {X_val.shape, y_val.shape} \nTest: {X_test.shape, y_test.shape}')
        
        clf.fit(X_train, y_train) # Train
        if debug:
            # Validate
            print(f'Validation Accuracy: {clf.score(X_val, y_val)}') 
        
        y_pred = clf.predict(X_test) # Test
        cm += confusion_matrix(y_test, y_pred)
         
        if pbar is not None:
            pbar.update(1)
    
    return cm, get_accuracy(cm)

def _distance_classify_kfcv(clf, data, num_classes, 
                            subject_num, n_splits=4, 
                            random_state=42, pbar=None, debug=False): 
    
    print(f'Performing {n_splits}-Fold CV')
    
    X, y, _ = make_tslearn_dataset(data, subset_val=subject_num)
    cm = np.zeros((num_classes, num_classes))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train, test in kf.split(X, y):
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]
        
        if debug:
            print(f'Train: {X_train.shape, y_train.shape} \nTest: {X_test.shape, y_test.shape}')
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        cm += confusion_matrix(y_test, y_pred)
        
        if pbar is not None:
            pbar.update(1) 

    return cm, get_accuracy(cm)

def _get_classifier(classifier, metric, **kwargs): 
    if classifier == 'svc':
        # SVM with GAK is seen to ba a good alternative to 1-NN with DTW since the latter is not a true distance. Kernel Options: gak, rbf, poly, sigmoid
        clf = TimeSeriesSVC(kernel=metric, gamma=0.1) 
    elif classifier == 'kmeans':
        # Important: Specify n_clusters
        clf = TimeSeriesKMeans(random_state=42, metric=metric, **kwargs)
    else:
        # By default, simply do 1-NN with DTW
        clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric=metric) 
    
    return clf

def _get_total_steps(data, subject_key, routine_key, exp_type='kfcv', **kwargs):
    subjects = list(set(data[subject_key]))
    if exp_type == 'loro':
        total_steps = 0
        for sub in subjects:
            subset_map = { subject_key: sub }
            routines = list(set(get_small_dataframe(data, subset_map)[routine_key]))
            total_steps += len(routines)
        return total_steps
    else: 
        return kwargs.get('n_splits') * len(subjects)