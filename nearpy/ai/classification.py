import pickle 
from tqdm import tqdm
from pathlib import Path 
import numpy as np

from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.svm import TimeSeriesSVC
from tslearn.clustering import TimeSeriesKMeans
from sklearn.ensemble import (
    RandomForestClassifier, 
    VotingClassifier, 
    AdaBoostClassifier, 
    BaggingClassifier
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

from ..utils import get_accuracy, fn_timer
from .utils import get_dataframe_subset, adapt_dataset_to_tslearn 
from ..utils.logs import log_print
from ..plots import plot_pretty_confusion_matrix

def classify_gestures(data, 
                      save_path, 
                      clf, 
                      data_type='time',
                      subject_key='subject', 
                      routine_key='routine', 
                      class_key='gesture', 
                      exp_name='kfold', 
                      exp_type='kfcv', 
                      n_splits=4, 
                      visualize=True, 
                      logger=None):
    '''
    Performs k-Fold Cross-Validation and Leave-One-Routine Out Testing for multi-class classification. Optionally, benchmarks classifier's inference performance  
    '''
    log_print(logger, 'info', f'Experiment: {exp_name}')
    save_path = Path(save_path) / exp_name
    Path.mkdir(save_path, exist_ok=True, parents=True)
    res_fname = save_path/ 'results.pkl'
    
    if res_fname.exists(): 
        log_print(logger, 'debug', 'Loading pre-existing results')
        cmat, acc, bench = pickle.load(open(res_fname, 'rb'))
    else:
        log_print(logger, 'debug', 'Creating new confusion matrix')
        cmat, acc, bench = {}, {}, {}

    # Set up tqdm 
    total_steps = _get_total_steps(
        data=data, 
        subject_key=subject_key, 
        routine_key=routine_key, 
        exp_type=exp_type,
        n_splits=n_splits
    )
    pbar = tqdm(total=total_steps, desc='Classification Progress')
    
    classes = list(set(data[class_key]))
    num_classes = len(classes)
    subjects = list(set(data[subject_key]))
    
    for sub in subjects:
        # Get classifier object 
        if exp_type == 'loro':
            cmat[sub], acc[sub], bench[sub] = _classify_loro(
                clf=clf, 
                data=data, 
                num_classes=num_classes, 
                subject_num=sub, 
                routine_key=routine_key, 
                subject_key=subject_key, 
                pbar=pbar, 
                data_type=data_type,                
                logger=logger
            )
        else: 
            cmat[sub], acc[sub], bench[sub] = _classify_kfcv(
                clf=clf, 
                data=data, 
                num_classes=num_classes, 
                subject_num=sub, 
                n_splits=n_splits, 
                pbar=pbar, 
                data_type=data_type,
                logger=logger
            )
        
        log_print(logger, 'info', f'Subject {sub} Accuracy: {acc[sub]}')
        pickle.dump([cmat, acc, bench], open(res_fname, 'wb'))
        
    pbar.close()
    
    # Print overall accuracy 
    log_print(logger, 'info', f'Overall Accuracy for {exp_name} {exp_type}: {get_accuracy(cmat)}')
    
    if visualize:
        plot_pretty_confusion_matrix(cmat, 
                                     classes, 
                                     save=True, 
                                     save_path=save_path)
        
def _classify_loro(clf, data, num_classes, 
                    subject_num, routine_key, 
                    subject_key, random_state=42, 
                    pbar=None, data_type='time', logger=None): 
    subset_map = { subject_key: subject_num }
    routines = list(set(get_dataframe_subset(data, subset_map)[routine_key]))
    if data_type == 'time':
        X, y, routs = adapt_dataset_to_tslearn(data, subset_val=subject_num)
    else:
        subset_map = {
            subject_key: subject_num,
        }
        subset = get_dataframe_subset(data, subset_map)
        X, y, routs = np.squeeze(list(subset['mag'])), np.array(subset['gesture']), np.array(subset[routine_key])

    cm = np.zeros((num_classes, num_classes))
    clf_benchmark = {
        'train': [], 
        'infer': []
    } 
    
    for rt in routines:
        log_print(logger, 'debug', f'Excluding routine {rt}')
        test_idx = (routs == rt)     
        # Train/Val/Test Split with Test being new routine
        X_train, X_val, y_train, y_val = train_test_split(X[~test_idx], y[~test_idx], test_size=0.3, random_state=random_state)
        X_test = X[test_idx]
        y_test = y[test_idx]
        
        log_print(logger, 'debug', f'Train: {X_train.shape, y_train.shape} \nVal: {X_val.shape, y_val.shape} \nTest: {X_test.shape, y_test.shape}')
        
        _, train_time = fn_timer(clf.fit, X_train, y_train)
        clf_benchmark['train'].append(train_time/len(y_train))
        
        # Validate
        log_print(logger, 'debug', f'Validation Accuracy: {clf.score(X_val, y_val)}') 
        
        y_pred, infer_time = fn_timer(clf.predict, X_test) # Test
        clf_benchmark['infer'].append(infer_time/len(y_test))
        
        cm += confusion_matrix(y_test, y_pred)
         
        if pbar is not None:
            pbar.update(1)
    
    return cm, get_accuracy(cm), clf_benchmark

def _classify_kfcv(clf, data, num_classes, 
                    subject_num, n_splits=4, 
                    random_state=42, pbar=None, 
                    data_type='time', logger=None):
    if data_type == 'time':
        # If we are working with non-feature datasets, we need to adapt our data 
        X, y, _ = adapt_dataset_to_tslearn(data, subset_val=subject_num)
    else:
        # Otherwise for feature datasets, we keep flatenned arrays
        subset_map = {
            'subject': subject_num,
        }
        subset = get_dataframe_subset(data, subset_map)
        X, y = np.squeeze(list(subset['mag'])), np.array(subset['gesture'])
    
    cm = np.zeros((num_classes, num_classes))
    clf_benchmark = {
        'train': [], 
        'infer': []
    } 
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train, test in kf.split(X, y):
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]
        
        log_print(logger, 'debug', f'Train: {X_train.shape, y_train.shape} \nTest: {X_test.shape, y_test.shape}')
        
        _, train_time = fn_timer(clf.fit, X_train, y_train)
        clf_benchmark['train'].append(train_time/len(y_train))
        # clf.fit(X_train, y_train)
        
        y_pred, infer_time = fn_timer(clf.predict, X_test)
        clf_benchmark['infer'].append(infer_time/len(y_test))
        # y_pred = clf.predict(X_test)
        
        cm += confusion_matrix(y_test, y_pred)
        
        if pbar is not None:
            pbar.update(1) 

    return cm, get_accuracy(cm), clf_benchmark

def get_classifier_obj(config): 
    '''
    Get a classifier object with either default values or a Grid Search parameter setup 
    '''
    classifier = config.get('classifier')
    assert classifier is not None, 'There must always be a classifier.'
    
    dt = DecisionTreeClassifier(max_depth=config.get('depth', 5)) 
    
    dist_clfs = {
        'svc':TimeSeriesSVC(kernel=config.get('metric'), 
                            gamma=0.1, 
                            random_state=42),
        'kmeans': TimeSeriesKMeans(metric=config.get('metric'), 
                                   n_clusters=config.get('n_clusters', 9),
                                   random_state=42),
        'nn': KNeighborsTimeSeriesClassifier(n_neighbors=1, 
                                             metric=config.get('metric'))
    }
    feat_clfs = {
        'rforest': RandomForestClassifier(n_estimators=config.get('n_estimators', 10), 
                                          criterion='log_loss', 
                                          random_state=42),
        'lda': LinearDiscriminantAnalysis(solver=config.get('solver', 'svd')), 
        'adaboost': AdaBoostClassifier(estimator=dt, 
                                       n_estimators=config.get('n_estimators', 10), 
                                       random_state=42), 
        'bagging': BaggingClassifier(estimator=dt, 
                                     n_estimators=config.get('n_estimators', 10), 
                                     n_jobs=-1, 
                                     random_state=42), 
        'qda': QuadraticDiscriminantAnalysis()
    }
    
    if classifier in feat_clfs.keys():
        clf = feat_clfs[classifier]
    elif classifier in dist_clfs.keys():
        clf = dist_clfs[classifier]
    else: 
        # Ensemble by default
        clf = VotingClassifier(estimators=list(feat_clfs.items()), 
                               voting=config.get('voting', 'hard'), 
                               n_jobs=-1)    
    
    grid_params = config.get('params')
    if grid_params is not None: 
        # Ensure we use all processors 
        clf = GridSearchCV(clf, grid_params, refit=True, n_jobs=-1)
        
    return clf

def _get_total_steps(data, subject_key, routine_key, exp_type='kfcv', **kwargs):
    subjects = list(set(data[subject_key]))
    if exp_type == 'loro':
        total_steps = 0
        for sub in subjects:
            subset_map = { subject_key: sub }
            routines = list(set(get_dataframe_subset(data, subset_map)[routine_key]))
            total_steps += len(routines)
        return total_steps
    else: 
        return kwargs.get('n_splits') * len(subjects)