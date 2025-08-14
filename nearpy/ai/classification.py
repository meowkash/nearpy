import pickle 
from tqdm import tqdm
from pathlib import Path 
import numpy as np
import pandas as pd 

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

from nearpy.utils import get_accuracy, fn_timer
from nearpy.io import log_print
from nearpy.plots import plot_pretty_confusion_matrix

from .utils import get_dataframe_subset, adapt_dataset_to_tslearn 

def classify_gestures(data, 
                      save_path: Path, 
                      clf, 
                      data_type: str = 'time',
                      class_key: str = 'gesture', 
                      data_key: str = 'mag', 
                      routine_key: str = 'routine', 
                      subject_key: str = 'subject', 
                      exp_name: str = 'kfold', 
                      exp_type: str = 'kfcv', 
                      n_splits: int = 4, 
                      num_vars: int  = 16, 
                      visualize: bool = True, 
                      logger = None):
    '''
    Performs k-Fold Cross-Validation and Leave-One-Routine Out Testing for multi-class classification. 
    Optionally, benchmarks classifier's inference performance  
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
                class_key=class_key, 
                data_key=data_key, 
                routine_key=routine_key, 
                subject_key=subject_key, 
                data_type=data_type,    
                num_vars=num_vars,            
                pbar=pbar, 
                logger=logger
            )
        else: 
            cmat[sub], acc[sub], bench[sub] = _classify_kfcv(
                clf=clf, 
                data=data, 
                num_classes=num_classes, 
                subject_num=sub, 
                class_key=class_key, 
                data_key=data_key, 
                routine_key=routine_key, 
                subject_key=subject_key, 
                data_type=data_type,
                n_splits=n_splits, 
                num_vars=num_vars,
                pbar=pbar, 
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
        
def _classify_loro(clf, 
                   data: pd.DataFrame, 
                   num_classes: int, 
                   subject_num: int, 
                   class_key: str, 
                   data_key: str, 
                   routine_key: str, 
                   subject_key: str, 
                   data_type: str = 'time', 
                   pbar = None, 
                   logger = None,
                   num_vars: int = 16, 
                   random_state: int = 42, 
                ): 
    subset_map = { subject_key: subject_num }
    routines = list(set(get_dataframe_subset(data, subset_map)[routine_key]))
    
    if data_type == 'time':
        X, y, routs = adapt_dataset_to_tslearn(
            data, 
            num_vars=num_vars,
            subject_num=subject_num,
            class_key=class_key,
            data_key=data_key,
            routine_key=routine_key, 
            subject_key=subject_key 
        )
    else:
        subset_map = {
            subject_key: subject_num,
        }
        subset_df = get_dataframe_subset(data, subset_map)
        # Check for NaN
        subset_df.dropna(inplace=True)
        X, y, routs = np.squeeze(list(subset_df[data_key])), np.array(subset_df[class_key]), np.array(subset_df[routine_key])

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

def _classify_kfcv(clf, 
                   data: pd.DataFrame, 
                   num_classes: int, 
                   subject_num: int, 
                   class_key: str, 
                   data_key: str, 
                   routine_key: str, 
                   subject_key: str, 
                   data_type: str = 'time', 
                   n_splits: int = 4, 
                   num_vars: int = 16, 
                   pbar = None, 
                   logger = None,
                   random_state: int = 42, 
                ):
    if data_type == 'time':
        # If we are working with non-feature datasets, we need to adapt our data 
        X, y, _ = adapt_dataset_to_tslearn(
            data, 
            num_vars=num_vars,
            subject_num=subject_num,
            class_key=class_key,
            data_key=data_key,
            routine_key=routine_key, 
            subject_key=subject_key 
        )
    else:
        # Otherwise for feature datasets, we keep flatenned arrays
        subset_map = {
            subject_key: subject_num,
        }
        subset_df = get_dataframe_subset(data, subset_map)

        # Check for NaN
        subset_df.dropna(inplace=True)

        X, y = np.squeeze(list(subset_df[data_key])), np.array(subset_df[class_key])
    
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
        'qda': QuadraticDiscriminantAnalysis(reg_param=0.1)
    }
    
    if classifier in feat_clfs.keys():
        clf = feat_clfs[classifier]
    elif classifier in dist_clfs.keys():
        clf = dist_clfs[classifier]
    else: 
        estimator_list = config.get('estimators', None)
        
        if estimator_list is None: 
            estimator_list = list(feat_clfs.items())
        else: 
            estimator_list = [(key, feat_clfs[key]) for key in estimator_list if key in feat_clfs.keys()]
        
        # Ensemble by default
        clf = VotingClassifier(estimators=estimator_list, 
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