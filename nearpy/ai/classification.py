import pickle 
from tqdm import tqdm
from pathlib import Path 
import numpy as np

from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.svm import TimeSeriesSVC
from tslearn.clustering import TimeSeriesKMeans
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold

from ..utils import get_accuracy
from .utils import get_dataframe_subset, adapt_dataset_to_tslearn
from ..plots import plot_pretty_confusion_matrix
    
def classify_gestures(data, base_path, clf, 
                 subject_key='subject', routine_key='routine', 
                 class_key='gesture', exp_name='kfold', 
                 exp_type='kfcv', n_splits=4, visualize=True):
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
        if exp_type == 'loro':
            cmat[sub], acc[sub] = _classify_loro(clf, data, num_classes, sub, routine_key, subject_key, pbar=pbar)
        else: 
            cmat[sub], acc[sub] = _classify_kfcv(clf, data, num_classes, subject_num=sub, n_splits=n_splits, pbar=pbar)
        
        print(f'Subject {sub} Accuracy: {acc[sub]}')
        pickle.dump([cmat, acc], open(res_fname, 'wb'))
        
    pbar.close()
    # Print overall accuracy 
    print(f'Overall Accuracy: {get_accuracy(cmat)}')
    
    if visualize:
        plot_pretty_confusion_matrix(cmat, classes, save=True, save_path=base_path)
        
def _classify_loro(clf, data, num_classes, 
                            subject_num, routine_key, 
                            subject_key, random_state=42, 
                            pbar=None, debug=False): 
    subset_map = { subject_key: subject_num }
    routines = list(set(get_dataframe_subset(data, subset_map)[routine_key]))
    
    X, y, routs = adapt_dataset_to_tslearn(data, subset_val=subject_num)
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

def _classify_kfcv(clf, data, num_classes, 
                            subject_num, n_splits=4, 
                            random_state=42, pbar=None, debug=False): 
    X, y, _ = adapt_dataset_to_tslearn(data, subset_val=subject_num)
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

def get_classifier_obj(classifier, **kwargs): 
    '''
    - SVM with GAK is seen to ba a good alternative to 1-NN with DTW since the latter is not a true distance. Kernel Options: gak, rbf, poly, sigmoid
    - LDA: eigen, lsqr
    - 
    '''
    dt = DecisionTreeClassifier(max_depth=kwargs.get('depth', 5)) 
    dist_clfs = {
        'svc':TimeSeriesSVC(kernel=kwargs.get('metric'), gamma=0.1),
        'kmeans': TimeSeriesKMeans(random_state=42, metric=kwargs.get('metric'), n_clusters=kwargs.get('n_clusters', 9)),
        'nn': KNeighborsTimeSeriesClassifier(n_neighbors=1, metric=kwargs.get('metric'))
    }
    feat_clfs = {
        'rforest': RandomForestClassifier(n_estimators=kwargs.get('n_estimators', 10), 
                                          criterion='log_loss', random_state=42),
        'lda': LinearDiscriminantAnalysis(solver=kwargs.get('solver', 'svd')), 
        'adaboost': AdaBoostClassifier(estimator=dt, n_estimators=kwargs.get('n_estimators', 10), 
                                       algorithm='SAMME', random_state=42), 
        'bagging': BaggingClassifier(estimator=dt, n_estimators=kwargs.get('n_estimators', 10))
    }
    
    if classifier in feat_clfs.keys():
        clf = feat_clfs[classifier]
    elif classifier in dist_clfs.keys():
        clf = dist_clfs[classifier]
    else: 
        # Ensemble by default
        clf = VotingClassifier(estimators=feat_clfs.items(), voting=kwargs.get('voting', 'hard'))    
    
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