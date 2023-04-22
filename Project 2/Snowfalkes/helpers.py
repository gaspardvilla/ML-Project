# Import the used libraries
import numpy as np
import pandas as pd
import pickle
from IPython.display import display

# Import sklearn librairies
from sklearn import *
from sklearn.metrics import *
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import *


# --------------------------------------------------------------------------------------- #


def classification_accuracy(y_true, y_pred):
    """
    Calculate the accurary for each class

    Args : 
        y_true : the real target to reach
        y_pred : the prediction for the target obtained with a model

    Returns :
        Classes of the true set
    """
    y_true_ = y_true.reset_index(drop = True)
    classes = y_true_.class_id.unique()
    for class_ in classes:
        msk = y_true_.class_id == class_
        true_set = y_true_[msk]
        pred_set = y_pred[msk]

        print(class_, ' : ', accuracy_score(true_set, pred_set))
    
    return classes


# --------------------------------------------------------------------------------------- #


def split_data(X, y, kfold = 5, seed = 0):
    """
    Split the data in a balanced way

    Args :
        X : dataset to split
        y : target to split
        kfold : number of splits (default = 5)
        seed: seed to use for the random state of the split (default = 0)

    Returns :
        Resulting split data in a X_train, y_train, X_test, y_test
    """
    skf = StratifiedKFold(n_splits = kfold, 
                            shuffle = True, 
                            random_state = seed)
    
    for train_idx, test_idx in skf.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        return X_train, y_train, X_test, y_test
        

# --------------------------------------------------------------------------------------- #


def smote_data_augmentation (X, y, seed = 0):
    """
    Data augmentation for imbalanced problem using the SMOTE algorithm

    Args :
        X, y : dataset to resample
        seed : seed to use for the random state for SMOTE algorithm (default = 0)

    Returns :
        Resampled X and y
    """
    sm = SMOTE(sampling_strategy = 'auto', random_state = seed)
    X_rs, y_rs = sm.fit_resample(X, y)

    return X_rs, y_rs


# --------------------------------------------------------------------------------------- #


def save_model(path, model):
    """
    Save model

    Args :
        path: path of the pickel file (.pkl)
        model: tuned model to save

    Returns :
        Pickel file containing the tuned model
    """
    # save the model to disk
    pickle.dump(model, open(path, 'wb'))


# --------------------------------------------------------------------------------------- #


def save_selected_features(path, model, X, method):
    """
    Save features obtained after feature selection

    Args :
        path : path of the pickel file (.pkl)
        model : fitted model obtained after feature selection
        model_name : name of the model to save (string)

    Returns :
        Pickel file containing an array of the name of the selected features
    """
    if method == 'PCA':
        pickle.dump(model, open(path, 'wb'))
    else:
        feature_idx = model.get_support()
        selected_features = X.columns[feature_idx]
        pickle.dump(selected_features, open(path, 'wb'))


# --------------------------------------------------------------------------------------- #


def feature_transform(model, X, method):
    """
    Transform the dataset in order to keep only features selected by method

    Args :
        model : model containing the features selected using the method
        X : initial dataset to reduce
        method : method used for feature selection

    Returns :
        Dataset reduced containing the features selected by method
    """

    if method == 'PCA':
        # return the new dataset projected on the principal components
        return model.transform(X)
    else:
        feature_idx = model.get_support()
        selected_features = X.columns[feature_idx]
        # return the reduced dataset
        return X[selected_features]


# --------------------------------------------------------------------------------------- #
