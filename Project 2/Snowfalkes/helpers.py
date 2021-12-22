# Import the used libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from IPython.display import display

# Import sklearn librairies
from sklearn.feature_selection import *
from sklearn.model_selection import *
from sklearn.ensemble import *
from sklearn.neural_network import *
from sklearn.pipeline import *
from sklearn.preprocessing import *
from sklearn.linear_model import *
from yellowbrick.model_selection import *
from sklearn.svm import *
from sklearn.decomposition import *
from sklearn import *
from sklearn.metrics import *
from sklearn.multiclass import *
from imblearn.over_sampling import SMOTE


# --------------------------------------------------------------------------------------- #


def classification_accuracy(y_true, y_pred):
    """
    Calculate the accurary for each class

    Args: 
        y_true : the real target to reach
        y_pred : the prediction for the target obtained with a model

    Returns:
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

    Returns:
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

        list_train = []
        list_test = []

        for i in range(1, 7):
            n_train = len(y_train[y_train['class_id'] == i])
            n_test = len(y_test[y_test['class_id'] == i])

            list_train.append(n_train)
            list_test.append(n_test)

        return X_train, y_train, X_test, y_test
        break


# --------------------------------------------------------------------------------------- #


def smote_data_augmentation (X, y, seed = 0):
    """
    data augmentation for imbalanced problem using the SMOTE algorithm

    Args:
        X, y: dataset to resample
        seed: seed to use for the random state for SMOTE algorithm (default = 0)

    Returns:
        Resampled X and y
    """
    sm = SMOTE(sampling_strategy='auto', random_state = seed)
    X_rs, y_rs = sm.fit_resample(X, y)

    return X_rs, y_rs


# --------------------------------------------------------------------------------------- #


def save_model(path, model):
    """
    Save model

    Args:
        path: path of the pickel file (.pkl)
        model: tuned model to save

    Returns:
        Pickel file containing the tuned model
    """
    # save the model to disk
    pickle.dump(model, open(path, 'wb'))


# --------------------------------------------------------------------------------------- #


def load_model(path):
    """ 
    Load the model from disk

    Args: 
        path: path of the file to load

    Returns:
        Loaded model
    """
    return pickle.load(open(path, 'rb'))


# --------------------------------------------------------------------------------------- #


def save_selected_features(path, model, X, method):
    """
    Save features obtained after feature selection

    Args:
        path: path of the pickel file (.pkl)
        model: fitted model obtained after feature selection
        model_name: name of the model to save (string)

    Returns:
        Pickel file containing an array of the name of the selected features
    """
    if method == 'PCA':
        pickle.dump(model, open(path, 'wb'))
    else:
        feature_idx = model.get_support()
        selected_features = X.columns[feature_idx]
        pickle.dump(selected_features, open(path, 'wb'))


# --------------------------------------------------------------------------------------- #


def load_selected_features(path, X, method):
    """ 
    Load the selected features from disk

    Args: 
        path: path of the file to load
        X: initial dataset on which feature selection will be applied
        method: method used for feature selection (string)

    Returns:
        Data with the selected features
    """
    if method == 'PCA':
        components = pickle.load(open(path, 'rb'))
        return components.transform(X)
    else:
        selected_features = pickle.load(open(path, 'rb'))
        return X[selected_features]


# --------------------------------------------------------------------------------------- #


def feature_transform(model, X, method):
    """
    Transform the dataset in order to keep only features selected by method

    Args:
        model: model containing the features selected using the method
        X: initial dataset to reduce
        method: method used for feature selection

    Returns:
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
