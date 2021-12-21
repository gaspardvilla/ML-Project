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


def  classification_accuracy_transformed(y_true, y_pred):
    target_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6']
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)
    return None


# --------------------------------------------------------------------------------------- #


def split_data(X, y, n_s = 5):
    """
    Split the data in a balanced way

    Args :
        X : dataset to split
        y : target to split
        n_s : number of splits

    Return the resulting split data in a X_train, y_train, X_test, y_test
    """
    skf = StratifiedKFold(n_splits = n_s, shuffle=True, random_state=0)
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

        #print('Train: ', list_train)
        #Sprint('Test: ', list_test)

        return X_train, y_train, X_test, y_test
        break


# --------------------------------------------------------------------------------------- #


def classes_transformed(classes):
    lb = preprocessing.LabelBinarizer()
    return pd.DataFrame(lb.fit_transform(classes))


# --------------------------------------------------------------------------------------- #


def smote_data_augmentation (X, y):
    """
    data augmentation for imbalanced problem using the SMOTE algorithm

    Args:
        X, y: dataset to resample

    Return the resampled X and y
    """
    sm = SMOTE(sampling_strategy='auto', random_state=0)
    X_rs, y_rs = sm.fit_resample(X, y)

    return X_rs, y_rs


# --------------------------------------------------------------------------------------- #


def save_model(filename, model):
    """
    Save model

    Args:
        filename: name of the pickel file ('name.pkl')
        model: tuned model

    Return a pickel file containing the tuned model
    """
    # save the model to disk
    return pickle.dump(model, open(filename, 'wb'))


# --------------------------------------------------------------------------------------- #


def load_model(filename):
    """ 
    Load the model from disk

    Args: 
        filename: name of the file to load

    Return the model
    """
    return pickle.load(open(filename, 'rb'))


# --------------------------------------------------------------------------------------- #


def save_selected_features(filename, model, X):
    """
    Save features obtained after feature selection

    Args:
        filename: name of the pickel file ('name.pkl')
        model: fitted model obtained after feature selection

    Return a pickel file containing an array of the name of the selected features
    """
    feature_idx = model.get_support()
    selected_freatures = X.columns[feature_idx]
    #selected_freatures = model.get_feature_names_out()
    return pickle.dump(selected_freatures, open(filename, 'wb'))


# --------------------------------------------------------------------------------------- #


def load_selected_features(filename, X):
    """ 
    Load the selected features from disk

    Args: 
        filename: name of the file to load

    Return the data with the selected features
    """
    selected_features = pickle.load(open(filename, 'rb'))

    return X[selected_features]