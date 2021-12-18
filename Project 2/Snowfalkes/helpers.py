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


class MASCDB_classes:
    
    def __init__(self, dir_path):

        # Get the paths for the hydro training sets
        hydro_cam0_path = os.path.join(dir_path, "hydro_trainingset/hydro_trainingset_cam0.pkl")
        hydro_cam1_path = os.path.join(dir_path, "hydro_trainingset/hydro_trainingset_cam1.pkl")
        hydro_cam2_path = os.path.join(dir_path, "hydro_trainingset/hydro_trainingset_cam2.pkl")

        # Get the paths for the riming training sets
        riming_cam0_path = os.path.join(dir_path, "riming_trainingset/riming_trainingset_cam0.pkl")
        riming_cam1_path = os.path.join(dir_path, "riming_trainingset/riming_trainingset_cam1.pkl")
        riming_cam2_path = os.path.join(dir_path, "riming_trainingset/riming_trainingset_cam2.pkl")

        # Read the dataframes for hydro classes
        self.hydro_cam0 = pd.read_pickle(hydro_cam0_path)
        self.hydro_cam1 = pd.read_pickle(hydro_cam1_path)
        self.hydro_cam2 = pd.read_pickle(hydro_cam2_path)

        # Read the dataframes for riming classes
        self.riming_cam0 = pd.read_pickle(riming_cam0_path)
        self.riming_cam1 = pd.read_pickle(riming_cam1_path)
        self.riming_cam2 = pd.read_pickle(riming_cam2_path)


    def get_class_cam(self, classifier, cam):
        # Select the data for classifier (i.e. riming or hydro) and camera number cam
        if classifier == "riming":
            if cam == 0:
                class_cam = self.riming_cam0
            elif cam == 1:
                class_cam = self.riming_cam1
            elif cam == 2:
                class_cam = self.riming_cam2
            else:
                raise ValueError("Wrong cam, it should be equal to: 0, 1 or 2.")
        elif classifier == "hydro":
            if cam == 0:
                class_cam = self.hydro_cam0
            elif cam == 1:
                class_cam = self.hydro_cam1
            elif cam == 2:
                class_cam = self.hydro_cam2
            else:
                raise ValueError("Wrong cam, it should be equal to: 0, 1 or 2.")
        else:
            raise ValueError("Wrong classifier, it should be either: 'riming' or 'hydro'.")
        return class_cam


    def get_sub_data_cam(self, classifier, cam, cam_data):
        # Get the classifier cam
        class_cam = self.get_class_cam(classifier, cam)

        # Get the sub data frame of cam_data containing flake_id of class_cam
        sub_cam_data = cam_data[cam_data['flake_id'].isin(class_cam['flake_id'])]

        # Return the result
        return sub_cam_data

    def get_sub_classes_cam(self, classifier, cam, cam_data):
        # Get the classifier cam
        class_cam = self.get_class_cam(classifier, cam)

        # Get the sub classes for this cam that are in cam_features
        sub_cam_classes = class_cam[class_cam['flake_id'].isin(cam_data['flake_id'])]

        # Return the classes for a specific camera
        return sub_cam_classes

    def get_classified_data(self, classifier, data_set):
        # For each camera, select the data we are interested in (i.e. the data that was classified)
        # cam0
        classified_data = self.get_sub_data_cam(classifier, 0, data_set.cam0)
        
        # cam1
        classified_data = pd.concat([classified_data, self.get_sub_data_cam(classifier, 1, data_set.cam1)])
        
        # cam2
        classified_data = pd.concat([classified_data, self.get_sub_data_cam(classifier, 2, data_set.cam2)])

        # Return the concatenated data frame that contains all the data point to consider
        return classified_data

    
    def get_classes(self, classifier, data):
        # Get the classes in cam 0
        classes = self.get_sub_classes_cam(classifier, 0, data.cam0)

        # Append the classes that are in cam 1
        classes = pd.concat([classes, self.get_sub_classes_cam(classifier, 1, data.cam1)])

        # Append the classes that are in cam 2
        classes = pd.concat([classes, self.get_sub_classes_cam(classifier, 2, data.cam2)])
       
        # Return all the labels 
        return classes


# --------------------------------------------------------------------------------------- #


def numpy_helpers(df, cols):
    """
        Get a numpy array out of the dataframe df.

    Args:
        df (DataFrame): Considered data frame.
        cols (string): The name of the columns that we want in numpy array format.

    Returns:
        nympay array: numpy array of the columns from our dataframe df.
    """
    np_array = df[cols].to_numpy()
    return np_array


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
    skf = StratifiedKFold(n_splits = n_s)
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
    # save the model to disk
    return pickle.dump(model, open(filename, 'wb'))


# --------------------------------------------------------------------------------------- #


def load_model(filename):
    # load the model from disk
    return pickle.load(open(filename, 'rb'))


# --------------------------------------------------------------------------------------- #