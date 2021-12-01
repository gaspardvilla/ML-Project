# Import the used libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Import sklearn librairies
from sklearn.feature_selection import *
from sklearn.model_selection import *
from sklearn.linear_model import *
from yellowbrick.model_selection import *
from sklearn.svm import *
from sklearn.decomposition import *



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
        # Create the input
        # Add for each cam
        # cam0
        classified_data = self.get_sub_data_cam(classifier, 0, data_set.cam0)
        
        #cam1
        classified_data = pd.concat([classified_data, self.get_sub_data_cam(classifier, 1, data_set.cam1)])
        
        #cam2
        classified_data = pd.concat([classified_data, self.get_sub_data_cam(classifier, 2, data_set.cam2)])

        # Return the concatenated data frame that contains all the data point to consider
        return classified_data

    
    def get_classses(self, classifier, data):
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



def cross_validation_method():
    return None



# --------------------------------------------------------------------------------------- #



def test_model():
    return None



# --------------------------------------------------------------------------------------- #



def features_selection (X, y, method, param, plot = False):

    if method == "lasso":
        # define and fit the method
        lasso = Lasso(alpha = param).fit(X, y)
        model = SelectFromModel(lasso, prefit = True)
        if plot == True:
            importance = np.abs(lasso.coef_)
            feature_names = np.array(X.columns)
            plt.bar(height=importance, x=feature_names)
            plt.title("Feature importances via coefficients")
            plt.show()
        return model.transform(X)

    elif method == "lassoCV":
        # define and fit the method
        lassoCV = LassoCV(cv = param).fit(X, y)
        model = SelectFromModel(lassoCV, prefit = True)
        if plot == True:
            importance = np.abs(lassoCV.coef_)
            feature_names = np.array(X.columns)
            plt.bar(height=importance, x=feature_names)
            plt.title("Feature importances via coefficients")
            plt.show()
        return model.transform(X)
        # transform the data
        return model.transform(X)

    elif method == "PCA":
        print('If param > 1 PCA has a number of components equal to param.')
        print('If param < 1 PCA select the best number of combonent in order to have an explained variance ratio equal to param')
        # define the method
        model = PCA(n_components = param)
        # transform the data
        components = model.fit_transform(X)
        if plot == True:
            pca = PCA()
            pca.fit(X)
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel('number of components')
            plt.ylabel('cumulative explained variance')
        return components

    elif method == "recursive":
        # define an estimator
        estimator = SVR(kernel="linear")
        # define and fit the method
        model = RFE(estimator, n_features_to_select=param)
        if plot == True:
            print('TODO')
        # transform the data
        return model.fit_transform(X, y)

    elif method == "recursiveCV":
        # define an estimator
        estimator = SVR(kernel = "linear")
        # define and fit the method
        model = RFECV(estimator, cv = param).fit(X, y)
        if plot == True:
            cv = StratifiedKFold(param)
            visualizer = RFECV(estimator, cv=cv)
            visualizer.fit(X, y)        # Fit the data to the visualizer
            visualizer.show() 
        # transform the data
        return model.transform(X)

    else:
        raise ValueError("Wrong method, it should be either: 'lasso', 'lassoCV', 'PCA', 'recursive' or 'recursiveCV'.")