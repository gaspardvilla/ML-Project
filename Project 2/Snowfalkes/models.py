from helpers import *

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


# --------------------------------------------------------------------------------------- #


def get_model_features_selection (X, y, method, param = None, plot = False):
    """
    Select features according to a specific model

    Args:
        X, y : data to use for fitting the model of feature selection
        param : parameter of the model (depends on the model used for feature selection)
        plot : True if you want to plot the corresponding graph of your model selected

    Return a model to use for feature selection : either lasso, lassoCV, PCA, recursive or recursiveCV
    """
    #Avoid warning transform the y_train in y_test using .ravel()
    y_ravel = np.array(y).ravel()

    if method == "lasso":
        # define and fit the method
        lasso = Lasso(alpha = param).fit(X, y_ravel)
        model = SelectFromModel(lasso, prefit = True)
        if plot == True:
            importance = np.abs(lasso.coef_)
            feature_names = np.array(X.columns)
            plt.bar(height=importance, x=feature_names)
            plt.title("Feature importances via coefficients")
            plt.show()
        return model

    elif method == "lassoCV":
        print("param = number of folds for cross validation (should be an int)")
        # define and fit the method
        lassoCV = LassoCV(cv = param).fit(X, y_ravel)
        model = SelectFromModel(lassoCV, prefit = True)
        if plot == True:
            importance = np.abs(lassoCV.coef_)
            feature_names = np.array(X.columns)
            plt.bar(height=importance, x=feature_names)
            plt.title("Feature importances via coefficients")
            plt.show()
        return model

    elif method == "PCA":
        print('If param > 1 PCA has a number of components equal to param.')
        print('If param < 1 PCA select the best number of combonent in order to have an explained variance ratio equal to param')
        # define the method
        PCA = PCA(n_components = param, feature_names_in_ = X.columns).fit(X)
        # transform the data
        model = SelectFromModel(PCA, prefit = True)
        if plot == True:
            pca = PCA()
            pca.fit(X)
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel('number of components')
            plt.ylabel('cumulative explained variance')
        return model

    elif method == "recursive":
        print("no param for this method")
        # define an estimator
        estimator = SVR(kernel="linear")
        # define and fit the method
        model = RFE(estimator, n_features_to_select=param).fit(X, y_ravel)
        if plot == True:
            print('Nothing to plot for this method. Try with method = recursiveCV')
        # return the model
        return model

    elif method == "recursiveCV":
        print("param = number of folds for cross validation (should be an int)")
        # define an estimator
        estimator = SVR(kernel = "linear") # we can try with other estimator functions such as GradientBoostingClassifier(), RandomForestClassifier(),...
        # define and fit the method
        model = RFECV(estimator, cv = param).fit(X, y_ravel)
        if plot == True:
            cv = StratifiedKFold(param)
            visualizer = RFECV(estimator, cv=cv)
            visualizer.fit(X, y)        # Fit the data to the visualizer
            visualizer.show() 
        # return the model
        return model

    else:
        raise ValueError("Wrong method, it should be either: 'lasso', 'lassoCV', 'PCA', 'recursive' or 'recursiveCV'.")


# --------------------------------------------------------------------------------------- #


def get_model_LR(ovr = False):
    """
    Select Logistic Regression model and parameters you would like to tune by using evaluate_model function

    Args:
        ovr (One Versus the Rest): True if you want to use the OneVSRestClassifier

    Return the Logistic Regression model and its parameters to tune
    """

    if ovr == True :
        model = OneVsRestClassifier(LogisticRegression(max_iter = 1000, class_weight = 'balanced', multi_class='multinomial', solver='lbfgs', penalty='none', random_state=0))
        param = {'estimator__penalty':['none', 'l1','l2'], 
				 'estimator__C':np.linspace(0.1, 0.11, num=10)}
    else :
        model = LogisticRegression(max_iter = 1000, class_weight = 'balanced', multi_class='multinomial', solver='lbfgs', penalty='none', random_state=0)
        param = {'penalty':['none', 'l1', 'l2'], 'C':np.linspace(0.1, 1, num=10)}
    
    return model, param


# --------------------------------------------------------------------------------------- #


def get_model_SVM(poly = False):
    
    """
    Select SVM model and parameters you would like to tune by using evaluate_model function

    Args:
        poly: True if you want to use the polynomial kernel in your SVM model

    Return the SVM model and its parameters to tune
    """

    if poly == True:
        param = {'estimator__C':np.linspace(1, 10, num=10), 'estimator__degree':np.linspace(0, 5, dtype = int)}
        model = OneVsRestClassifier(estimator=SVC(kernel='poly', decision_function_shape='ovr', class_weight='balanced', random_state=0))
    else:
        param = {'estimator__C':np.linspace(1, 10, num=10), 'estimator__kernel':['linear', 'rbf', 'sigmoid']}
        model = OneVsRestClassifier(estimator=SVC(decision_function_shape='ovr', class_weight='balanced', random_state=0))
    return model, param


# --------------------------------------------------------------------------------------- #


def get_model_RF():
    """
    Select RandomForest model and parameters to tune by using evaluate_model function

    Returns:
        The RandomForest model and the dictonnary of the hyperparameters to optimise with their scale
    """
    model = RandomForestClassifier(random_state=0, class_weight='balanced')

    param = {"n_estimators": np.linspace(200,2000,10, dtype=int),
            "min_samples_leaf": np.linspace(1,4,4, dtype=int),
            "max_depth": np.linspace(10,100,10, dtype=int),
            "min_samples_split": np.linspace(2,10,3,dtype=int)}
			  
    return model, param


# --------------------------------------------------------------------------------------- #


def get_model_MLP():
    """
    Select a neural network model and parameters to tune by using evaluate_model function

    Returns:
        The MLP model and the dictonnary of the hyperparameters to optimise with their scale
    """
    model = MLPClassifier(hidden_layer_sizes = (100,50,50,100), random_state=0)

    param = {"activation" : ['tanh', 'relu'],
            "solver": ['sgd', 'adam'],
            "alpha": np.logspace(0,-4,5),
            "learning_rate_init": np.logspace(-1,-5,10),
            "learning_rate": ['constant', 'adaptive']}
			  
    return model, param


# --------------------------------------------------------------------------------------- #