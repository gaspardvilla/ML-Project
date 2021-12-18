# Import the used libraries
from helpers import *
from models import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from IPython.display import display
from sklearn.pipeline import Pipeline

from sklearn.datasets import *
from sklearn.ensemble import *
from sklearn.experimental import *
from sklearn.model_selection import *

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
from sklearn.metrics import *



# --------------------------------------------------------------------------------------- #


def evaluate_model(model, param, X_train, y_train, X_test, y_test):
    """
    Grid Search for the model to select the best parameters
    Evaluation of a model 

    Args:
        model : the model used for Grid Search and to evaluate 
        param : the parameters to tune during Grid Search
        X_train : data training set
        y_train : target to reach during the train
        X_test : data testing set
        y_test : target to reach during the test 

    Return the best model
    """
    #Avoid warning transform the y_train in y_test using .ravel()
    y_train_ravel = np.array(y_train).ravel()

    #Grid Search to tune the parameters
    clf = GridSearchCV(model, param, verbose=2).fit(X_train, y_train_ravel)

    #Predict using the best fitted model on the train set to verify we avoid overfitting
    y_pred_train = clf.predict(X_train)

    #Compute the total accuracy on the training set
    print('Accuracy score on the training set:')
    print(accuracy_score(y_train, y_pred_train))
    
    #Compute the accuracy for each class on the training set
    print('Accuracy for each class on the training set:')
    classification_accuracy(y_train, y_pred_train)

    #Predict using the best fitted model on the test set
    y_pred = clf.predict(X_test)
    print('Best parameters for the fitted model:')
    print(clf.best_params_)

    #Compute the total accuracy on the testing set
    print('Accuracy score on the testing set:')
    print(accuracy_score(y_test, y_pred))
    
    #Compute the accuracy for each class on the testing set
    print('Accuracy for each class on the testing set:')
    classification_accuracy(y_test, y_pred)
    
    return clf

# --------------------------------------------------------------------------------------- #