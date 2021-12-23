# Import the used libraries
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.pipeline import Pipeline


# Import sklearn librairies
from sklearn.metrics import *
from sklearn.model_selection import *


# --------------------------------------------------------------------------------------- #


def evaluate_model(model, param, X_train, y_train, X_test, y_test, verbosity = 0):
    """
    Grid Search for the model to select the best parameters
    Evaluation of a model 

    Args :
        model : the model used for Grid Search and to evaluate 
        param : the parameters to tune during Grid Search
        X_train : data training set
        y_train : target to reach during the train
        X_test : data testing set
        y_test : target to reach during the test 
        verbosity :

    Returns :
        Best model containing the best hyperparameters for the model
    """
    #Avoid warning transform the y_train in y_test using .ravel()
    y_train_ravel = np.array(y_train).ravel()

    #Grid Search to tune the parameters
    scoring = {'Accuracy': make_scorer(accuracy_score)}
    clf = GridSearchCV(model, param, scoring = scoring, refit = 'Accuracy', return_train_score = True, verbose = verbosity).fit(X_train, y_train_ravel)

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

