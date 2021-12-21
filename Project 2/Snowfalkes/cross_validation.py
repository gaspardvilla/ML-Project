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
    scoring = {"Accuracy": make_scorer(accuracy_score)}
    clf = GridSearchCV(model, param, scoring=scoring, refit="Accuracy", return_train_score=True, verbose=1).fit(X_train, y_train_ravel)

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

def plot_cv_results(cv):
    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV evaluating using accuracy scorers", fontsize=16)

    plt.xlabel('estimator_C')
    plt.ylabel("Score")

    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0.73, 1)

    # Get the regular numpy array from the MaskedArray
    results = cv.cv_results_
    scoring = 'Accuracy'
    X_axis = np.array(results['param_estimator__C'].data, dtype=float)

    for sample, style in (("train", "--"), ("test", "-")):
        sample_score_mean = results["mean_%s_%s" % (sample, scoring)]
        sample_score_std = results["std_%s_%s" % (sample, scoring)]
        ax.fill_between(
            X_axis,
            sample_score_mean - sample_score_std,
            sample_score_mean + sample_score_std,
            alpha=0.1 if sample == "test" else 0,
            )
        ax.plot(
            X_axis,
            sample_score_mean,
            style,
            alpha=1 if sample == "test" else 0.7,
            label="%s (%s)" % (scoring, sample),
            )

    best_index = np.nonzero(results["rank_test_%s" % scoring] == 1)[0][0]
    best_score = results["mean_test_%s" % scoring][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot(
        [
            X_axis[best_index],
        ]
        * 2,
        [0, best_score],
        linestyle="-.",
        marker="x",
        markeredgewidth=3,
        ms=8,
    )

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score, (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid(False)
    plt.show()