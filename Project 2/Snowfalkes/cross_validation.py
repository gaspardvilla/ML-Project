# Import the used libraries
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.pipeline import Pipeline


# Import sklearn librairies
from sklearn import *
from sklearn.metrics import *
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import *


# Import functions accuracy
from helpers import *
from dataloader import *
from dataprocess import *
from models import *


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


def all_models_train_and_test():
    """
    This funcion computes all the results displayed on the 
    report. 

    Returns:
        [dict]: Return train and test error dictionnaries
    """

    
    k_fold = 5
    seed = 0

    data_set, classes = load_data_sets(classifier = 'riming')
    X_rim, y_rim = processing(data_set, classes, 'riming')

    data_set, classes = load_data_sets(classifier = 'hydro')
    X_hyd, y_hyd = processing(data_set, classes, 'hydro')


    X_dict = {
        'riming': X_rim,
        'hydro': X_hyd
        }
    y_dict = {
        'riming': y_rim,
        'hydro': y_hyd
        }


    models_name = get_list_models()
    method = 'lassoCV'
    skf = StratifiedKFold(n_splits = k_fold, 
                            shuffle = True, 
                            random_state = seed)

    train = {
            'riming': {
                        'MLP': np.array([]), 
                        'MLR': np.array([]), 
                        'SVM': np.array([]), 
                        'SVM_poly': np.array([]),
                        'RF': np.array([])
            },
            'hydro': {
                        'MLP': np.array([]), 
                        'MLR': np.array([]), 
                        'SVM': np.array([]), 
                        'SVM_poly': np.array([]),
                        'RF': np.array([])
            }
    }

    test = {
            'riming': {
                        'MLP': np.array([]), 
                        'MLR': np.array([]), 
                        'SVM': np.array([]), 
                        'SVM_poly': np.array([]),
                        'RF': np.array([])
            },
            'hydro': {
                        'MLP': np.array([]), 
                        'MLR': np.array([]), 
                        'SVM': np.array([]), 
                        'SVM_poly': np.array([]),
                        'RF': np.array([])
            }
    }

    for classifier_ in ['riming', 'hydro']:

        for model_name in models_name:

            name = classifier_ + '_' + str(model_name)

            X = X_dict[classifier_]
            y = y_dict[classifier_]
            
            for train_idx, test_idx in skf.split(X, y):
                model = best_model(name)

                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

                model_feat_selec = get_model_features_selection(X_train, 
                                                                y_train, 
                                                                method, 
                                                                k_fold, 
                                                                seed = seed)

                X_train_reduce = feature_transform(model_feat_selec, X_train, method)
                X_test_reduce = feature_transform(model_feat_selec, X_test, method)

                model.fit(X_train_reduce, y_train)

                y_train_pred = model.predict(X_train_reduce)
                y_test_pred = model.predict(X_test_reduce)

                score_train = accuracy_score(y_train, y_train_pred)
                score_test = accuracy_score(y_test, y_test_pred)

                train[classifier_][model_name] = np.append(train[classifier_][model_name], score_train)
                test[classifier_][model_name] = np.append(test[classifier_][model_name], score_test)
        
    for classifier_ in ['riming', 'hydro']:

        for model_name in models_name:

            print(classifier_+'_'+model_name, ' train mean = ', np.mean(train[classifier_][model_name]))
            print(classifier_+'_'+model_name, ' train std = ', np.std(train[classifier_][model_name]))
            
            print(classifier_+'_'+model_name, ' test mean = ', np.mean(test[classifier_][model_name]))
            print(classifier_+'_'+model_name, ' test std = ', np.std(test[classifier_][model_name]))

    return train, test
