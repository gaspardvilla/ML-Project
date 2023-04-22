# Import the used libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import warnings

# Import sklearn librairies
from sklearn import *
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
from sklearn.multiclass import *
from sklearn.neighbors import *


# --------------------------------------------------------------------------------------- #


def get_model_features_selection(X, y, method, param = None, plot = False, seed = 0):
    """
    Select features according to a specific model

    Args :
        X, y : data to use for fitting the method of feature selection
        method : method to apply for feature selection ('lasso', 'lassoCV', 'PCA', 'recursive', 'recursiveCV')
        param : parameter of the method (depends on the model used for feature selection) (default=None)
        plot : True if you want to plot the corresponding graph for the method (default=False)
        seed : seed to apply to random_state (default = 0) 

    Returns :
        Model to use for feature selection : either lasso, lassoCV, PCA, recursive or recursiveCV
    """
    # avoid warning transform the y_train in y_test using .ravel()
    y_ravel = np.array(y).ravel()

    if method == "lasso":
        print("MESSAGE : param is the coefficient for regularization (positive float)")
        # define and fit the method
        lasso = Lasso(alpha = param, random_state = seed).fit(X, y_ravel)
        model = SelectFromModel(lasso, prefit = True)
        if plot:
            importance = np.abs(lasso.coef_)
            feature_names = np.array(X.columns)
            plt.bar(height=importance, x=feature_names)
            plt.title("Feature importances via coefficients")
            plt.show()

        # retrun the model
        return model

    elif method == "lassoCV":
        print("MESSAGE : param is the number of folds for cross validation (should be an int)")
        # define and fit the method
        lassoCV = LassoCV(cv = param, random_state = seed).fit(X, y_ravel)
        model = SelectFromModel(lassoCV, prefit = True)
        if plot:
            importance = np.abs(lassoCV.coef_)
            feature_names = np.array(X.columns)
            plt.bar(height=importance, x=feature_names)
            plt.title("Feature importances via coefficients")
            plt.show()

        # return the model
        return model

    elif method == "PCA":
        print('MESSAGE : If param > 1 PCA has a number of components equal to param.')
        print('MESSAGE : If param < 1 PCA select the best number of combonent in order to have an explained variance ratio equal to param')
        # define the method
        pca = PCA(n_components = param, random_state = seed).fit(X)
        if plot:
            pca = PCA()
            pca.fit(X)
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel('number of components')
            plt.ylabel('cumulative explained variance')

        # return the components
        return pca

    elif method == "recursive":
        print("MESSAGE : no param for this method")
        # define an estimator
        estimator = SVR(kernel="linear")
        # define and fit the method
        model = RFE(estimator, n_features_to_select = param).fit(X, y_ravel)
        if plot:
            print('Nothing to plot for this method. Try with method = recursiveCV')

        # return the model
        return model

    elif method == "recursiveCV":
        print("MESSAGE : param is the number of folds for cross validation (should be an int)")
        # define an estimator
        estimator = SVR(kernel = "linear") 
        # define and fit the method
        model = RFECV(estimator, cv = param).fit(X, y_ravel)
        if plot:
            cv = StratifiedKFold(param)
            visualizer = RFECV(estimator, cv = cv)
            # fit the data to the visualizer
            visualizer.fit(X, y)       
            visualizer.show() 

        # return the model
        return model

    elif method == 'forward selection':
        print('MESSAGE : param is the number of features that we want to keep in our model.')
        estimator = OneVsRestClassifier(LogisticRegression(max_iter = 1000, 
                                                class_weight = 'balanced', 
                                                multi_class='multinomial', 
                                                random_state=seed))
        if param != None:
            model = SequentialFeatureSelector(estimator, n_features_to_select = param).fit(X, y)
        else:
            raise ValueError("No value was given for the variable 'param' for \
                                the forward selection method")

        if plot:
            warnings.warn("No plot for the froward selection.")

        return model

    else:
        raise ValueError("Wrong method, it should be either: 'lasso', 'lassoCV', 'PCA', \
                                'recursive', 'recursiveCV' or 'forward selection'.")


# --------------------------------------------------------------------------------------- #


def get_model_MLR(seed = 0):
    """
    Select Multinomial Logistic Regression model and parameters you would like to tune by using evaluate_model function

    Args :
        seed : seed to apply to random_state (default = 0)

    Returns:
        Logistic Regression model and its parameters to tune
    """

    model = OneVsRestClassifier(LogisticRegression(max_iter = 1000, 
                                                    class_weight = 'balanced', 
                                                    multi_class='multinomial', 
                                                    solver='lbfgs', 
                                                    penalty='l2', 
                                                    random_state = seed))
    param = {'estimator__C':np.linspace(0, 20, num = 100)}
    
    return model, param


# --------------------------------------------------------------------------------------- #


def get_model_SVM(poly = False, seed = 0):
    
    """
    Select SVM model and parameters you would like to tune by using evaluate_model function

    Args :
        poly: True if you want to use the polynomial kernel in your SVM model (default = False)
        seed : seed to apply to random_state (default = 0)

    Returns :
        SVM model and its parameters to tune
    """

    if poly:
        param = {'estimator__C':np.linspace(1, 10, 50), 'estimator__degree':np.linspace(2, 5, 4, dtype = int)}
        model = OneVsRestClassifier(estimator=SVC(kernel='poly', 
                                                    decision_function_shape='ovr', 
                                                    class_weight='balanced', 
                                                    random_state = seed))
    else:
        param = {'estimator__C':np.linspace(1, 10, num=100), 'estimator__kernel':['linear', 'rbf', 'sigmoid']}
        model = OneVsRestClassifier(estimator=SVC(decision_function_shape='ovr', 
                                                        class_weight='balanced', 
                                                        random_state = seed))
    
    return model, param


# --------------------------------------------------------------------------------------- #


def get_model_RF(seed = 0):
    """
    Select RandomForest model and parameters to tune by using evaluate_model function

    Args:
        seed : seed to apply to random_state (default = 0)

    Returns:
        RandomForest model and the dictonnary of the hyperparameters to optimise with their scale
    """
    model = RandomForestClassifier(random_state = seed, 
                                    class_weight='balanced')

    param = {"n_estimators": np.linspace(200, 1000, 5, dtype = int),
            "min_samples_leaf": np.linspace(1, 4, 4, dtype = int),
            "max_depth": np.linspace(10, 50, 5, dtype = int),
            "min_samples_split": np.linspace(2, 10, 3, dtype = int)}
			  
    return model, param


# --------------------------------------------------------------------------------------- #


def get_model_MLP(seed = 0):
    """
    Select a neural network model and parameters to tune by using evaluate_model function

    Args : 
        seed : seed to apply to random_state (default = 0)

    Returns :
        MLP model and the dictonnary of the hyperparameters to optimise with their scale
    """
    model = MLPClassifier(hidden_layer_sizes = (100, 50, 50, 100), 
                            solver = 'sgd',
                            learning_rate = 'constant',
                            random_state = seed)

    param = {"activation" : ['tanh', 'relu'],
            "alpha": np.logspace(0, -4, 5),
            "learning_rate_init": np.logspace(-1, -5, 15)}
			  
    return model, param


# --------------------------------------------------------------------------------------- #


def get_model_KNN():
    """
    Returns:
        K-NN model and the dictonnary of the hyperparameters to optimise with their scale
    """
    model = KNeighborsClassifier()

    param = {'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'n_neighbors': np.linspace(4, 10, dtype = int),
            'weights': ['uniform', 'distance'],
            'leaf_size': np.linspace(10, 40, 4, dtype = int)}
    
    return model, param


# --------------------------------------------------------------------------------------- #


def get_list_models():
    """
    Return just the list of pre-trained models
    """
    return ['MLP', 'MLR', 'SVM', 'SVM_poly', 'RF']


# --------------------------------------------------------------------------------------- #


def best_model(name, seed = 0):
    """
    Return the best trained model
    """
    if name == 'hydro_MLP':
        model = MLPClassifier(hidden_layer_sizes = (100, 50, 50, 100), 
                                solver = 'sgd',
                                learning_rate = 'constant',
                                random_state = seed, #
                                activation = 'relu',
                                alpha = 0.001,
                                learning_rate_init = 0.0072)
    elif name == 'hydro_MLR':
        model = OneVsRestClassifier(LogisticRegression(max_iter = 1000, 
                                                        class_weight = 'balanced', 
                                                        multi_class='multinomial', 
                                                        solver='lbfgs', 
                                                        penalty='l2', 
                                                        random_state = seed, #
                                                        C = 3.43))
    elif name == 'hydro_SVM':
        model = OneVsRestClassifier(estimator=SVC(decision_function_shape='ovr', 
                                                            class_weight='balanced', 
                                                            random_state = seed, #
                                                            C = 10,
                                                            kernel = 'rbf'))
    elif name == 'hydro_SVM_poly':
        model = OneVsRestClassifier(estimator=SVC(kernel='poly', 
                                                    decision_function_shape='ovr', 
                                                    class_weight='balanced', 
                                                    random_state = seed, #
                                                    C = 9.63, 
                                                    degree = 3))
    elif name == 'hydro_RF':
        model = RandomForestClassifier(random_state = seed, 
                                        class_weight='balanced', #
                                        max_depth = 20,
                                        min_samples_leaf = 1,
                                        min_samples_split = 2,
                                        n_estimators = 400)
    elif name == 'riming_MLP':
        model = MLPClassifier(hidden_layer_sizes = (100, 50, 50, 100), 
                                solver = 'sgd',
                                learning_rate = 'constant',
                                random_state = seed, #
                                activation = 'tanh',
                                alpha = 0.01,
                                learning_rate_init = 0.0518)
    elif name == 'riming_MLR':
        model = OneVsRestClassifier(LogisticRegression(max_iter = 1000, 
                                                        class_weight = 'balanced', 
                                                        multi_class='multinomial', 
                                                        solver='lbfgs', 
                                                        penalty='l2', 
                                                        random_state = seed, #
                                                        C = 16.36))
    elif name == 'riming_SVM':
        model = OneVsRestClassifier(estimator=SVC(decision_function_shape='ovr', 
                                                        class_weight='balanced', 
                                                        random_state = seed, #
                                                        C = 9.09,
                                                        kernel = 'rbf'))
    elif name == 'riming_SVM_poly':
        model = OneVsRestClassifier(estimator=SVC(kernel='poly', 
                                                    decision_function_shape='ovr', 
                                                    class_weight='balanced', 
                                                    random_state = seed, #
                                                    C = 10, 
                                                    degree = 3))
    elif name == 'riming_RF':
        model = RandomForestClassifier(random_state = seed, 
                                        class_weight='balanced', #
                                        max_depth = 30,
                                        min_samples_leaf = 1,
                                        min_samples_split = 2,
                                        n_estimators = 400)
    else:
        raise ValueError('Wrong name')
    return model

