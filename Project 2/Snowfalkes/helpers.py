# Import the used libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
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


def test_model(X_train, y_train, X_test, y_test, method, class_acc = True):

    y_train = np.array(y_train).ravel()

    if method == 'logisitic regression':
        model = LogisticRegressionCV(cv=5,  penalty='l1', solver='saga', max_iter=100, class_weight='balanced').fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        print('The accuracy for the train set with  this model is : ', accuracy_score(y_train, y_train_pred))
        print('The accuracy for the test set with this model is : ', accuracy_score(y_test, y_pred))
        if class_acc:
            print(classification_accuracy(y_test, y_pred))
        return y_pred
    
    elif method == 'SVM':
        model = make_pipeline(StandardScaler(), SVC(gamma='auto')).fit(X_train,y_train)
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        print('The accuracy for the train set with  this model is : ', accuracy_score(y_train, y_train_pred))
        print('The accuracy for the test set with this model is : ', accuracy_score(y_test, y_pred))
        if class_acc:
            print(classification_accuracy(y_test, y_pred))
        return y_pred

    elif method == 'random forest':
        model = RandomForestClassifier(n_estimators=1000, max_depth= 200, min_samples_leaf=5, class_weight = 'balanced').fit(X_train,y_train)
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        print('The accuracy for the train set with  this model is : ', accuracy_score(y_train, y_train_pred))
        print('The accuracy for the test set with  this model is : ', accuracy_score(y_test, y_pred))
        if class_acc:
            print(classification_accuracy(y_test, y_pred))
        return model

    elif method == 'gradient boosting':
        model = GradientBoostingClassifier(learning_rate=0.1).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        print('The accuracy for the train set with  this model is : ', accuracy_score(y_train, y_train_pred))
        print('The accuracy for the test set with this model is : ', accuracy_score(y_test, y_pred))
        if class_acc:
            print(classification_accuracy(y_test, y_pred))
        return y_pred

    elif method == 'feed forward neural network':
        model =MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        print('The accuracy for the train set with  this model is : ', accuracy_score(y_train, y_train_pred))
        print('The accuracy for the test set with this model is : ', accuracy_score(y_test, y_pred))
        if class_acc:
            print(classification_accuracy(y_test, y_pred))
        return y_pred
    else:
        raise ValueError("Wrong method, it should be either: 'logisitic regression', 'SVM', 'random forest', 'gradient boosting' or 'feed forward neural network'.")
    

# --------------------------------------------------------------------------------------- #


def features_selection (X, y, method, param = None, plot = False):
    return None

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
        return model

    elif method == "lassoCV":
        print("param = number of folds for cross validation (should be an int)")
        # define and fit the method
        lassoCV = LassoCV(cv = param).fit(X, y)
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
        model = PCA(n_components = param)
        # transform the data
        components = model.fit(X)
        if plot == True:
            pca = PCA()
            pca.fit(X)
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel('number of components')
            plt.ylabel('cumulative explained variance')
        return components

    elif method == "recursive":
        print("no param for this method")
        # define an estimator
        estimator = SVR(kernel="linear")
        # define and fit the method
        model = RFE(estimator, n_features_to_select=param).fit(X, y)
        if plot == True:
            print('Nothing to plot for this method. Try with method = recursiveCV')
        # return the model
        return model

    elif method == "recursiveCV":
        print("param = number of folds for cross validation (should be an int)")
        # define an estimator
        estimator = SVR(kernel = "linear") # we can try with other estimator functions such as GradientBoostingClassifier(), RandomForestClassifier(),...
        # define and fit the method
        model = RFECV(estimator, cv = param).fit(X, y)
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


def get_models_LR(ovr = False):
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

    Return the accuracy score for the tuned model
    """
    clf = GridSearchCV(model, param, verbose=1).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.best_params_)


    return accuracy_score(y_test, y_pred)


# --------------------------------------------------------------------------------------- #
