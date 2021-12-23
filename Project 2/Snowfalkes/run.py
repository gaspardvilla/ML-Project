# Import other libraries
from IPython.display import display
import numpy as np
import pandas as pd
import pyarrow
import os

# Import sklearn tools
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn import *
from sklearn.pipeline import *
from sklearn.preprocessing import *
from sklearn.ensemble import *
from sklearn.svm import *
from sklearn.preprocessing import * 

# Import files
from helpers import *
from cross_validation import *
from models import *
from dataloader import load_data_sets
from dataprocess import processing


# load the dataset
classifier = 'hydro'
data_set, classes = load_data_sets(classifier = classifier)
X, y = processing(data_set, classes, classifier)

# set a seed for all the random states
s = 0

# set the number of kfold
k_fold = 5

# get a train and test set for modelization
X_train, y_train, X_test, y_test = split_data(X, y, kfold = k_fold, seed = s)

# select features
method = 'lassoCV'
model_selec = get_model_features_selection(X_train, y_train, method, 5, seed = s)

# reduce the dataset with the selected features
X_train_selec = feature_transform(model_selec, X_train, method)
X_test_selec = feature_transform(model_selec, X_test, method)

# data augmentation with SMOTE
X_train_selec, y_train = smote_data_augmentation(X_train_selec, y_train)

# SVM model
svm, param = get_model_SVM(poly = True, seed = s)
cv_SVM_poly = evaluate_model(svm, param, X_train_selec, y_train, X_test_selec, y_test, verbosity = 2)

# Save best model
path = 'Models/trained_model/'+str(classifier)+'_SVM_poly.pkl'
save_model(path, cv_SVM_poly)