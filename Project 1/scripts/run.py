# Import everything we need
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from methods import *
from losses import *
from plots import *
from EDA import *
from cross_validation import *
import os

# -------------------------------------------------------------------------- #

# Load the train set data set
DATA_TRAIN_PATH =  "../data/train.csv"
data_y, data_set, ids = load_csv_data(DATA_TRAIN_PATH, sub_sample=False)

# Classification of the output
y_0, y_1, y_2, y_3 = y_classification(data_y, data_set)

#EDA for each class
class_0, class_1, class_2, class_3 = EDA_class(data_set)

# Load the train set data set
DATA_TRAIN_PATH =  "../data/train.csv"
data_y_tr, data_set_tr, ids = load_csv_data(DATA_TRAIN_PATH, sub_sample=False)

# Classification of the output
y_0_tr, y_1_tr, y_2_tr, y_3_tr = y_classification(data_y_tr, data_set_tr)

#EDA for each class
class_0_tr, class_1_tr, class_2_tr, class_3_tr = EDA_class(data_set_tr)

# Load the test_set
DATA_TEST_PATH = "../data/test.csv"
_, data_test_set, ids_test = load_csv_data(DATA_TEST_PATH)

# EDA for each class
class_0_test, class_1_test, class_2_test, class_3_test = EDA_class(data_test_set)

# -------------------------------------------------------------------------- #

# Cross validation and building model

# Initialization of some parameters
max_degree = 6

# Class 0
# -------------------------------------------------------------------------- #

# Setting parameters
print('Class 0')
param_ridge_0 = Parameters()
param_ridge_0.set_degree(max_degree)
param_ridge_0.set_method(ridge_regression_)
param_ridge_0.set_to_test(['lambda'])
param_ridge_0.set_viz(False)
param_ridge_0.set_use_backward_selection(True)
param_ridge_0.set_use_interactions(True)
# Cross validation
param_ridge_0 = cross_validation_poly_gas(y_0, class_0, param_ridge_0)

# Building models
class_0_ = build_polynomial_features(class_0_tr, param_ridge_0)
class_0_test_ = build_polynomial_features(class_0_test, param_ridge_0)

# See if it is necessary
# param_ridge_0 = cross_validation(y_0_tr, class_0_, param_ridge_0)

print(param_ridge_0.feature_list)
print(param_ridge_0.polynomial_selection)
print(param_ridge_0.best_error)
print(param_ridge_0.kept_interactions)

# Train and get the prediction
loss_0, w_0 = param_ridge_0.method(y_0_tr, class_0_, param_ridge_0)
y_pred_0 = predict_labels(w_0, class_0_test_)



# Class 1
# -------------------------------------------------------------------------- #

# Setting parameters
neg_log = Neg_log()
mae = MAE()
print('Class 1')
param_ridge_1 = Parameters()
param_ridge_1.set_degree(max_degree)
param_ridge_1.set_loss_fct(neg_log)
param_ridge_1.set_method(ridge_regression_)
param_ridge_1.set_loss_fct(neg_log)
param_ridge_1.set_to_test(['lambda'])
param_ridge_1.set_lambda_range(np.logspace(-5,-1,30))
param_ridge_1.set_viz(False)
param_ridge_1.set_use_backward_selection(True)
param_ridge_1.set_use_interactions(True)
# Cross validation
param_ridge_1 = cross_validation_poly_gas(y_1, class_1, param_ridge_1)

# Building models
class_1_ = build_polynomial_features(class_1_tr, param_ridge_1)
class_1_test_ = build_polynomial_features(class_1_test, param_ridge_1)

# See if it is necessary
# param_ridge_1 = cross_validation(y_1_tr, class_1_, param_ridge_1)

print(param_ridge_1.feature_list)
print(param_ridge_1.polynomial_selection)
print(param_ridge_1.best_error)
print(param_ridge_1.kept_interactions)

# Train and get the prediction
loss_1, w_1 = param_ridge_1.method(y_1_tr, class_1_, param_ridge_1)
y_pred_1 = predict_labels(w_1, class_1_test_)



# Class 2
# -------------------------------------------------------------------------- #

# Setting parameters
print('Class 2')
param_ridge_2 = Parameters()
param_ridge_2.set_degree(max_degree)
param_ridge_2.set_method(ridge_regression_)
param_ridge_2.set_loss_fct(neg_log)
param_ridge_2.set_to_test(['lambda'])
param_ridge_2.set_lambda_range(np.logspace(-6,-1,30))
param_ridge_2.set_viz(False)
param_ridge_2.set_use_backward_selection(True)
param_ridge_2.set_use_forward_selection(True)
param_ridge_2.set_use_interactions(True)
# Cross validation
param_ridge_2 = cross_validation_poly_gas(y_2, class_2, param_ridge_2)

# Building models
class_2_ = build_polynomial_features(class_2_tr, param_ridge_2)
class_2_test_ = build_polynomial_features(class_2_test, param_ridge_2)

# See if it is necessary
# param_ridge_2 = cross_validation(y_2_tr, class_2_, param_ridge_2)

print(param_ridge_2.feature_list)
print(param_ridge_2.polynomial_selection)
print(param_ridge_2.best_error)
print(param_ridge_2.kept_interactions)

# Train and get the prediction
loss_2, w_2 = param_ridge_2.method(y_2_tr, class_2_, param_ridge_2)
y_pred_2 = predict_labels(w_2, class_2_test_)



# Class 3
# -------------------------------------------------------------------------- #

# Setting parameters
print('Class 3')
param_ridge_3 = Parameters()
param_ridge_3.set_degree(max_degree)
param_ridge_3.set_method(ridge_regression_)
param_ridge_3.set_loss_fct(neg_log)
param_ridge_3.set_to_test(['lambda'])
param_ridge_3.set_lambda_range(np.logspace(-4, 0, 30)) #-4 0
param_ridge_3.set_viz(False)
param_ridge_3.set_use_backward_selection(True)
param_ridge_3.set_use_forward_selection(True)
param_ridge_3.set_use_interactions(True)
# Cross validation
param_ridge_3 = cross_validation_poly_gas(y_3, class_3, param_ridge_3)

# Building models
class_3_ = build_polynomial_features(class_3_tr, param_ridge_3)
class_3_test_ = build_polynomial_features(class_3_test, param_ridge_3)

# See if it is necessary
# param_ridge_3 = cross_validation(y_3_tr, class_3_, param_ridge_3)

print(param_ridge_3.feature_list)
print(param_ridge_3.polynomial_selection)
print(param_ridge_3.best_error)
print(param_ridge_3.kept_interactions)

# Train and get the prediction
loss_3, w_3 = param_ridge_3.method(y_3_tr, class_3_, param_ridge_3)
y_pred_3 = predict_labels(w_3, class_3_test_)



# Submission
# -------------------------------------------------------------------------- #

y_pred = rebuild_y(y_pred_0, y_pred_1, y_pred_2, y_pred_3, data_test_set)
OUTPUT_PATH = '../data/test_prediction_submission.csv' # TODO: fill in desired name of output file for submission
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

