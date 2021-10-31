# Import everything we need
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *
from losses import *
from plots import *
from EDA import *
from cross_validation import *
import os

# -------------------------------------------------------------------------- #

# Load the train set data set
DATA_TRAIN_PATH =  "../data/train.csv"
data_y, data_set, ids = load_csv_data(DATA_TRAIN_PATH, sub_sample=True)

# Classification of the output
y_0, y_1, y_2, y_3 = y_classification(data_y, data_set)

#EDA for each class
class_0, class_1, class_2, class_3 = EDA_class(data_set)

# -------------------------------------------------------------------------- #

max_degree = 10

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #


# Class 1
# -------------------------------------------------------------------------- #

# Setting parameters
neg_log = Neg_log()
print('Class 1')
param_ridge_1 = Parameters()
param_ridge_1.set_degree(max_degree)
param_ridge_1.set_loss_fct(neg_log)
param_ridge_1.set_method(ridge_regression)
param_ridge_1.set_lambda_range(np.logspace(-6,-1,30))
param_ridge_1.set_to_test(['lambda'])
param_ridge_1.set_viz(False)
param_ridge_1.set_use_backward_selection(True)
param_ridge_1.set_use_forward_selection(True)
param_ridge_1.set_use_interactions(True)
# Cross validation
param_ridge_1 = cross_validation_poly_gas(y_2, class_2, param_ridge_1)

print(param_ridge_1.feature_list)
print(param_ridge_1.polynomial_selection)
print(param_ridge_1.best_error)
print(param_ridge_1.kept_interactions)

# Building models
class_2_ = build_polynomial_features(class_2, param_ridge_1)

# Print the final parameters
print(param_ridge_1.best_gamma)
print(param_ridge_1.gamma)