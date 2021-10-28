import math
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *
from EDA import *
from losses import *

def method_evaluation(y, data_set, parameters, k_indices, k):
    """return the loss of the method"""
    loss_tr = [] #to save the training loss for each training set
    loss_te = [] #to save the testint loss for each test set

    # get k'th subgroup in test, others in train: 
    te_idx = k_indices[k] # takes the indices of the data that corresponds to the k'th subgroup
    tr_idx = k_indices[~(np.arange(k_indices.shape[0]) == k)] # select the subgroups that are not 
                                                              # the test one and put them in train
                                                              # np.arange is for creating an array 
                                                              # containing the indices of the data 
                                                              # that are for the train
    tr_idx = tr_idx.reshape(-1) # put everything in a list

    x_tr = data_set[tr_idx]
    x_te = data_set[te_idx]
    y_tr = y[tr_idx]
    y_te = y[te_idx]
    
    # ridge regression:
    loss_tr_i, w = parameters.method(y_tr, x_tr, parameters)

    # Calculate the accuracy for the train and test set
    y_pred_tr = predict_labels(w, x_tr)
    nb_errors_tr, percentage_error_tr = counting_errors(y_pred_tr, y_tr)
    y_pred_te = predict_labels(w, x_te)
    nb_errors_te, percentage_error_te = counting_errors(y_pred_te, y_te)

    loss_tr.append(percentage_error_tr)
    loss_te.append(percentage_error_te)
    
    # calculate the loss for train and test data:       
    # loss_tr.append(loss_tr_i)
    # loss_te.append(parameters.loss_fct.cost(y_te, x_te, w))

    return loss_tr, loss_te
    
# -------------------------------------------------------------------------- #

def classic_cv(y_, class_, parameters, idx):
    seed = parameters.seeds[0]

    # split data in k fold
    k_indices = build_k_indices(y_, parameters.k_fold, seed)
    # define lists to store the loss of training data and test data
    loss_te = []
    loss_tr = []

    for param in parameters.range(idx):
        parameters.set_param(idx, param)

        for k in range(parameters.k_fold):
            # cross validation:
            loss_tr_i, loss_te_i = method_evaluation(y_, class_, parameters, k_indices, k)

        loss_tr.append(np.mean(loss_tr_i))
        loss_te.append(np.mean(loss_te_i))
    
    best_param = parameters.range(idx)[np.argmin(loss_te)]
    parameters.set_best_param(idx, best_param)
    parameters.set_param(idx, best_param)

    # Display the results
    min_test_error = np.min(loss_te)
    print('Test error: ' +str(min_test_error)+ '\nBest ' \
        +str(parameters.names[idx-1])+ ': ' +str(parameters.best_param(idx)))

    # Visualization
    if parameters.viz:
        cross_validation_visualization(parameters.range(idx), loss_tr, loss_te)
    
    return parameters

# -------------------------------------------------------------------------- #

def cross_validation_1_param(y_class, data_class, parameters):
    # Set the best parameter
    parameters = classic_cv(y_class, data_class, parameters, 1)
    
    # Return the optimal parameters for the considered method and loss function
    return parameters

# -------------------------------------------------------------------------- #

def cross_validation_2_param(y_class, data_class, parameters):
    # Set the best parameters for the 1st parameter
    parameters = classic_cv(y_class, data_class, parameters, 1)

    # Set the best parameters for the 2nd parameter using the 1st one found before
    parameters = classic_cv(y_class, data_class, parameters, 2)

    # Return the optimal parameters for the considered method and loss function
    return parameters

# -------------------------------------------------------------------------- #

def cross_validation(y_class, data_class, parameters):
    # Initialization of the optimal paramters

    if parameters.nb_to_test == 2:
        # Find the two optimal parameters
        parameters = cross_validation_2_param(y_class, data_class, parameters)

    elif parameters.nb_to_test == 1:
        # Find the optimal parameter
        parameters = cross_validation_1_param(y_class, data_class, parameters)

    elif parameters.nb_to_test == 0:
        print('No parameter to optimize for this method and this loss function')

    # Return the optimal parameters for the considered method and loss function
    return parameters

# -------------------------------------------------------------------------- #

def build_poly(data_set, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # polynomial basis function: 
    poly_basis = np.ones((data_set.shape[0], 1)) # first column full of ones for degree 0
    for i in range (data_set.shape[1]):
        for d in range (1, degree+1):
            poly_basis = np.c_[poly_basis, pow(data_set[:,i], d)] # add a new column with the x to the power i
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    return poly_basis

# -------------------------------------------------------------------------- #

def cross_validation_poly(y_class, data_class, parameters):
    for d in parameters.degree:
        param = cross_validation(y_class, data_class, parameters)
        if (param.best_accuracy < parameters.best_accuracy):
            parameters.set_best_accuracy(parameters, param.best_accuracy)
            parameters.set_best_degree(parameters, d)
    return parameters

# -------------------------------------------------------------------------- #

def test_function(data_y, data_set, parameters, class_ind):
    # Classification of the data set depending on the feature 23 (or 22?)
    # Classification of y
    y_classes = y_classification(data_y, data_set)
    y_class = y_classes[class_ind]

    # Classification of the data set
    data_classes = EDA_class(data_set)
    data_class = data_classes[class_ind]

    # Initialization of the weight parameters
    parameters.set_init_w(np.zeros(data_class.shape[1]))

    # Evaluation part
    opt_paramters = cross_validation(y_class, data_class, parameters)

    # Return the optimal parameters for the considered method and loss function
    return opt_paramters

# -------------------------------------------------------------------------- #

def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)