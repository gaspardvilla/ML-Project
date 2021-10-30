import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *
from EDA import *
from losses import *

# -------------------------------------------------------------------------- #

def method_evaluation(y, data_set, parameters, k_indices, k):
    """return the loss of the method"""

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
    
    # method to compute w and the loss:
    loss_tr_i, w = parameters.method(y_tr, x_tr, parameters)

    # Calculate the accuracy for the train and test set
    y_pred_tr = predict_labels(w, x_tr)
    nb_errors_tr, percentage_error_tr = counting_errors(y_pred_tr, y_tr)
    y_pred_te = predict_labels(w, x_te)
    nb_errors_te, percentage_error_te = counting_errors(y_pred_te, y_te)

    return percentage_error_tr, percentage_error_te
    
# -------------------------------------------------------------------------- #

def classic_cv(y_, class_, parameters, idx):
    seed = parameters.seeds[0]

    # split data in k fold
    k_indices = build_k_indices(y_, parameters.k_fold, seed)
    # define lists to store the loss of training data and test data
    error_te = []
    error_tr = []

    for param in parameters.range(idx):
        parameters.set_param(idx, param)
        error_tr_i = [-1]
        error_te_i = [-1]

        for k in range(parameters.k_fold):
            # cross validation:
            percentage_error_tr, percentage_error_te = \
                method_evaluation(y_, class_, parameters, k_indices, k)
            error_tr_i = np.c_[error_tr_i, [percentage_error_tr]]
            error_te_i = np.c_[error_te_i, [percentage_error_te]]

        error_tr.append(np.mean(error_tr_i[0, 1:]))
        error_te.append(np.mean(error_te_i[0, 1:]))

    best_param = parameters.range(idx)[np.argmin(error_te)]
    parameters.set_best_param(idx, best_param)
    parameters.set_param(idx, best_param)
    parameters.set_best_error(np.min(error_te))

    # Display the results
    min_test_error = np.min(error_te)
    if parameters.viz:
        print('Test error: ' +str(min_test_error)+ '\nBest ' \
            +str(parameters.names[idx-1])+ ': ' +str(parameters.best_param(idx)))

    # Visualization
    if parameters.viz:
        cross_validation_visualization(parameters.range(idx), error_tr, error_te, parameters)
    
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
        # split data in k fold
        seed = parameters.seeds[0]
        k_indices = build_k_indices(y_class, parameters.k_fold, seed)

        error_tr_i = [-1]
        error_te_i = [-1]

        for k in range(parameters.k_fold):
            # cross validation:
            percentage_error_tr, percentage_error_te = \
                method_evaluation(y_class, data_class, parameters, k_indices, k)
            error_tr_i = np.c_[error_tr_i, [percentage_error_tr]]
            error_te_i = np.c_[error_te_i, [percentage_error_te]]
        
        parameters.set_best_error(np.mean(error_te_i[0, 1:]))
        if parameters.viz:
            print('Test error: ' +str(parameters.best_error)+ '\n')
            print('No parameter to optimize for this method and this loss function')

    # Return the optimal parameters for the considered method and loss function
    return parameters

# -------------------------------------------------------------------------- #

def build_poly(data_set, degrees):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # polynomial basis function: 
    poly_basis = np.ones((data_set.shape[0], 1)) # first column full of ones for degree 0
    for d in degrees:
        for i in range (data_set.shape[1]):
            poly_basis = np.c_[poly_basis, pow(data_set[:,i], d)] # add a new column with the x to the power i
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    return poly_basis

# -------------------------------------------------------------------------- #

def add_feature(data_class, opt_class, feat_idx, degree):
    opt_class = np.c_[opt_class, pow(data_class[:, int(feat_idx)], degree)]
    return opt_class

# -------------------------------------------------------------------------- #

def add_interaction(data_class, opt_class, feat_idx_1, feat_idx_2):
    opt_class_ = np.c_[opt_class, \
        np.multiply(data_class[:, int(feat_idx_1)], data_class[:, int(feat_idx_2)])]
    return opt_class_

# -------------------------------------------------------------------------- #

def cross_validation_poly_gas(y_class, data_class, parameters):
    print('Forward step')

    # Forward pass
    forward_error = 100
    forward_lambda = parameters.best_lambda
    forward_gamma = parameters.best_gamma
    nb_features = data_class.shape[1]
    forward_class = np.ones([data_class.shape[0], 1])
    # Loop for the forward selection
    for degree in range(1, parameters.degree + 1):
        for feat_idx in range(nb_features):
            forward_class_current = add_feature(data_class, forward_class, feat_idx, degree)
            parameters = cross_validation(y_class, forward_class_current, parameters)
            if (parameters.best_error <= forward_error):
                forward_class = forward_class_current.copy()
                forward_error = parameters.best_error
                forward_lambda = parameters.best_lambda
                forward_gamma = parameters.best_gamma
                parameters.add_feature(feat_idx, degree)
    
    print('Backward step')

    # Backward pass
    backward_error = 100
    backward_lambda = parameters.best_lambda
    backward_gamma = parameters.best_gamma
    # Loop for the backward selection
    if parameters.use_backward_selection:
        backward_class = build_poly(data_class, range(1, parameters.degree + 1))
        backward_error = 100
        feat_list_back = [-1]
        idx = backward_class.shape[1]-1
        while idx >= 0:
            backward_class_current = remove_feature(backward_class, [idx])
            parameters = cross_validation(y_class, backward_class_current, parameters)
            if (parameters.best_error <= backward_error):
                # Class and index update
                backward_class = backward_class_current.copy()
                feat_list_back = np.c_[feat_list_back, idx]

                # Parameter update
                backward_error = parameters.best_error
                backward_lambda = parameters.best_lambda
                backward_gamma = parameters.best_gamma
            idx = idx - 1
    
    print('Interactions step')

    # Forward for interactions
    if parameters.use_interactions:
        for idx_1 in range(data_class.shape[1]):
            for idx_2 in range(idx_1 + 1, data_class.shape[1]):
                forward_class_current = add_interaction(data_class, forward_class, idx_1, idx_2)
                parameters = cross_validation(y_class, forward_class_current, parameters)
                if (parameters.best_error <= forward_error):
                    forward_class = forward_class_current.copy()
                    forward_error = parameters.best_error
                    forward_lambda = parameters.best_lambda
                    forward_gamma = parameters.best_gamma
                    parameters.add_interactions(idx_1, idx_2)

    # Small print
    print('forward: ', forward_error)
    print('backward: ', backward_error)
    
    # Comparison between backward and forward
    if backward_error < forward_error:
        error = backward_error
        parameters.set_polynomial_selection('Backward')
        parameters.set_selected_feature(feat_list_back)
        parameters.set_best_gamma(backward_gamma)
        parameters.set_best_lambda(backward_lambda)
    else:
        error = forward_error
        parameters.set_polynomial_selection('Forward')  
        parameters.set_best_gamma(forward_gamma)
        parameters.set_best_lambda(forward_lambda)      

    
    # Update the very best parameters
    parameters.set_best_error(error)
    parameters.set_lambda(parameters.best_lambda)
    parameters.set_gamma(parameters.best_gamma)
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

def cross_validation_visualization(lambds, error_tr, error_te, parameters):
    """visualization the curves of error_tr and error_te."""
    plt.semilogx(lambds, error_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, error_te, marker=".", color='r', label='test error')
    plt.xlabel('%s' %parameters.names[0])
    plt.ylabel("error")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.show()

# -------------------------------------------------------------------------- #

def build_polynomial_features(data_set, parameters):
    # In the case of forward solution
    if parameters.polynomial_selection == 'Forward':
        returned_set = np.ones([data_set.shape[0],1])
        for idx in range(1, parameters.feature_list.shape[1]):
            returned_set = add_feature(data_set, returned_set, \
                parameters.feature_list[0, int(idx)], parameters.feature_list[1, int(idx)])

        if (parameters.use_interactions) and (parameters.kept_interactions.sum() != 0.0):
            for idx in range(1, parameters.kept_interactions.shape[1]):
                returned_set = add_interaction(data_set, returned_set, \
                    parameters.kept_interactions[0, int(idx)], parameters.kept_interactions[1, int(idx)])
    else:
        returned_set = build_poly(data_set, range(1, parameters.degree + 1))
        returned_set = remove_feature(returned_set, parameters.feature_list[1:])

    return returned_set

# -------------------------------------------------------------------------- #