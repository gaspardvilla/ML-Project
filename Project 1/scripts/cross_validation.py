import math
import numpy as np
import matplotlib.pyplot as plt
from implementations import *

def cross_validation(y, x, loss_fct, k_indices, k, method, initial_w, lambda_ = 0.7, gamma = 0.01, max_iters = 50):
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

    x_tr = x[tr_idx]
    x_te = x[te_idx]
    y_tr = y[tr_idx]
    y_te = y[te_idx]
    
    # ridge regression:
    loss_tr_i, w = method(y_tr, x_tr, loss_fct, initial_w, lambda_, gamma, max_iters)
    
    # calculate the loss for train and test data:       
    loss_tr.append(loss_tr_i)
    loss_te.append(loss_fct.cost(y_te, x_te, w))

    return loss_tr, loss_te

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
    
# -------------------------------------------------------------------------- #

def cross_validation_plot(y_, class_, loss_fct, method, lambdas = np.logspace(-4, 0, 30), gammas = np.logspace(-4, 0, 30)):
    seed = 1
    k_fold = 4
    
    initial_w = np.zeros(class_.shape[1])
    # split data in k fold
    k_indices = build_k_indices(y_, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []

    for lambda_ in lambdas:
        rmse_tr_i = []
        rmse_te_i = []

        for k in range(k_fold):
            # cross validation:
            loss_tr_i, loss_te_i = cross_validation(y_, class_, loss_fct, k_indices, k, method, initial_w, lambda_, gamma = 0.01, max_iters = 50) 
            rmse_tr_i.append(np.sqrt(2 * loss_tr_i))
            rmse_te_i.append(np.sqrt(2 * loss_te_i))

        rmse_tr.append(np.mean(rmse_tr_i))
        rmse_te.append(np.mean(rmse_te_i))

    cross_validation_visualization(lambdas, rmse_tr, rmse_te)