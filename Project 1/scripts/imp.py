import numpy as np
import math
from implementations import *
from losses import *

# -------------------------------------------------------------------------- #

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """This function calculates the least squares solution using gradient descent algorithm.
    
    Args:
        y: the considerated output
        tx: the considered data set
        initial_w: the intial weights
        max_iters: numbers of iterations to do for gradient descent
        gamma: parameter used for updating the weights after each iteration

    Returns:
        loss: the loss obtained with our method least_squares(y, data_set, parameters) in methods.py
        w: the optimal weights obtained with our method least_squares(y, data_set, parameters) in methods.py
    """
    #Setting parameters with the arguments' values
    param = Parameters()
    param.set_init_w(initial_w)
    param.set_max_iter(max_iters)
    param.set_gamma(gamma)
    
    # Compute loss and weight with our method least_squares_GD(y, data_set, parameters) in methods.py
    loss, w = least_squares_GD_(y, data_set, param)
    
    return loss, w

# -------------------------------------------------------------------------- #

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """This function calculates the least squares solution using stochastic gradient descent algorithm.
    
    Args:
        y: the considerated output
        tx: the considered data set
        initial_w: the intial weights
        max_iters: numbers of iterations to do for gradient descent
        gamma: parameter used for updating the weights after each iteration

    Returns:
        loss: the loss obtained with our method least_squares(y, data_set, parameters) in methods.py
        w: the optimal weights obtained with our method least_squares(y, data_set, parameters) in methods.py
    """
    #Setting parameters with the arguments' values
    param = Parameters()
    param.set_init_w(initial_w)
    param.set_max_iter(max_iters)
    param.set_gamma(gamma)
    
    # Compute loss and weight with our method least_squares_SGD(y, data_set, parameters) in methods.py
    loss, w = least_squares_SGD_(y, data_set, param)
    
    return loss, w

# -------------------------------------------------------------------------- #

def least_squares(y, tx):
    """This function calculates the least squares solution.
    
    Args:
        y: the considerated output
        tx: the considered data set

    Returns:
        loss: the loss obtained with our method least_squares(y, data_set, parameters) in methods.py
        w: the optimal weights obtained with our method least_squares(y, data_set, parameters) in methods.py
    """
    param = Parameters()
    
    # Compute loss and weight with our method least_squares(y, data_set, parameters) in methods.py
    loss, w = least_squares_(y, tx, param)
    
    return loss, w
# -------------------------------------------------------------------------- #

def ridge_regression(y, tx, lambda_ ):
    """This function implements Ridge regression.
    
    Args:
        y: the considerated output
        tx: the considered data set
        lambda_: parameter for ridge penalty

    Returns:
        loss: the loss obtained with our method least_squares(y, data_set, parameters) in methods.py
        w: the optimal weights obtained with our method least_squares(y, data_set, parameters) in methods.py
    """
    #Setting parameters with the argument value
    param = Parameters()
    param.set_lambda(lambda_)
    
    # Compute loss and weight with our method ridge_regression(y, data_set, parameters) in methods.py
    loss, w = ridge_regression_(y, tx, param)
    
    return loss, w
    
# -------------------------------------------------------------------------- #

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """This function compute the logistic regression.
    
    Args:
        y: the considerated output
        tx: the considered data set
        initial_w: the intial weights
        max_iters: numbers of iterations to do for gradient descent
        gamma: parameter used for updating the weights after each iteration

    Returns:
        loss: the loss obtained with our method least_squares(y, data_set, parameters) in methods.py
        w: the optimal weights obtained with our method least_squares(y, data_set, parameters) in methods.py
    """
    #Setting parameters with the arguments' values
    param = Parameters()
    param.set_init_w(initial_w)
    param.set_max_iter(max_iters)
    param.set_gamma(gamma)
    
    # Compute loss and weight with our method logistic_regression(y, data_set, parameters) in methods.py
    loss, w = logistic_regression_(y, data_set, param)
    
    return loss, w

# -------------------------------------------------------------------------- #

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """This function compute the regularized logistic regression.
    
    Args:
        y: the considerated output
        tx: the considered data set
        initial_w: the intial weights
        lambda_ : parameter for regularization
        max_iters: numbers of iterations to do for gradient descent
        gamma: parameter used for updating the weights after each iteration

    Returns:
        loss: the loss obtained with our method least_squares(y, data_set, parameters) in methods.py
        w: the optimal weights obtained with our method least_squares(y, data_set, parameters) in methods.py
    """
    #Setting parameters with the arguments' values
    param = Parameters()
    param.set_init_w(initial_w)
    param.set_lambda(lambda_)
    param.set_max_iter(max_iters)
    param.set_gamma(gamma)
    
    # Compute loss and weight with our method reg_logistic_regression(y, data_set, parameters) in methods.py
    loss, w = reg_logistic_regression_(y, data_set, param)
    
    return loss, w

# -------------------------------------------------------------------------- #
    
