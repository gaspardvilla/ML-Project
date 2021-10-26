import math
import numpy as np
from proj1_helpers import *

# -------------------------------------------------------------------------- #

def least_squares(y, data_set, parameters):
    """calculate the least squares solution."""
    # Define a and b for solving linear system 'ax = b'
    a = data_set.T.dot(data_set)
    b = data_set.T.dot(y)

    # Computation of the solution
    w = np.linalg.solve(a, b)

    # Computation of the MSE
    loss = parameters.loss_fct.cost(y, data_set, w)

    # returns MSE and optimal weights
    return loss, w

# -------------------------------------------------------------------------- #

def least_squares_GD(y, data_set, parameters):
    # Gradient descent algorithm (with MSE loss => maybe change loss)
    
    # Definition of all the parameters
    loss = 0
    w = parameters.initial_w
    
    # Loop for on the number of iterations
    for nb_iter in range(parameters.max_iter):
        
        # Compute the gradient
        grad = parameters.loss_fct.grad(y, data_set, w)
        
        # Compute the loss (MSE)
        loss = parameters.loss_fct.cost(y, data_set, w)
        
        # Update the weight parameters
        w = w - (parameters.gamma * grad)
    
    # Return the final loss and weight
    return loss, w

# -------------------------------------------------------------------------- #

def ridge_regression(y, data_set, parameters):
    """implement ridge regression."""    
    a = data_set.T.dot(data_set) + (2 * data_set.shape[0] * parameters.lambda_ * np.identity(data_set.shape[1]))
    b = data_set.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = parameters.loss_fct.cost(y, data_set, w)
    return loss, w

# -------------------------------------------------------------------------- #

def least_squares_SGD(y, data_set, parameters):
    """Stochastic gradient descent algorithm."""

    # Initialization of some parameters
    ws = [parameters.initial_w]
    losses = []
    w = parameters.initial_w

    # Big loop
    for n_iter in range(parameters.max_iter):
        for mini_batch_y, mini_batch_tx in batch_iter(y, data_set, parameters.mini_batch_size):
        
            grad = parameters.loss_fct.grad(mini_batch_y, mini_batch_tx,w)
        
            loss = parameters.loss_fct.cost(mini_batch_y, mini_batch_tx, w)
        
            w = w - (parameters.gamma * grad)

            # store w and loss
            ws.append(w)
            losses.append(loss)
            '''uncomment if you want to print the losses for each steps'''
            #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            #      bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
            
    return loss, w



# -------------------------------------------------------------------------- #

def logistic_regression(y, data_set, parameters):
    # Initialization
    # w = parameters.initial_w
    data_set = np.c_[np.ones((y.shape[0], 1)), data_set]
    w = np.zeros((data_set.shape[1],))
    loss = 0

    # Loop for on the number of iterations
    for nb_iter in range(parameters.max_iter):
        #loss = loss_fct.cost(y, data_set, w)
        loss = parameters.loss_fct.cost(y, data_set, w)
        #print(loss)
        #grad = loss_fct.grad(y, data_set, w)
        grad = parameters.loss_fct.grad(y, data_set, w)
        w = w - (parameters.gamma * grad)
    
    # Return the results
    return loss, w

# -------------------------------------------------------------------------- #

def reg_logistic_regression(y, data_set, parameters):
    
    # build data_set
    losses = []
    w = parameters.initial_w

    # start the logistic regression
    for iter in range(parameters.max_iter):

        # get loss and update w.
        loss = parameters.loss_fct.cost(y, data_set, w) + (parameters.lambda_ * w.T.dot(w))
        grad = parameters.loss_fct.grad(y, data_set, w) + (2 * parameters.lambda_ * w)
        w = w - (parameters.gamma * grad)

        # log info
        # print("Current iteration={}, loss={}".format(iter, loss))
        
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < parameters.threshold:
            break
    
    #return 
    return loss, w