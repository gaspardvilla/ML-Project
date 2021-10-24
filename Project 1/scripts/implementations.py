import math
import numpy as np
from proj1_helpers import *

# -------------------------------------------------------------------------- #

def least_squares(y, data_set, loss_fct, initial_w = 0, lambda_ = 0.7, gamma = 0.01, max_iters = 50):
    """calculate the least squares solution."""
    # Define a and b for solving linear system 'ax = b'
    a = data_set.T.dot(data_set)
    b = data_set.T.dot(y)

    # Computation of the solution
    w = np.linalg.solve(a, b)

    # Computation of the MSE
    loss = loss_fct.cost(y, data_set, w)

    # returns MSE and optimal weights
    return loss, w

# -------------------------------------------------------------------------- #

def least_squares_GD(y, data_set, loss_fct, initial_w, lambda_ = 0.7, gamma = 0.01, max_iters = 50):
    # Gradient descent algorithm (with MSE loss => maybe change loss)
    
    # Definition of all the parameters
    loss = 0
    w = initial_w
    
    # Loop for on the number of iterations
    for nb_iter in range(max_iters):
        
        # Compute the gradient
        grad = loss_fct.grad(y, data_set, w)
        
        # Compute the loss (MSE)
        loss = loss_fct.cost(y, data_set, w)
        
        # Update the weight parameters
        w = w - (gamma * grad)
    
    # Return the final loss and weight
    return loss, w

# -------------------------------------------------------------------------- #

def ridge_regression(y, tx, loss_fct, initial_w = 0, lambda_ = 0.7, gamma = 0.01, max_iters = 50):
    """implement ridge regression."""    
    a = tx.T.dot(tx) + (2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1]))
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = loss_fct.cost(y, tx, w)
    return mse, w

# -------------------------------------------------------------------------- #

def least_squares_SGD(y, data_set, loss_fct, initial_w = 0, lambda_ = 0.7, gamma = 0.01, max_iters = 50, mini_batch_size = 1):
    """Stochastic gradient descent algorithm."""

    # Initialization of some parameters
    ws = [initial_w]
    losses = []
    w = initial_w

    # Big loop
    for n_iter in range(max_iters):
        for mini_batch_y, mini_batch_tx in batch_iter(y, data_set, mini_batch_size):
        
            grad = loss_fct.grad(mini_batch_y, mini_batch_tx,w)
        
            loss = loss_fct.cost(mini_batch_y, mini_batch_tx, w)
        
            w = w - (gamma * grad)

            # store w and loss
            ws.append(w)
            losses.append(loss)
            '''uncomment if you want to print the losses for each steps'''
            #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            #      bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
            
    return loss, w



# -------------------------------------------------------------------------- #

def logistic_regression(y, tx, loss_fct, initial_w = 0, lambda_ = 0.7, gamma = 0.01, max_iters = 50):
    # Initialization
    w = initial_w
    loss = 0

    # Loop for on the number of iterations
    for nb_iter in range(max_iters):
        #loss = loss_fct.cost(y, tx, w)
        loss = loss_fct.cost(y, tx, w)
        #print(loss)
        #grad = loss_fct.grad(y, tx, w)
        grad = loss_fct.grad(y, tx, w)
        w = w - (gamma * grad)
    
    # Return the results
    return loss, w

# -------------------------------------------------------------------------- #

def reg_logistic_regression(y, data_set, loss_fct, initial_w = 0, lambda_ = 0.1, gamma = 0.01, max_iters = 1000, threshold = 1e-8):
    
    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), data_set]
    losses = []
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):

        # get loss and update w.
        loss = loss_fct.cost(y, tx, initial_w) + (lambda_ * initial_w.T.dot(initial_w))
        grad = loss_fct.grad(y, tx, initial_w) + (2 * lambda_ * initial_w)
        w = w - (gamma * grad)

        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    #return 
    return loss, w