import math
import numpy as np
from proj1_helpers import *

# -------------------------------------------------------------------------- #

def least_squares(y, data_set, parameters):
    """This function calculates the least squares solution.
    
    Args:
        y: the considerated output
        data_set: the considered data set
        parameters: the parameters to consider for the computation of loss and weights

    Returns:
        loss: the loss obtained
        w: the optimal weights obtained
    """
    # Define a and b for solving linear system 'ax = b'
    parameters.set_init_w(np.zeros((data_set.shape[1],)))
    a = data_set.T.dot(data_set)
    b = data_set.T.dot(y)

    # Solve for the weight
    w = np.linalg.solve(a, b)

    # Computation of the loss according to the cost function of the considered parameters
    loss = parameters.loss_fct.cost(y, data_set, w)

    # returns loss and optimal weights
    return loss, w

# -------------------------------------------------------------------------- #

def least_squares_GD(y, data_set, parameters):
    """ This function performs gradient descent (GD) algorithm 
    
    Args:
        y: the considerated output
        data_set: the considered data set
        parameters: the parameters to consider

    Returns:
        loss: the loss obtained
        w: the optimal weights obtained
    """
    
    # Initialization of some parameters
    loss = 0
    parameters.set_init_w(np.zeros((data_set.shape[1],)))
    w = parameters.initial_w
    
    # Loop for the number of iterations defined by parameters
    for nb_iter in range(parameters.max_iter):
        
        # Compute the gradient according to the gradient function of the considered parameters
        grad = parameters.loss_fct.grad(y, data_set, w)
        
        # Compute the loss according to the cost function of the considered parameters
        loss = parameters.loss_fct.cost(y, data_set, w)
        
        # Update the weight accorging to the value gamma of the considered parameters
        w = w - (parameters.gamma * grad)
    
    # Return the final loss and weight
    return loss, w

# -------------------------------------------------------------------------- #

def ridge_regression(y, data_set, parameters):
    """This function implements ridge regression.
    
    Args:
        y: the considerated output
        data_set: the considered data set
        parameters: the parameters to consider

    Returns:
        loss: the loss obtained
        w: the optimal weights obtained
    """  
    
    # Initialization of some parameters
    parameters.set_init_w(np.zeros((data_set.shape[1],)))   
    
    # Define a and b for solving linear system 'ax = b'
    a = data_set.T.dot(data_set) + (2 * data_set.shape[0] * parameters.lambda_ * np.identity(data_set.shape[1]))
    b = data_set.T.dot(y)
    
    # Solve for the weight
    w = np.linalg.solve(a, b)
    
    # Compute the loss according to the cost function of the considered parameters
    loss = parameters.loss_fct.cost(y, data_set, w)
    
    # Return the final loss and weight
    return loss, w

# -------------------------------------------------------------------------- #

def least_squares_SGD(y, data_set, parameters):
    """ This function performs stochastic gradient descent (SGD) algorithm. 
    
    Args:
        y: the considerated output
        data_set: the considered data set
        parameters: the parameters to consider

    Returns:
        loss: the loss obtained
        w: the optimal weights obtained
    """   

    # Initialization of some parameters
    """uncomment if you want to print the losses for each steps
    ws = [parameters.initial_w]
    losses = []
    """
    parameters.set_init_w(np.zeros((data_set.shape[1],)))
    w = parameters.initial_w

    # Loop for the number of iterations defined by parameters
    for n_iter in range(parameters.max_iter):
        # Loop for the mini batch
        for mini_batch_y, mini_batch_tx in batch_iter(y, data_set, parameters.mini_batch_size):
            
            # Compute the gradient according to the gradient function of the considered parameters
            grad = parameters.loss_fct.grad(mini_batch_y, mini_batch_tx,w)
        
            # Compute the loss according to the cost function of the considered parameters
            loss = parameters.loss_fct.cost(mini_batch_y, mini_batch_tx, w)
        
            # Update the weight accorging to the value gamma of the considered parameters
            w = w - (parameters.gamma * grad)
            
            """uncomment if you want to print the losses for each steps
            # store w and loss
            ws.append(w)
            losses.append(loss)
            
            
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
            """
    
    # Return the final loss and weight
    return loss, w

# -------------------------------------------------------------------------- #

def logistic_regression(y, data_set, parameters):
    """ This function performs logistic regression algorithm. 
    
    Args:
        y: the considerated output
        data_set: the considered data set
        parameters: the parameters to consider

    Returns:
        loss: the loss obtained
        w: the optimal weights obtained
    """ 
    
    # Initialization of some parameters
    parameters.set_init_w(np.zeros((data_set.shape[1],)))
    w = parameters.initial_w
    loss = 0

    # Loop for the number of iterations defined by parameters
    for nb_iter in range(parameters.max_iter):
        # Compute the gradient according to the gradient function of the considered parameters
        grad = parameters.loss_fct.grad(y, data_set, w)
        
        # Compute the loss according to the cost function of the considered parameters
        loss = parameters.loss_fct.cost(y, data_set, w)
        
        # Update the weight accorging to the value gamma of the considered parameters
        w = w - (parameters.gamma * grad)
    
    # Return the final loss and weight
    return loss, w

# -------------------------------------------------------------------------- #

def reg_logistic_regression(y, data_set, parameters):
    """ This function performs regularized logistic regression algorithm. 
    
    Args:
        y: the considerated output
        data_set: the considered data set
        parameters: the parameters to consider

    Returns:
        loss: the loss obtained
        w: the optimal weights obtained
    """
    
    # Initialization of some parameters
    losses = []
    parameters.set_init_w(np.zeros((data_set.shape[1],)))
    w = parameters.initial_w

    # Loop for the number of iterations defined by parameters
    for iter in range(parameters.max_iter):
        
        # Compute the gradient according to the gradient function of the considered parameters
        grad = parameters.loss_fct.grad(y, data_set, w) + (2 * parameters.lambda_ * w)
        
        # Compute the loss according to the cost function of the considered parameters
        loss = parameters.loss_fct.cost(y, data_set, w) + (parameters.lambda_ * w.T.dot(w))

        # Update the weight accorging to the value gamma of the considered parameters
        w = w - (parameters.gamma * grad)

        # log info
        # print("Current iteration={}, loss={}".format(iter, loss))
        
        # Converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < parameters.threshold:
            break
    
    # Return the final loss and weight
    return loss, w

# -------------------------------------------------------------------------- #