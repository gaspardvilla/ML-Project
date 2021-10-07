import math
import numpy as np

# Functions that will be used for the Project 1

# -------------------------------------------------------------------------- #

def compute_mse(y, tx, w):
    # Calculate the loss using the MSE formula
    
    # Definition of the parameters
    MSE_loss = 0
    N = len(y)
    error = y - tx.dot(w)
    
    # Compute the loss MSE
    MSE_loss = (1 / (2 * N)) * np.transpose(error).dot(error)
    return MSE_loss

# -------------------------------------------------------------------------- #

def compute_gradient_mse(y, tx, w):
    # Compute the gradient of the Mean Square Error
    
    # Definition of the parameters
    N = len(y)
    e = y - tx.dot(w)
    
    # Compute directly the gradient
    grad = (-1 / N) * np.transpose(tx).dot(e)
    
    return grad

# -------------------------------------------------------------------------- #

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    # Gradient descent algorithm
    
    # Definition of all the parameters
    weights = [initial_w]
    losses = []
    weight = initial_w
    
    # Loop for on the number of iterations
    for nb_iter in range(max_iters):
        
        # Compute the gradient
        grad = compute_gradient_mse(y, tx, weight)
        
        # Compute the loss (MSE)
        loss = compute_mse(y, tx, weight)
        
        # Update the weight parameters
        weight = weight - (gamma * grad)
        
        # Store the losses and the weights
        weights.append(weight)
        losses.append(loss)
        
        # Print the loss at each iteration
        print("Gradient Descent({bi}/{ti}): loss={l}".format(bi = nb_iter,
                                                ti = max_iters - 1, l = loss))
        
    return loss, weight

# -------------------------------------------------------------------------- #

def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = compute_mse (y, tx, w)
    # returns mse, and optimal weights
    return mse, w

# -------------------------------------------------------------------------- #

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    a = tx.T.dot(tx) + 2*tx.shape[0]*lambda_*np.identity(tx.shape[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = compute_mse (y, tx, w)
    return mse, w


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e = y - tx.dot(w)
    
    grad = -1/len(tx) * np.transpose(tx).dot(e)
    
    return grad


def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, batch_size = 10):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        
            stoch_gradient = compute_stoch_gradient(minibatch_y,minibatch_tx,w)
        
            loss = compute_mse(minibatch_y,minibatch_tx,w)
        
            w = w - gamma * stoch_gradient
            # store w and loss
            ws.append(w)
            losses.append(loss)
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
            
    return losses, ws