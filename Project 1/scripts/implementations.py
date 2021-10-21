import math
import numpy as np

# Functions that will be used for the Project 1

# -------------------------------------------------------------------------- #

def compute_mse(y, tx, w):
    # Calculate the loss using the MSE formula
    
    # Definition of the parameters
    mse = 0
    N = len(y)
    e = y - tx.dot(w)
    
    # Compute the loss MSE
    mse = (1 / (2 * N)) * e.T.dot(e)
    return mse

# -------------------------------------------------------------------------- #

def compute_gradient_mse(y, tx, w):
    # Compute the gradient of the Mean Square Error
    e = y - tx.dot(w)
    gradient = -tx.T.dot(e)/len(e)
    
    return gradient

# -------------------------------------------------------------------------- #

def least_squares(y, tx, initial_w = 0, lambda_ = 0.7, gamma = 0.01, max_iters = 50):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = compute_mse (y, tx, w)
    # returns mse, and optimal weights
    return mse, w

# -------------------------------------------------------------------------- #

def least_squares_GD(y, tx, initial_w = 0, lambda_ = 0.7, gamma = 0.01, max_iters = 50):
    # Gradient descent algorithm
    
    # Definition of all the parameters
    loss = 0
    w = initial_w
    
    # Loop for on the number of iterations
    for nb_iter in range(max_iters):
        
        # Compute the gradient
        grad = compute_gradient_mse(y, tx, w)
        
        # Compute the loss (MSE)
        loss = compute_mse(y, tx, w)
        
        # Update the weight parameters
        w = w - (gamma * grad)
    return loss, w

# -------------------------------------------------------------------------- #

def ridge_regression(y, tx, initial_w = 0, lambda_ = 0.7, gamma = 0.01, max_iters = 50):
    """implement ridge regression."""
    a = tx.T.dot(tx) + 2*tx.shape[0]*lambda_*np.identity(tx.shape[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = compute_mse (y, tx, w)
    return mse, w

# -------------------------------------------------------------------------- #

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e = y - tx.dot(w)
    
    gradient = -1/len(tx) * tx.T.dot(e)
    
    return gradient

# -------------------------------------------------------------------------- #

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

# -------------------------------------------------------------------------- #

def stochastic_gradient_descent(y, tx, initial_w = 0, lambda_ = 0.7, gamma = 0.01, max_iters = 50, batch_size = 10):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        
            stoch_gradient = compute_stoch_gradient(minibatch_y, minibatch_tx,w)
        
            loss = compute_mse(minibatch_y, minibatch_tx,w)
        
            w = w - gamma * stoch_gradient
            # store w and loss
            ws.append(w)
            losses.append(loss)
            '''uncomment if you want to print the losses for each steps'''
            #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            #      bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
            
    return loss, w

# -------------------------------------------------------------------------- #

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

# -------------------------------------------------------------------------- #

def compute_loss_lr(y, tx, w):
    #return loss for logistic regression
    t = tx.dot(w)
    sigma = sigmoid(t)
    return - (y.T.dot(np.log(sigma)) + (1 - y).T.dot(np.log(1 - sigma)))

# -------------------------------------------------------------------------- #

def compute_gradient(y, tx, w):
    sigma = sigmoid(tx.dot(w))
    grad = np.transpose(tx).dot(sigma - y)
    return grad

# -------------------------------------------------------------------------- #

def logistic_regression_GD(y, tx, initial_w = 0, lambda_ = 0.7, gamma = 0.01, max_iters = 50):
    # Initialization
    w = initial_w
    loss = 0

    # Loop for on the number of iterations
    for nb_iter in range(max_iters):
        loss = compute_loss_lr(y, tx, w)
        grad = compute_gradient(y, tx, w)
        w = w - gamma * grad
    
    # Return the results
    return loss, w

# -------------------------------------------------------------------------- #

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

# -------------------------------------------------------------------------- #

def cross_validation(y, x, k_indices, k, method, initial_w, lambda_ = 0.7, gamma = 0.01, max_iters = 50):
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
    print('tr_idx ' + str(tr_idx.shape))
    print('te_idx ' + str(te_idx.shape))

    x_tr = x[tr_idx]
    x_te = x[te_idx]
    y_tr = y[tr_idx]
    y_te = y[te_idx]
        
    # ridge regression:
    loss_tr_i, w = method(y_tr, x_tr, initial_w, lambda_, gamma, max_iters)
    
    # calculate the loss for train and test data:       
    loss_tr.append(loss_tr_i)
    loss_te.append(compute_mse(y_te, x_te, w))

    return loss_tr, loss_te

# -------------------------------------------------------------------------- #
