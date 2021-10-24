import math
import numpy as np

# Functions that will be used for the Project 1

# -------------------------------------------------------------------------- #

def compute_mse(y, data_set, w):
    # Calculate the loss using the MSE formula
    
    # Definition of the parameters
    mse = 0
    N = len(y)
    e = y - data_set.dot(w)
    
    # Compute the loss MSE
    mse = (1 / (2 * N)) * e.T.dot(e)
    return mse

# -------------------------------------------------------------------------- #

def compute_gradient_mse(y, data_set, w):
    # Initilization
    N = len(y)

    # Compute the gradient of the Mean Square Error
    e = y - data_set.dot(w)
    grad = -(1 / N) * data_set.T.dot(e)
     
    # Return the results grad
    return grad

# -------------------------------------------------------------------------- #

def least_squares(y, data_set, initial_w = 0, lambda_ = 0.7, gamma = 0.01, max_iters = 50):
    """calculate the least squares solution."""
    # Define a and b for solving linear system 'ax = b'
    a = data_set.T.dot(data_set)
    b = data_set.T.dot(y)

    # Computation of the solution
    w = np.linalg.solve(a, b)

    # Computation of the MSE
    mse = compute_mse(y, data_set, w)

    # returns MSE and optimal weights
    return mse, w

# -------------------------------------------------------------------------- #

def least_squares_GD(y, data_set, initial_w, lambda_ = 0.7, gamma = 0.01, max_iters = 50):
    # Gradient descent algorithm (with MSE loss => maybe change loss)
    
    # Definition of all the parameters
    loss = 0
    w = initial_w
    
    # Loop for on the number of iterations
    for nb_iter in range(max_iters):
        
        # Compute the gradient
        grad = compute_gradient_mse(y, data_set, w)
        
        # Compute the loss (MSE)
        loss = compute_mse(y, data_set, w)
        
        # Update the weight parameters
        w = w - (gamma * grad)
    
    # Return the final loss and weight
    return loss, w

# -------------------------------------------------------------------------- #

def ridge_regression(y, tx, initial_w = 0, lambda_ = 0.7, gamma = 0.01, max_iters = 50):
    """implement ridge regression."""    
    a = tx.T.dot(tx) + (2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1]))
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = compute_mse(y, tx, w)
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

def least_squares_SGD(y, data_set, initial_w = 0, lambda_ = 0.7, gamma = 0.01, max_iters = 50, mini_batch_size = 1):
    """Stochastic gradient descent algorithm."""

    # Initialization of some parameters
    ws = [initial_w]
    losses = []
    w = initial_w

    # Big loop
    for n_iter in range(max_iters):
        for mini_batch_y, mini_batch_tx in batch_iter(y, data_set, mini_batch_size):
        
            stoch_gradient = compute_gradient_mse(mini_batch_y, mini_batch_tx,w)
        
            loss = compute_mse(mini_batch_y, mini_batch_tx, w)
        
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

def compute_loss_lr(y, data_set, w):
    # Initialization of sigma
    t = data_set.dot(w)
    sigma = sigmoid(t)

    # Direct computation of the loss
    loss = -(y.T.dot(np.log(sigma)) + (1 - y).T.dot(np.log(1 - sigma)))

    # Return the loss
    return loss

# -------------------------------------------------------------------------- #

def compute_gradient_lr(y, tx, w):
    # Initialization of sigma
    sigma = sigmoid(tx.dot(w))

    # Direct computation of the gradient
    grad = np.transpose(tx).dot(sigma - y)

    # Return the gradient
    return grad

# -------------------------------------------------------------------------- #

def logistic_regression(y, tx, initial_w = 0, lambda_ = 0.7, gamma = 0.01, max_iters = 50):
    # Initialization
    w = initial_w
    loss = 0

    # Loop for on the number of iterations
    for nb_iter in range(max_iters):
        loss = compute_loss_lr(y, tx, w)
        #print(loss)
        grad = compute_gradient_lr(y, tx, w)
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

def reg_logistic_regression(y, data_set, initial_w = 0, lambda_ = 0.1, gamma = 0.01, max_iters = 1000, threshold = 1e-8):
    
    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), data_set]
    losses = []
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):

        # get loss and update w.
        loss = compute_loss_lr(y, tx, initial_w) + (lambda_ * initial_w.T.dot(initial_w))
        grad = compute_gradient_lr(y, tx, initial_w) + (2 * lambda_ * initial_w)
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