import numpy as np

def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    opt_w = np.linalg.solve(a, b)
    e = y - tx.dot(opt_w)
    mse = 1/2*np.mean(e**2)
    # returns mse, and optimal weights
    return mse, opt_w

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    a = tx.T.dot(tx) + 2*tx.shape[0]*lambda_*np.identity(tx.shape[1])
    b = tx.T.dot(y)
    weights = np.linalg.solve(a, b)
    return weights
