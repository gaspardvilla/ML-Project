import numpy as np

def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = compute_mse (y, tx, w)
    # returns mse, and optimal weights
    return mse, weights

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    a = tx.T.dot(tx) + 2*tx.shape[0]*lambda_*np.identity(tx.shape[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = compute_mse (y, tx, w)
    return mse, w
