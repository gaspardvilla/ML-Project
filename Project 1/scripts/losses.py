import math
import numpy as np

# File that contains all the cost functions with these attributes.

# MSE loss
class MSE():
    def cost(self, y, data_set, w):
        # Definition of the parameters
        loss = 0
        N = len(y)
        e = y - data_set.dot(w)
        
        # Compute the loss MSE
        loss = (1 / (2 * N)) * e.T.dot(e)

        # Return the MSE
        return loss
    
    def grad(self, y, data_set, w):
        # Initilization
        N = len(y)

        # Compute the gradient of the Mean Square Error
        e = y - data_set.dot(w)
        grad = -(1 / N) * data_set.T.dot(e)
        
        # Return the results grad
        return grad

# -------------------------------------------------------------------------- #

# MAE loss
class MAE():
    def cost(self, y, data_set, w):
        # Definition of the parameters
        loss = 0
        N = len(y)
        e = y - data_set.dot(w)
        
        # Compute the loss MSE
        loss = (1 / N) * np.sqrt(e.T.dot(e))

        # Return the MSE
        return loss
    
    def grad(self, y, data_set, w):
        # Initilization
        N = len(y)

        # Compute the gradient of the Mean Square Error
        e = y - data_set.dot(w)
        grad = -(1 / N) * np.multiply(np.sign(e), data_set).T.dot(np.ones(len(y)))
        
        # Return the results grad
        return grad

# -------------------------------------------------------------------------- #

# Negative log-likelihood loss
class Neg_log():
    # Sigmoid functions
    def sigmoid(self, t):
        return 1.0 / (1 + np.exp(t))
    
    def cost(self, y, data_set, w):
        # Initialization of sigma
        t = data_set.dot(w)
        sigma = self.sigmoid(t)

        # Direct computation of the loss
        loss = -(y.T.dot(np.log(sigma)) + (1 - y).T.dot(np.log(1 - sigma)))

        # Return the loss
        return loss
    
    def grad(self, y, data_set, w):
        # Initialization of sigma
        sigma = self.sigmoid(data_set.dot(w))

        # Direct computation of the gradient
        grad = np.transpose(data_set).dot(sigma - y)

        # Return the gradient
        return grad

# -------------------------------------------------------------------------- #