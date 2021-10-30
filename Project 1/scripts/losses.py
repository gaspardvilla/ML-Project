import numpy as np

from implementations import least_squares_GD, ridge_regression

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
        return np.exp(t) / (1 + np.exp(t))
    
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
        grad = data_set.T.dot(sigma - y)

        # Return the gradient
        return grad

# -------------------------------------------------------------------------- #

# Set a class parameters for the implementations functions and other stuff
class Parameters(object):
    def __init__(self):
        # Initialization of the principal paramaters
        self.initial_w = 0
        self.gamma = 1e-3
        self.lambda_ = 1e-3
        self.degree = 1
        self.max_iter = 100
        self.threshold = 1e-6
        self.k_fold = 4
        self.mini_batch_size = 1
        # Set the range of lambda and gamma for the cross validation
        self.lambda_range = np.logspace(-8, 0, 30)
        self.gamma_range = np.logspace(-8, 0, 30)
        # Set the seeds use in cross validation
        self.seeds = np.arange(1)
        # Indicates the number of parameters to test and which of them for the cross validation
        self.nb_to_test = 0
        self.names = []
        self.best_lambda = self.lambda_
        self.best_gamma = self.gamma

        # To supress after that
        self.best_degree =self.degree
        self.feature_list = np.zeros(2)

        # Visualization
        self.viz = False
        # Method and loss function
        self.method = least_squares_GD
        self.loss_fct = MSE()
        # Indicator if the loss is logitic regression
        self.logistic = False
        # Optimal test error
        self.best_error = 100

        self.polynomial_selection = 'Forward'
        self.use_backward_selection = True
        self.use_interactions = True
        self.kept_interactions = np.zeros(2)

    # Setting all the parameters of this class
    def set_init_w(self, initial_w):
        self.initial_w = initial_w
    
    def set_gamma(self, gamma):
        self.gamma = gamma
    
    def set_lambda(self, lambda_):
        self.lambda_ = lambda_
        
    def set_degree(self, degree):
        self.degree = degree
    
    def set_max_iter(self, max_iter):
        self.max_iter = max_iter
    
    def set_threshold(self, threshold):
        self.threshold = threshold
    
    def set_lambda_range(self, range):
        self.lambda_range = range

    def set_gamma_range(self, range):
        self.gamma_range = range
    
    def set_nb_seeds(self, nb_seeds):
        self.seeds = np.arange(nb_seeds)

    def set_k_fold(self, k_fold):
        self.k_fold = k_fold

    def set_mini_batch_size(self, mini_batch_size):
        self.mini_batch_size = mini_batch_size

    def set_viz(self, viz):
        self.viz = viz

    def set_method(self, method):
        self.method = method
    
    def set_loss_fct(self, loss_fct):
        self.loss_fct = loss_fct

    def set_best_error(self, test_error):
        self.best_error = test_error

    def set_to_test(self, names):
        self.nb_to_test = len(names)
        self.names = names

    def add_feature(self, feature, degree):
        self.feature_list = np.c_[self.feature_list, [feature, degree]]

    def set_polynomial_selection(self, polynomial_selection):
        self.polynomial_selection = polynomial_selection
    
    def set_selected_feature(self, feature_list):
        self.feature_list = feature_list

    def set_use_backward_selection(self, backward_selection):
        self.use_backward_selection = backward_selection

    def set_use_interactions(self, interactions):
        self.use_interactions = interactions

    def add_interactions(self, idx_1, idx_2):
        self.kept_interactions = np.c_[self.kept_interactions, [idx_1, idx_2]]
    
    def set_param(self, idx, param):
        if self.names[idx-1] == 'gamma':
            self.gamma = param
        elif self.names[idx-1] == 'lambda':
            self.lambda_ = param
        else:
            print('Wrong name for the parameters to test, need to set lambda or gamma')
    
    def set_best_gamma(self, gamma):
        self.best_gamma = gamma

    def set_best_lambda(self, lambda_):
        self.best_lambda = lambda_

    def set_best_param(self, idx, param):
        if self.names[idx-1] == 'gamma':
            self.best_gamma = param
        elif self.names[idx-1] == 'lambda':
            self.best_lambda = param
        else:
            print('Wrong name for the parameters to test, need to set lambda or gamma')
            
    def set_best_degree(self,degree):
        self.best_degree = degree
    
    def best_param(self, idx):
        if self.names[idx-1] == 'gamma':
            return self.best_gamma
        elif self.names[idx-1] == 'lambda':
            return self.best_lambda
        else:
            print('Wrong name for the parameters to test, need to set lambda or gamma')
    
    def range(self, idx):
        if self.names[idx-1] == 'gamma':
            return self.gamma_range
        elif self.names[idx-1] == 'lambda':
            return self.lambda_range
        else:
            print('Wrong name for the parameters to test, need to set lambda or gamma')




# -------------------------------------------------------------------------- #