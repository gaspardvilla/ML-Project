# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
from EDA import *


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


def counting_errors(pred_set, true_set):
    # Count the number of true predictions
    N = np.shape(pred_set)[0]
    Nb_errors = np.count_nonzero(pred_set != true_set)
    
    print("Numbers of errors : ", Nb_errors, " // Error accuracy [%] : %", (Nb_errors / N) * 100)
    

def rebuild_y(y_0,y_1,y_2,y_3,data_set):
    rb_y = np.zeros(data_set.shape[0])
    
    ind_class_0, ind_class_1, ind_class_2, ind_class_3 = indices_classification(data_set)
    
    rb_y[ind_class_0] = y_0
    rb_y[ind_class_1] = y_1
    rb_y[ind_class_2] = y_2
    rb_y[ind_class_3] = y_3
    
    return rb_y

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