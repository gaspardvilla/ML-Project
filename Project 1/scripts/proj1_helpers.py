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
