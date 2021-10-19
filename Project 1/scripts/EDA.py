import math
import numpy as np

# The idea is to split a large set into a test set and a train set to see the
# correctness of our model, and optimize it.
def train_test_separator(truth, large_set, spliter, seed):
    # Random indices
    rand_indices = np.arange(np.shape(large_set)[0])
    np.random.shuffle(rand_indices)
    
    # Get spliter % of our data for the train set and 100 - spliter for the test
    Split = int(spliter * np.shape(large_set)[0])
    
    # Time to split
    ind_train = rand_indices[:Split]
    train_set = large_set[ind_train]
    train_truth = truth[ind_train]
    
    ind_test = rand_indices[Split:]
    test_set = large_set[ind_test]
    test_truth = truth[ind_test]
    
    # Return the results
    return train_set, train_truth, test_set, test_truth



def standardize(data_set):
    data_set = data_set - np.mean(data_set, axis=0)
    data_set = data_set / np.std(data_set, axis=0)
    return data_set



def categorization(data_set):
    # Direct computation
    cat_A = np.array(np.where(data_set[:,22] == 0)[0])
    cat_B = np.array(np.where(data_set[:,22] == 1)[0])
    cat_C = np.array(np.where(data_set[:,22] == 2)[0])
    cat_D = np.array(np.where(data_set[:,22] == 3)[0])
    
    # Return the results
    return cat_A, cat_B, cat_C, cat_D




def indices_outliers(feature):
    '''find outliers indices''' 
    Q1 = np.percentile(feature, 25)
    Q3 = np.percentile(feature, 75)
    IQR = Q3 - Q1 # interquartile range
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    ind_out = []
    for idx in range (len(feature)):
        if ((feature[idx] < lower_bound) or (feature[idx] > upper_bound)):
            ind_out.append(idx)
    # return the indices of the feature that contain outliers
    return ind_out     


def treating_outliers(feature):
    '''replace outliers' values by the mean'''
    indices = np.arange(0, len(feature))
    ind_out = indices_outliers(feature)
    ind_in = np.delete(indices, ind_out)
    ind_in = ind_in.reshape(-1)
    f_mean = feature[ind_in].mean()
    feature[ind_out] = f_mean
    # return feature with outliers' values replaced by the mean (without outliers' values)
    return feature



def clean_train_set(train_set):
    # Loop over all the columns of the train set
    for i in range (train_set.shape[1]):
        train_set[:,i] = treating_outliers(train_set[:,i])
    
    # Return the cleaned train set
    return train_set



def clean_test_set(test_set):
    # Loop over all the columns of the test set
    for i in range (test_set.shape[1]):
        test_set[:,i] = treating_outliers(test_set[:,i])
    
    # Return the cleaned test set
    return test_set