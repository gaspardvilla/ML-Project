import math
import numpy as np

# The idea is to split a large set into a test set and a train set to see the
# correctness of our model, and optimize it.
def train_test_separator(y, data_set, spliter, seed):
    # Random indices
    rand_indices = np.arange(np.shape(data_set)[0])
    np.random.shuffle(rand_indices)
    
    # Get spliter % of our data for the train set and 100 - spliter for the test
    Split = int(spliter * np.shape(data_set)[0])
    
    # Time to split
    ind_train = rand_indices[:Split]
    train_set = data_set[ind_train]
    y_train = y[ind_train]
    
    ind_test = rand_indices[Split:]
    test_set = data_set[ind_test]
    y_test = y[ind_test]
    
    # Return the results
    return train_set, y_train, test_set, y_test

# -------------------------------------------------------------------------- #

def standardize(data_set):

    data_set = data_set - np.mean(data_set, axis=0)
    data_set = data_set / np.std(data_set, axis=0)
    
    return data_set

# -------------------------------------------------------------------------- #

def constant_feature (feature):
    constant = feature[0]
    for i in range(len(feature)):
        if (feature[i] != constant):
            return False
    return True

# -------------------------------------------------------------------------- #

def clean_constant_features(tx):
    ind_const = []
    for i in range (tx.shape[1]):
        if (constant_feature(tx[:,i]) == True):
            ind_const.append(i)     
    tx = np.delete(tx, ind_const, axis = 1)
    return tx

# -------------------------------------------------------------------------- #

def clean_correlated_features(data_set):
    '''correlation matrix'''
    C = np.corrcoef(data_set.T) # correlation coefficient between features
                          # C is a 30x30 array
                          # Cij is the correlation coefficient between feature i and feature j 
    # Select upper triangle of correlation matrix
    C = np.triu(C, k=0)

    '''get rid of features with correlation coefficient > 0.95'''
    threshold = 0.95
    corr_features = []
    for i in range(len(C[0]) - 1):
        for j in range(len(C[0]) - 1):
            if ((i != j) and C[i][j] > threshold):
                corr_features.append([i, j])
    # if there is no correlated features, return the data set
    if (not corr_features):
        return data_set
    corr_feat_to_delete = [corr_features[0][0]]
    for i in range (1, len(corr_features) - 1):
        if (corr_features[i][0] != corr_features[i-1][0]):
            corr_feat_to_delete.append(corr_features[i][0])

    data_set= np.delete(data_set, corr_feat_to_delete, axis = 1) 
    return data_set

# -------------------------------------------------------------------------- #

def indices_classification(data_set):
    # Put the data in one of the four classes in function of the value of feature 23
    
    # Direct computation
    ind_class_0 = np.array(np.where(data_set[:,22] == 0)[0])
    ind_class_1 = np.array(np.where(data_set[:,22] == 1)[0])
    ind_class_2 = np.array(np.where(data_set[:,22] == 2)[0])
    ind_class_3 = np.array(np.where(data_set[:,22] == 3)[0])
    
    # Return the results
    return ind_class_0, ind_class_1, ind_class_2, ind_class_3

# -------------------------------------------------------------------------- #

def classification(data_set):
    # Put the data in one of the four classes in function of the indices
    
    # Direct computation
    ind_class_0, ind_class_1, ind_class_2, ind_class_3 = indices_classification(data_set)
    class_0 = data_set[ind_class_0]
    class_1 = data_set[ind_class_1]
    class_2 = data_set[ind_class_2]
    class_3 = data_set[ind_class_3]
    
    # Return the results
    return class_0, class_1, class_2, class_3

# -------------------------------------------------------------------------- #

def y_classification(y, data_set):
    #return the y associated to each class

    # Direct computation
    ind_class_0, ind_class_1, ind_class_2, ind_class_3 = indices_classification(data_set)
    y_0 = y[ind_class_0]
    y_1 = y[ind_class_1]
    y_2 = y[ind_class_2]
    y_3 = y[ind_class_3]
    
    return y_0, y_1, y_2, y_3

# -------------------------------------------------------------------------- #

def ind_features_to_delete_class_0(class_0):
    '''
    delete 12 features that shouldn't be taken into account for the model for 
    class 0:
    
    DER_deltaeta_jet_jet = feature 4
    DER_mass_jet_jet = feature 5
    DER_prodeta_jet_jet = feature 6
    DER_pt_tot = feature 8
    DER_sum_pt = feature 9
    DER_lep_eta_centrality = feature 12
    PRI_jet_leading_pt = feature 23
    PRI_jet_leading_eta = feature 24
    PRI_jet_leading_phi = feature 25
    PRI_jet_subleading_pt = feature 26
    PRI_jet_subleading_eta = feature 27
    PRI_jet_subleading_phi = feature 28
    '''
    ind_features_to_delete = np.array([4, 5, 6, 8, 9, 12, 23, 24 , 25, 26, 27, 28]) 
    return ind_features_to_delete
    
# -------------------------------------------------------------------------- #

def ind_features_to_delete_class_1(class_1):
    '''
    delete 7 features that shouldn't be taken into account for the model for 
    class 0:
    
    DER_deltaeta_jet_jet = feature 4
    DER_mass_jet_jet = feature 5
    DER_prodeta_jet_jet = feature 6
    DER_lep_eta_centrality = feature 12
    PRI_jet_subleading_pt = feature 26
    PRI_jet_subleading_eta = feature 27
    PRI_jet_subleading_phi = feature 28
    ''' 
    ind_features_to_delete = np.array([4, 5, 6, 12, 26, 27, 28]) 
    return ind_features_to_delete

# -------------------------------------------------------------------------- #

def ind_features_to_delete_class_3(class_3):
    '''
    delete 1 feature that shouldn't be taken into account for the model for 
    class 0:
    
    DER_pt_tot = feature 8
    ''' 
    ind_features_to_delete = np.array([8])     
    return ind_features_to_delete
    
# -------------------------------------------------------------------------- #

def clear_features(class_, k=2):
    if k == 0 :
        class_ = np.delete(class_, ind_features_to_delete_class_0(class_), axis = 1)
    if k == 1 :
        class_ = np.delete(class_, ind_features_to_delete_class_1(class_), axis = 1) 
    if k == 3 :
        class_ = np.delete(class_, ind_features_to_delete_class_3(class_), axis = 1)
        
    return class_

# -------------------------------------------------------------------------- #

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

# -------------------------------------------------------------------------- #

def treating_outliers(feature):
    '''replace outliers' values by the median'''
    indices = np.arange(0, len(feature))
    ind_out = indices_outliers(feature)
    ind_in = np.delete(indices, ind_out)
    ind_in = ind_in.reshape(-1)
    f_median = np.percentile(feature[ind_in], 50)
    feature[ind_out] = f_median
    # return feature with outliers' values replaced by the median (without outliers' values)
    return feature

# -------------------------------------------------------------------------- #
''' comme clean_test_set et clean_train_test font la même chose mais juste sur
 des data set differents je me suis dit que c'était sûrement mieux de faire 
 juste une fonction et on lui passe le train ou le test set en argument '''
def clean_set(data_set):
    # Loop over all the columns of the test set
    for i in range (data_set.shape[1]):
        data_set[:,i] = treating_outliers(data_set[:,i])
    
    # Return the cleaned data set
    return data_set

# -------------------------------------------------------------------------- #

def EDA(data_set):
    ''' do all the EDA for a specific data set'''
    # clean constant features
    data_set = clean_constant_features(data_set)
    
    # clean outliers
    data_set = clean_set(data_set)
    
    # standardization
    data_set = standardize(data_set)
    
    # clean correlated features
    data_set = clean_correlated_features(data_set)
    
    return data_set

# -------------------------------------------------------------------------- #

def EDA_class(data_set):
    # Split the data set into classes in function of the value of feature 23 but they still have all the 30 features
    class_0, class_1, class_2, class_3 = classification(data_set)
    
    # Delete features in function of the class
    class_0 = clear_features(class_0, k = 0)
    class_1 = clear_features(class_1, k = 1)

    # ? No clean for class 2 ?

    class_3 = clear_features(class_3, k = 3)
    
    # EDA for each class
    class_0 = EDA(class_0)
    class_1 = EDA(class_1)
    class_2 = EDA(class_2)
    class_3 = EDA(class_3)
    
    return class_0, class_1, class_2, class_3

# -------------------------------------------------------------------------- #