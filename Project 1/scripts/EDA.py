import math
import numpy as np



# -------------------------------------------------------------------------- #

def standardize(data_set):
    """
        This function standardizes the data set.

    Args:
        data_set: the considered data set

    Returns:
        data_set: The standardized data set.
    """

    # Direct standardization
    data_set = data_set - np.mean(data_set, axis=0)
    data_set = data_set / np.std(data_set, axis=0)
    
    # Returned the standardized data set
    return data_set

# -------------------------------------------------------------------------- #

def constant_feature(feature):
    """
        This function indicates wether this feature is constant.


    Args:
        feature: The considered feature.

    Returns:
        It returns True if the considered feature is constant.
    """
    
    # Initialization of the constant term
    constant_term = feature[0]

    # Loop over all the elements in the feature
    for i in range(len(feature)):
        if (feature[i] != constant_term):
            return False
    return True

def remove_feature(data_set, idx):
    """
        This function just removes the feature indicated by the indices in 'idx'
        from the considered data set.

    Args:
        data_set: The considered data set
        idx: the indices of the features to remove from tha data set

    Returns:
        data_set_: The updated data set where the feature were removed.
    """

    # Remove the features in indices from the data set
    data_set_ = np.delete(data_set, idx, axis = 1)

    # Return the updated data set
    return data_set_

# -------------------------------------------------------------------------- #

def clean_constant_features(data_set):
    """
        This function removes the possible features from the data set that 
        seems to be constant.

    Args:
        data_set: The considered data set

    Returns:
        [type]: the upadted data set where all the constant feature were removed
    """

    # Initialization of the constant features indices
    ind_const = []

    # Check which feature should be removed
    for i in range(data_set.shape[1]):
        if (constant_feature(data_set[:,i]) == True):
            ind_const.append(i)

    # Remove the constant features from the data set
    data_set_ = remove_feature(data_set, ind_const)

    # Return the updated data set 
    return data_set_

# -------------------------------------------------------------------------- #

def clean_correlated_features(data_set):
    """
        This function removes all the correlated features from the considered 
        data set. The idea is to keep one feature among three correlated feature, 
        instead of keeping the three.


    Args:
        data_set: the considered data set

    Returns:
        returned_set: the udpdated datacset without any correlated feature
    """

    corr_matrix = np.corrcoef(data_set.T) # correlation coefficient between features
                          # C is a 30x30 array
                          # Cij is the correlation coefficient between feature i and feature j 
    
    # Initialization
    threshold = 0.95
    correlated_features = []

    # Loop over all the elements of the correlation matrix (symmetric matrix 
    # and we do not consider the diagonal).
    for i in range(corr_matrix.shape[0]):
        for j in range(i+1, corr_matrix.shape[0]):
            if np.abs(corr_matrix[i, j]) > threshold:
                correlated_features.append([i, j])
    
    # If there is no correlated features, return directly the data set
    if (not correlated_features):
        return data_set

    # Select all the correlated feature to remove (without removing everyone)
    corr_feat_to_delete = [correlated_features[0][0]]
    for i in range(1, len(correlated_features)):
        if (correlated_features[i][0] != correlated_features[i-1][0]):
            corr_feat_to_delete.append(correlated_features[i][0])

    # Remove the correlated features
    returned_set = remove_feature(data_set, corr_feat_to_delete) 

    # Return the data set
    return returned_set

# -------------------------------------------------------------------------- #

def indices_classification(data_set):
    """
        This function classifies the data set into 4 distinct data sets given by
        the value of the feature 23.

    Args:
        data_set: The considered data set.
    
    Returns:
        ind_class_0, ind_class_1, ind_class_2, ind_class_3: the indices where each
        observation belongs to one of the 4 classes.
    """
    
    # Direct computation
    ind_class_0 = np.array(np.where(data_set[:,22] == 0)[0])
    ind_class_1 = np.array(np.where(data_set[:,22] == 1)[0])
    ind_class_2 = np.array(np.where(data_set[:,22] == 2)[0])
    ind_class_3 = np.array(np.where(data_set[:,22] == 3)[0])
    
    # Return the indices
    return ind_class_0, ind_class_1, ind_class_2, ind_class_3

# -------------------------------------------------------------------------- #

def classification(data_set):
    """
        This function splits the considered data set into 4 distinct data given by
        the four possible values in the feature 23.

    Args:
        data_set: The considered data set.

    Returns:
        class_0, class_1, class_2, class_3: the four data set (we call them 'classes')
    """
    # Put the data in one of the four classes in function of the indices
    
    # Direct computation
    ind_class_0, ind_class_1, ind_class_2, ind_class_3 = indices_classification(data_set)

    # Split the data set
    class_0 = data_set[ind_class_0]
    class_1 = data_set[ind_class_1]
    class_2 = data_set[ind_class_2]
    class_3 = data_set[ind_class_3]
    
    # Return the results
    return class_0, class_1, class_2, class_3

# -------------------------------------------------------------------------- #

def y_classification(y, data_set):
    """
        This function splits the observation set y into small observation set, 
        corresponding to the split given by the data set.

    Args:
        y: teh considered observation set
        data_set: the considered data set

    Returns:
        y_0, y_1, y_2, y_3: the 4 splitted observations set
    """
    #return the y associated to each class

    # Direct computation
    ind_class_0, ind_class_1, ind_class_2, ind_class_3 = indices_classification(data_set)

    # Split the observation set y
    y_0 = y[ind_class_0]
    y_1 = y[ind_class_1]
    y_2 = y[ind_class_2]
    y_3 = y[ind_class_3]
    
    # Return the four observations set
    return y_0, y_1, y_2, y_3

# -------------------------------------------------------------------------- #

def ind_features_to_delete_class_0(class_0):
    '''
    delete 12 features that shouldn't be taken into account for the model for 
    class 0 (see paper):
    
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
    class 0 (see paper):
    
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
    class 0 (see paper):
    
    DER_pt_tot = feature 8
    ''' 
    ind_features_to_delete = np.array([8])     
    return ind_features_to_delete
    
# -------------------------------------------------------------------------- #

def clear_features(data_set, k=2):
    """
        This function removes the feature that are not defined for each classes,
        indicated in the paper that produce this data set.

    Args:
        data_set : the considered class 
        k: indicates which class we are working on. Defaults to 2.

    Returns:
        data_set: the updated class.
    """

    # Check which class we are working on
    if k == 0 :
        data_set = np.delete(data_set, ind_features_to_delete_class_0(data_set), axis = 1)
    if k == 1 :
        data_set = np.delete(data_set, ind_features_to_delete_class_1(data_set), axis = 1) 
    if k == 3 :
        data_set = np.delete(data_set, ind_features_to_delete_class_3(data_set), axis = 1)
        
    # Remove feature 0: too many undefined values in all classes
    data_set = remove_feature(data_set, [0])

    # Return the updted class
    return data_set

# -------------------------------------------------------------------------- #

def indices_outliers(feature):
    """
        This function finds the outlier indices in the considered feature. The 
        idea is to take all the outliers outside the interval 'mean +/- IQR' and
        set them to the median

    Args:
        feature: the considered feature.

    Returns:
        ind_out: the indices were the outliers are in the considered feature.
    """

    # Set the Interquartile Range (IQR)
    Q1 = np.percentile(feature, 25)
    Q3 = np.percentile(feature, 75)
    IQR = Q3 - Q1

    # Initialization
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    ind_out = []

    # Loop over all the values in the considered feature
    for idx in range(len(feature)):
        if ((feature[idx] < lower_bound) or (feature[idx] > upper_bound)):
            ind_out.append(idx)
    
    # return the indices of the feature that contain outliers
    return ind_out     

# -------------------------------------------------------------------------- #

def treating_outliers(feature):
    """
        This function set all the outliers in the considered feature to the median
        (considered without the outliers).

    Args:
        feature: The considered feature.

    Returns:
        feature: the updated feature
    """
    
    # Direct computation
    indices = np.arange(0, len(feature))
    ind_out = indices_outliers(feature)
    ind_in = np.delete(indices, ind_out)
    ind_in = ind_in.reshape(-1)
    f_median = np.percentile(feature[ind_in], 50)
    feature[ind_out] = f_median

    # return feature with outliers' values replaced by the median
    return feature

# -------------------------------------------------------------------------- #

def clean_set(data_set):
    """
        This function treat with the outliers over all the features in the 
        considered data set.

    Args:
        data_set: the considered data set  

    Returns:
        data_set: the updated data set
    """

    # Loop over all the feature of the data set
    for i in range (data_set.shape[1]):
        data_set[:,i] = treating_outliers(data_set[:,i])
    
    # Return the cleaned data set
    return data_set

# -------------------------------------------------------------------------- #

def EDA(data_set):
    ''' do all the EDA for a specific data set'''

    # clean constant features
    data_set = clean_constant_features(data_set)
    
    # log(1+x) filter
    data_set = log_filter(data_set)

    # clean outliers
    data_set = clean_set(data_set)
    
    # standardization
    data_set = standardize(data_set)
    
    # clean correlated features
    data_set = clean_correlated_features(data_set)
    
    return data_set
# -------------------------------------------------------------------------- #

def graph_analysis_removal(class_0, class_1, class_2, class_3):
    '''Remove all the feature we found useless in the classification with the
    graph analysis done over all the classes.'''

    # Class 0
    class_0 = remove_feature(class_0, [2, 5, 7, 8, 9, 10, 11, 13, 14])
    # 7 9 10 14

    # Class 1
    class_1 = remove_feature(class_1, [4, 7, 10, 11, 13, 14, 15, \
        16, 17, 18, 19])
    # 2 3 7 9 12 16 17 19

    # Class 2
    class_2 = remove_feature(class_2, [3, 9, 11, 12, 15, 16, 18, \
        19, 20, 21])

    # Class 3
    class_3 = remove_feature(class_3, [2, 7, 8, 9, 10, 13, 14, \
        16, 17, 18, 19, 20, 21])
    # 0, 2, 4, 6, 8, 9, 12, 15, 21, 24

    # Return the classes 
    return class_0, class_1, class_2, class_3

# -------------------------------------------------------------------------- #

def log_filter(data_set):
    '''This function applies a 'log(1+|x|) * sgn(x)' transformation to all the 
    features.'''

    # Loop over all the features of the data set
    for idx in range(data_set.shape[1]):
        data_set[:, idx] = np.multiply(np.sign(data_set[:, idx]), np.log1p(np.abs(data_set[:, idx])))
    
    # Return the updated data set
    return data_set


# -------------------------------------------------------------------------- #

def EDA_class(data_set):
    ''' This function applies everything we need for our Exploratory Data 
    Analysis (EDA) including the classification of our data set.'''

    # Split the data set into classes in function of the value of feature 23 but they still have all the 30 features
    class_0, class_1, class_2, class_3 = classification(data_set)
    
    # Delete features in function of the class
    class_0 = clear_features(class_0, k = 0)
    class_1 = clear_features(class_1, k = 1)
    class_3 = clear_features(class_3, k = 3)

    # EDA for each class
    class_0 = EDA(class_0)
    class_1 = EDA(class_1)
    class_2 = EDA(class_2)
    class_3 = EDA(class_3)

    # Remove features from graph analysis
    class_0, class_1, class_2, class_3 = graph_analysis_removal(class_0, class_1, class_2, class_3)
    
    # Return the upadted classes
    return class_0, class_1, class_2, class_3

# -------------------------------------------------------------------------- #