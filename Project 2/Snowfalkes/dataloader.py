"""
In this file, there will be all the functions used for the data
loading in dataframes format.
"""

# ------------------------------------------------------------------ #
# Imports
# ------------------------------------------------------------------ #

import pandas as pd
import os
import pickle
import gdown


# ------------------------------------------------------------------ #
# Main functions
# ------------------------------------------------------------------ #


def load_data_sets(classifier = 'hydro'):
    """
    The idea of this function is to load the data sets from the 3 cams,
    and the classses depending on the classifier we want.

    Args:
        classifier (str, optional): classification we want to determine 
        the classses data we will have. Defaults to 'hydro'.

    Returns:
        [Dataframes]: two data frames corresponding to the data from the
        3 cams and the dataframe containig the labels classification.
    """

    # Get the data
    data_set = get_MASCDB_data()

    # Get the classification
    classes = get_classes(classifier)
    
    return data_set, classes


# ------------------------------------------------------------------ #


def load_model(classifier, model):
    """
    Load the pre-trained model

    Args:
        classifier (str): classifier
        model (str): name of the wanted model

    Returns:
        model: return a pre-trained model
    """

    # File name
    file_name = str(classifier) + '_' + str(model) + '.pkl'

    # Initialization of the path
    path = 'Models/'
    path = os.path.join(path, 'trained_model')
    path = os.path.join(path, file_name)

    # Download the file from Google drive if it doesn't exists
    if not os.path.isfile(path):
        try:
            download(path, classifier)
        except:
            raise ValueError("Check the format of classifier or model")

    # Return the model
    return pickle.load(open(path, 'rb'))


# ------------------------------------------------------------------ #


def load_selected_features(path, X, method):
    """ 
    Load the selected features from disk

    Args : 
        path : path of the file to load
        X : initial dataset on which feature selection will be applied
        method : method used for feature selection (string)

    Returns :
        Data with the selected features
    """
    if method == 'PCA':
        components = pickle.load(open(path, 'rb'))
        return components.transform(X)
    else:
        selected_features = pickle.load(open(path, 'rb'))
        return X[selected_features]


# ------------------------------------------------------------------ #


def load_wrong_classification(classifier):
    """
    This fucntion loads the wrong classifications depending on the 
    classifier.

    Args:
        classifier (str): classifier

    Raises:
        ValueError: classifier is either 'riming' or 'hydro'

    Returns:
        [type]: [description]
    """

    # File name
    file_name = str(classifier) + '_wrong_classifications.pkl'

    # Initialization of the path
    path = 'Data/'
    path = os.path.join(path, 'wrong_classifications')
    path = os.path.join(path, file_name)

    print(path)

    # Download the file from Google drive if it doesn't exists
    if not os.path.isfile(path):
        try:
            download(path, classifier)
        except:
            raise ValueError("classifier is either 'riming' or 'hydro'")

    # Return the model
    return pd.read_pickle(path)


# ------------------------------------------------------------------ #
# 1st order
# ------------------------------------------------------------------ #


def get_classes(classifier):
    """
    This function get the human labeling classification depending of 
    the classifier we want to look at.

    Args:
        classifier (str): classifier we want to look at.

    Raises:
        ValueError: In case the clqssifier is not 'riming' ro 'hydro'.

    Returns:
        [Dataframe]: Human labeling classification.
    """

    # Initialization of the dataframe
    classes = pd.DataFrame()

    # Path to get the data
    path = 'Data/'
    path = os.path.join(os.getcwd(), path)

    # Hydrometeor classification
    if classifier == 'hydro':

        # Get classes from all cameras
        for cam in range(3):

            # Update the path of the file of the i-th cam
            path_cam = os.path.join(path, str(classifier)+'_trainingset/hydro_trainingset_cam'+str(cam)+ '.pkl')

            # Download the file from Google drive if it doesn't exists
            if not os.path.isfile(path_cam):
                download(path, classifier)
            
            # Read the pickle file
            classes_cam = pd.read_pickle(path_cam).reset_index(drop = True)

            # Add a label cam for each data
            classes_cam['cam'] = cam

            # Concatenate the current dataframe with the general one
            classes = pd.concat([classes, classes_cam])
    
    # Riming classification
    elif classifier == 'riming':

        # Get classes from all cameras
        for cam in range(3):

            # Update the path of the file of the i-th cam
            path_cam = os.path.join(path, str(classifier)+'_trainingset/riming_trainingset_cam'+str(cam)+ '.pkl')

            # Download the file from Google drive if it doesn't exists
            if not os.path.isfile(path_cam):
                download(path, classifier)
            
            # Read the pickle file
            classes_cam = pd.read_pickle(path_cam).reset_index(drop = True)

            # Add a label cam for each data
            classes_cam['cam'] = cam
            
            # Concatenate the current dataframe with the general one
            classes = pd.concat([classes, classes_cam])
        
        # Rename the column 'riming_id' in 'class_id' to be robust with
        # the code that follows.
        classes = riming_pre_process_classes(classes)
        
    # Wrong format
    else:
        raise ValueError("The string classifier must be 'hydro' or \
                                            'riming'.")

    # Return the final result
    return classes


# ------------------------------------------------------------------ #


def get_MASCDB_data():
    """
    Get all the numeric data from the MASCDB data base from the 3 cams.

    Returns:
        [Dataframe]: All the numeric data in a dataframe
    """

    # Initialization of the dataframe
    data_set = pd.DataFrame()

    # Set the path to get the data
    path = 'Data/'
    path = os.path.join(os.getcwd(), path)

    # Get data from all cameras
    for cam in range(3):

        # Update the path of the file of the i-th cam
        path_cam = os.path.join(path, 'MASCDB/MASCdb_cam'+str(cam)+ '.parquet')

        # Download the file from Google drive if it doesn't exists
        if not os.path.isfile(path_cam):
            download(path, 'MASCDB')
        
        # Read the parquet file
        data_set_cam = pd.read_parquet(path_cam).reset_index(drop = True)

        # Rename the column 'cam_id' to 'cam' for robustness with
        # classes dataframe.
        data_set_cam = data_set_cam.drop('cam_id', axis = 1)
        data_set_cam['cam'] = cam

        # Concatenate the current dataframe with the general one
        data_set = pd.concat([data_set, data_set_cam])

    # Return the final result
    return data_set


# ------------------------------------------------------------------ #
# 2nd order
# ------------------------------------------------------------------ #


def riming_pre_process_classes(classes):
    """
    This function just renames the column 'riming_id' into 'class_id'
    for robustness between the classes of hydrometeor.

    Args:
        classes (Dataframe): Classes (from riming normally)

    Returns:
        [Dataframe]: Return classes with renamed column 'class_id'
    """
    
    # Rename column
    classes['class_id'] = classes.riming_id
    classes = classes.drop('riming_id', axis = 1)

    # Return the final result
    return classes


# ------------------------------------------------------------------ #


def download(path, category):
    """Download the respective dataset from Google Drive.

    Args:
        path (string): Path to the dataset.
        category (string): category from the dictonnary
    """

    # Get the file name of what we are looking for
    filename = os.path.split(path)[-1]

    # Get the dictionnary of all the file adresses from our google drive
    files = get_drive_dictionnary()

    # Set the URL of the google drive
    url = f'https://drive.google.com/uc?id={files[category][filename]}'
    gdown.download(url, path, quiet=False)


# ------------------------------------------------------------------ #
# 3rd order
# ------------------------------------------------------------------ #


def get_drive_dictionnary():
    """
    Get the dictionnary containing all the keys to access the files on 
    a Google Drive we need to downlaod.

    Returns:
        files: Dictonnary of all the files
    """

    # Dictionnary
    files = {
        'hydro': {
            'hydro_trainingset_cam0.pkl': '1z6FAPAT0H7xqOLtxbYXKJgrrX8Rt4oGy',
            'hydro_trainingset_cam1.pkl': '1amwSErcW31Atos7JBW8BS4PwFu7qiT8c',
            'hydro_trainingset_cam2.pkl': '1hyeTsXppiktAZyNYc83xs2OVW9GODVLf',
            'hydro_MLR.pkl': '1O33XhdjBOSpLxS_0UqkicVAUbmoHf-g9',
            'hydro_SVM.pkl': '1kwXJ3IDg2QbrZkDufieyk0cF_ZwdmEgJ',
            'hydro_SVM_poly.pkl': '1tZ84GaxDMvurD1bQl59d3YfbPxljpcpA',
            'hydro_RF.pkl': '1Vp6gMQv_jeVLVygon1vt0FoMQeP24yx2',
            'hydro_MLP.pkl': '1ApXv-DQzzBgYfclINX2EjUpwgihp0TPC',
            'hydro_wrong_classifications.pkl': '1m8ohaedQh3S2MIBK3LkeC35RG2y02XvT'
        },

        'riming': {
            'riming_trainingset_cam0.pkl': '1p64qrIyB9iedRzZSX2A-CnQSC57a2vdj',
            'riming_trainingset_cam1.pkl': '1xfhOP3-Hss97WeHqNwNio8Cvgyi9g0t6',
            'riming_trainingset_cam2.pkl': '1q7U-BCLBdHfqe_q0-b0S7Iw3QuiCfOrC',
            'riming_MLR.pkl': '1Oe0GcZcOwqKuIVcGbSZzscI0uWvm4TIz',
            'riming_SVM.pkl': '18ChTxFmB7T3Gin2_JKIjYjxlx09daJfE',
            'riming_SVM_poly.pkl': '1VPQyTDkyKWBwAhM16R2fnf0zLfuEIoY3',
            'riming_RF.pkl': '1deAn7p6sRP-HZcvD2bL7gzMtq3rFsBvE',
            'riming_MLP.pkl': '1YfQw9dUSnn4WK86p5-_TUD4UvELW00XE',
            'riming_wrong_classifications.pkl': '1pEHEImhnlrh30LatK4lOKkrEiQrhJJNq'
        },

        'MASCDB': {
            'MASCdb_cam0.parquet': '1DHfFaf1GtkuCEAB9b3vkMyBOrGaugK15',
            'MASCdb_cam1.parquet': '1qyj7lgjRr9hznfXATrVqwhf44J7_91Fr',
            'MASCdb_cam2.parquet': '1oUEXfi6Bo6vbePclAdGRHmkpqdS_VUs3',
            'MASCdb_triplet.parquet': '18JO1BHy2bUU-Esv9DHKvYSmv7VBGg9qp'
        },

        'classification.pkl': '1HM0XwrX91OuXDR1CVU80e1yp3g6rvaHm'
    }

    # Return the final files
    return files


# ------------------------------------------------------------------ #