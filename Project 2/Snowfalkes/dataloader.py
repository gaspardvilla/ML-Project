"""
In this file, there will be all the functions used for the data
pre-processing, the cleaning of the data set and the normalization.
"""

# ------------------------------------------------------------------ #
# IMPORTS

import pandas as pd
import os
import gdown


# ------------------------------------------------------------------ #


# Here the idea is to get the hydrometeor/riming data sets and the 
# response where the data are cleaned depending on the response.
def load_data_sets(classifier = 'hydro'):

    # Get the data
    data_set = get_MASCDB_data()

    # Get the classification
    classes = get_classes(classifier)
    
    return data_set, classes


# ------------------------------------------------------------------ #


def get_classes(classifier):

    # Initialization of the dataframe
    classes = pd.DataFrame()

    path = 'Data/'
    path = os.path.join(os.getcwd(), path)

    if classifier == 'hydro':

        for cam in range(3):
            path_cam = os.path.join(path, str(classifier)+'_trainingset/hydro_trainingset_cam'+str(cam)+ '.pkl')

            if not os.path.isfile(path_cam):
                download(path, classifier)
            
            classes_cam = pd.read_pickle(path_cam).reset_index(drop = True)

            classes_cam['cam'] = cam

            classes = pd.concat([classes, classes_cam])
    
    elif classifier == 'riming':
        for cam in range(3):
            path_cam = os.path.join(path, str(classifier)+'_trainingset/riming_trainingset_cam'+str(cam)+ '.pkl')

            if not os.path.isfile(path_cam):
                download(path, classifier)
            
            classes_cam = pd.read_pickle(path_cam).reset_index(drop = True)

            classes_cam['cam'] = cam

            classes = pd.concat([classes, classes_cam])
        classes = riming_pre_process_classes(classes)
        

    else:
        raise ValueError("The string classifier must be 'hydro' or \
                                            'riming'.")

    return classes


# ------------------------------------------------------------------ #


def riming_pre_process_classes(classes):
    classes['class_id'] = classes.riming_id
    classes = classes.drop('riming_id', axis = 1)
    return classes


# ----------------------------------------------------------------- #


def get_MASCDB_data():

    # Initialization of the dataframe
    data_set = pd.DataFrame()

    path = 'Data/'
    path = os.path.join(os.getcwd(), path)

    for cam in range(3):
        path_cam = os.path.join(path, 'MASCDB/MASCdb_cam'+str(cam)+ '.parquet')

        if not os.path.isfile(path_cam):
            download(path, 'MASCDB')
        
        data_set_cam = pd.read_parquet(path_cam).reset_index(drop = True)

        data_set_cam['cam'] = cam

        data_set = pd.concat([data_set, data_set_cam])

    return data_set


# ----------------------------------------------------------------- #


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

    url = f'https://drive.google.com/uc?id={files[category][filename]}'
    gdown.download(url, path, quiet=False)


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
        },

        'riming': {
            'riming_trainingset_cam0.pkl': '1p64qrIyB9iedRzZSX2A-CnQSC57a2vdj',
            'riming_trainingset_cam1.pkl': '1xfhOP3-Hss97WeHqNwNio8Cvgyi9g0t6',
            'riming_trainingset_cam2.pkl': '1q7U-BCLBdHfqe_q0-b0S7Iw3QuiCfOrC',
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