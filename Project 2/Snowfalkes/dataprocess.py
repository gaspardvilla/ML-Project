"""
In this file, there will be all the functions used for the data
pre-processing, the cleaning of the data set and the normalization.
"""


# ------------------------------------------------------------------ #
# IMPORTS
# ------------------------------------------------------------------ #


import pandas as pd
import os
from tqdm import tqdm

from sklearn.preprocessing import *

tqdm.pandas()


# ------------------------------------------------------------------ #
# Main function
# ------------------------------------------------------------------ #


def processing(data_set, classes, classifier):

    # Copy the dataframes
    X = data_set.copy()
    y = classes.copy()

    # Remove all the columns that are not interesting for us
    X, y = column_remover(X, y)

    # Old method
    X, y = clean_perso(X, y, classifier)

    return X, y


# ------------------------------------------------------------------ #
# 1st order
# ------------------------------------------------------------------ #


def column_remover(X, y):
    # Get the columns to delete for our experiences
    black_list_words = ['roi', 'riming', 'melting', 'snowflake', 'hl']
    cols_to_delete = list(filter(lambda cols: any(word in cols for word in black_list_words), X.columns))
    cols_to_delete.extend(['datetime', 'pix_size', 'flake_number_tmp', 'cam_id'])
    X = X.drop(cols_to_delete, axis = 1)
    return X, y
    

# ------------------------------------------------------------------ #


def clean_perso(X, y, classifier):

    # Remove miss classification
    y = remove_wrong_classifications(y, classifier)

    # Get all the data where we have right classification
    X = select_right_classification(X, y)

    # Join the class id for the data
    X, y = join_classes(X, y)


    # Standardization of the data set
    X = standardization(X)

    return X, y


# ------------------------------------------------------------------ #
# 2nd order
# ------------------------------------------------------------------ #


def remove_wrong_classifications(y, classifier):

    # Get all the duplicates having the same flake id in y
    all_duplicates = y[y.duplicated(subset = ['flake_id'], 
                                                    keep = False)]

    # Get all the miss classification
    wrong_classification = all_duplicates.drop_duplicates(subset = ['flake_id', 'class_id'], 
                                                            keep = False)
    save_wrong_classifications(wrong_classification, classifier)

    # Get the flake id of the wrong duplicates
    wrong_classification_id = wrong_classification.drop_duplicates(subset = ['flake_id'], 
                                                    keep = 'first')

    # Get all the flake id with y
    classes_id = y.drop_duplicates(subset = ['flake_id'], 
                                                    keep = 'first')

    # Remove the wrong flake id from all the flake id
    cleaned_classes_id = pd.concat([classes_id, wrong_classification_id]).drop_duplicates(subset = ['flake_id'], 
                                                                            keep = False)
    y = y[y.flake_id.isin(cleaned_classes_id.flake_id)].reset_index(drop = True)

    # Return teh result
    return y


# ------------------------------------------------------------------ #


def select_right_classification(X, y):

    X_transformed = X[X.flake_id.isin(y.flake_id)].reset_index(drop = True)
    X_to_keep = X_transformed.apply(lambda row: keep_right_row(row, y), axis = 1)
    X_transformed = X_transformed[X_to_keep]

    return X_transformed


# ------------------------------------------------------------------ #


def join_classes(X, y):

    y = y[y.flake_id.isin(X.flake_id)]

    y = y.drop_duplicates(subset = 'flake_id', keep = 'first')
    y = y.drop('cam', axis = 1).set_index('flake_id', drop = True)

    X = X.drop('cam', axis = 1).set_index('flake_id', drop = True)

    X = X.join(y)

    y = pd.DataFrame(X['class_id'])
    X = X[X.columns.difference(['class_id'])]

    return X, y


# ------------------------------------------------------------------ #


def standardization(X):

    mascdb_data_modified_copy = X.copy()
    power_transformer = PowerTransformer(method = 'yeo-johnson', standardize = True)
    X_transformed = power_transformer.fit_transform(mascdb_data_modified_copy)

    # Set the transformed data
    X[X.columns] = X_transformed

    return X


# ------------------------------------------------------------------ #
# 3rd order
# ------------------------------------------------------------------ #


def save_wrong_classifications(wrong_classification, classifier):
    wrong_classification = wrong_classification.sort_values(by = ['flake_id'])
    path = 'Data/wrong_classifications'
    path = os.path.join(path, classifier+'.pkl')
    wrong_classification.to_pickle(path)


# ------------------------------------------------------------------ #


def keep_right_row(row, y):

    if row.cam in y[y.flake_id == row.flake_id].cam.to_list():
        return True
    else:
        return False


# ------------------------------------------------------------------ #