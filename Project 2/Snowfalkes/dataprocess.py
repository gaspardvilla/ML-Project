"""
In this file, there will be all the functions used for the data
pre-processing, the cleaning of the data set and the normalization.
"""


# ------------------------------------------------------------------ #
# Imports
# ------------------------------------------------------------------ #


import pandas as pd
import os
from tqdm import tqdm

from sklearn.preprocessing import *

# For progress bar when progress_apply function is used on dataframes
tqdm.pandas()


# ------------------------------------------------------------------ #
# Main function
# ------------------------------------------------------------------ #


def processing(data_set, classes, classifier):
    """
    The idea of this function is to give the data set and all the 
    classsification's data we have and clean them for a merge to know
    exactly what will be our dataframe. (standardization for example)

    Args:
        data_set (Dataframes): Data set
        classes (Dataframe): classfication's data
        classifier (str): classifier

    Returns:
        [Dataframes]: two dataframes that we will use for train and 
        test sets.
    """

    # Copy the dataframes
    X = data_set.copy()
    y = classes.copy()

    # Remove all the columns that are not interesting for us
    X = column_remover(X)

    # Clean the dataframes X and y
    X, y = clean(X, y, classifier)

    return X, y


# ------------------------------------------------------------------ #
# 1st order
# ------------------------------------------------------------------ #


def column_remover(X):
    """
    Remove all the columns that we do not want to keep in our data set

    Args:
        X (Dataframe): data set 
        y ([Dataframes]): classes

    Returns:
        [type]: [description]
    """

    # Get a black list of words in the name of the columns
    black_list_words = ['roi', 'riming', 'melting', 'snowflake', 'hl']

    # List all columns that have word in the previous black list
    cols_to_delete = list(filter(lambda cols: any(word in cols for word in black_list_words), X.columns))

    # Add some exterior columns
    cols_to_delete.extend(['datetime', 'pix_size', 'flake_number_tmp'])

    # Drop the columns
    X = X.drop(cols_to_delete, axis = 1)

    # Return the final result
    return X
    

# ------------------------------------------------------------------ #


def clean(X, y, classifier):
    """
    This function cleans and apply a pre-processing over all the data 
    set.

    Args:
        X (Dataframe): data set
        y (Dataframe): classes
        classifier (str): classifier

    Returns:
        [Dataframes]: Return X and y pre processed
    """

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
    """
    This function inspects the human labeling classification to see if
    some snowflakes are not well classified. Meaning that in y, we have
    one flake id on two different cam that have not the same class_id
    when it should be.

    Args:
        y (Dataframe): classification data
        classifier (str): classifier

    Returns:
        [Dataframe]: Return y but without miss classfification
    """

    # Get all the duplicates having the same flake id in y
    all_duplicates = y[y.duplicated(subset = ['flake_id'], 
                                                    keep = False)]

    # Get all the miss classification
    wrong_classification = all_duplicates.drop_duplicates(subset = ['flake_id', 'class_id'], 
                                                            keep = False)

    # Save the wrong classifications to solve the problem for future
    # studies.
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
    """
    This function selection in the data set X all the data where we have
    a good classification given by the function just before. And we also
    select the classes that we have data on it (because there is some of
    them where we do not have any data on it)

    Args:
        X (Dataframe): data set
        y (Dataframe): classification

    Returns:
        [Dataframe]: Filtered X
    """

    # Keep only the snowflakes that have their flake id in y
    X_transformed = X[X.flake_id.isin(y.flake_id)].reset_index(drop = True)

    # Check which line have its flake id in y
    X_to_keep = X_transformed.apply(lambda row: keep_right_row(row, y), axis = 1)

    # Filter X
    X_transformed = X_transformed[X_to_keep]

    # Return the data frame
    return X_transformed


# ------------------------------------------------------------------ #


def join_classes(X, y):
    """
    This function assign the class to every snowflake in the data set.

    Args:
        X (Dataframe): data set
        y (Dataframe): classification

    Returns:
        [Dataframes]: New data frames X and y ready for training 
        and testing.
    """ 

    # Keep only the classes from y that have their flake id in X
    y = y[y.flake_id.isin(X.flake_id)]

    # Drop the duplicates because useless
    y = y.drop_duplicates(subset = 'flake_id', keep = 'first')

    # Drop the cam columns from X and y
    y = y.drop('cam', axis = 1).set_index('flake_id', drop = True)
    X = X.drop('cam', axis = 1).set_index('flake_id', drop = True)

    # Join the class_id to the snowflakes in X
    X = X.join(y)

    # Rebuild y with new indices
    y = pd.DataFrame(X['class_id'])

    # Remove column 'class_id' from X
    X = X[X.columns.difference(['class_id'])]

    # Return the both data frames
    return X, y


# ------------------------------------------------------------------ #


def standardization(X):
    """
    This function applies a transform to the data to normalize them.

    Args:
        X (Dataframe): [description]

    Returns:
        [type]: [description]
    """

    # Applied the transformation (positive and negative valeus included)
    X_transformed = X.copy()
    power_transformer = PowerTransformer(method = 'yeo-johnson', standardize = True)
    X_transformed = power_transformer.fit_transform(X_transformed)

    # Set the transformed data
    X[X.columns] = X_transformed

    # Return the result
    return X


# ------------------------------------------------------------------ #
# 3rd order
# ------------------------------------------------------------------ #


def save_wrong_classifications(wrong_classification, classifier):
    """
    This function just saves the wrong classification in a pickle files
    to keep a track on them and re-check the images for the future
    studies.

    Args:
        wrong_classification (Dataframe): wrong classification dataframes
        classifier (str): classifier
    """

    wrong_classification = wrong_classification.sort_values(by = ['flake_id'])
    path = 'Data/wrong_classifications'
    path = os.path.join(path, classifier+'.pkl')
    wrong_classification.to_pickle(path)


# ------------------------------------------------------------------ #


def keep_right_row(row, y):
    """
    This function is used in the apply function to select the flake id 
    that has the cam number in the dataframe y.

    Args:
        row (row of Dataframe): row of the current dataframe where the 
        function apply is used.
        y (Dataframe): containing the cam nunmbers and the flake id

    Returns:
        [Boolean]: Return true if 'row' corresponds to a row in y, 
        and return False otherwise.
    """

    if row.cam in y[y.flake_id == row.flake_id].cam.to_list():
        return True
    else:
        return False


# ------------------------------------------------------------------ #