import pandas as pd
import os
from tqdm import tqdm

from sklearn.preprocessing import *

from dataloader import load_data_sets

tqdm.pandas()


# ------------------------------------------------------------------ #


def get_processed_data(classifier = 'hydro'):

    # Directly the dataset
    data_set, classes = load_data_sets(classifier = classifier)

    # Remove all the columns that are not interesting for us
    data_set, classes = column_remover(data_set, classes)

    data_set['cam_id'] = data_set.cam
    data_set = data_set.set_index('flake_id').drop('cam', axis = 1)
    data_set['flake_id'] = data_set.index
    classes = classes.drop('cam', axis = 1)
    display(data_set)
    display(classes)

    # Be sure that all teh falke id class are consistent
    data_set, classes = clean(data_set, classes, classifier)

    return data_set, classes


# ------------------------------------------------------------------ #

def clean(data_set, classes, classifier):

    # Get all the wrong duplicates flakes
    mascdb_classes_copy = classes.copy()

    mascdb_classes_copy_1 = mascdb_classes_copy[mascdb_classes_copy.duplicated(subset = None, keep = False)]
    mascdb_classes_copy_2 = mascdb_classes_copy[mascdb_classes_copy.duplicated(subset=['flake_id'], keep = False)]

    mascdb_classes_wrong_duplicates = pd.concat([mascdb_classes_copy_1, mascdb_classes_copy_2]).drop_duplicates(keep = False)



    # Get the flake id of the wrong duplicates
    mascdb_classes_wrong_duplicates_unique = mascdb_classes_wrong_duplicates.drop_duplicates(subset = ['flake_id'], keep = 'first')

    # Get all the flake id with classes
    flake_id_classes = mascdb_classes_copy.drop_duplicates(subset=['flake_id'], keep = 'first')

    # Remove the wrong flake id from all the flake id
    mascdb_classes_modified = pd.concat([flake_id_classes, mascdb_classes_wrong_duplicates_unique]).drop_duplicates(subset=['flake_id'], keep = False)



    # Now, we want to be sure to have one class for each snowflakes
    mascdb_data_modified = data_set[data_set.flake_id.isin(mascdb_classes_modified.flake_id)]



    # Transform the data
    mascdb_data_modified_copy = mascdb_data_modified.copy()
    power_transformer = PowerTransformer(method = 'yeo-johnson', standardize = True)
    mascdb_data_modified_std = power_transformer.fit(mascdb_data_modified_copy.drop(['flake_id'], axis=1))
    mascdb_data_modified_std = power_transformer.transform(mascdb_data_modified_copy.drop(['flake_id'], axis=1))

    # Set the transformed data
    mascdb_data_modified[mascdb_data_modified.columns.difference(['flake_id'])]  = mascdb_data_modified_std


    # Split into a data set X_ and a response set y_
    X_ = mascdb_data_modified[mascdb_data_modified.columns.difference(['flake_id'])]
    y_ = mascdb_classes_modified.copy().set_index('flake_id')

    # Get a column as flake_id
    X_['flake_id'] = X_.index

    # Supress all the duplicates flake_id and get the correponding class
    X_ = X_.drop_duplicates(subset = 'flake_id', keep = 'first').join(y_)



    # Split into a data set X and a response set y
    y = pd.DataFrame(X_['class_id'])
    X = X_[X_.columns.difference(['flake_id', 'class_id'])]

    return X, y


def clean_perso(data_set, classes, classifier):

    # Remove miss classification
    classes = remove_wrong_classifications(classes, classifier)

    # Get all the data where we have right classification
    X = select_right_classification(data_set, classes)

    # Clean the data set
    X = clean_duplicates(X)

    # WRONG
    # y_to_keep = classes.apply(lambda row: keep_right_row(row, X), axis = 1)
    # y = classes[y_to_keep]

    # GOOD
    y = classes[classes.flake_id.isin(X.flake_id)].drop_duplicates(subset = 'flake_id', 
                                                                    keep = 'first').set_index('flake_id')

    X = X.drop('cam', axis = 1).set_index('flake_id')

    X = X.join(y)
    y = pd.DataFrame(X['class_id'])
    X = X[X.columns.difference(['class_id', 'cam'])]

    return X, y


# ------------------------------------------------------------------ #


def clean_duplicates(X):

    #X = X.groupby(['flake_id']).mean().reset_index(drop = False)

    X = X.drop_duplicates(subset = ['flake_id'], keep = 'first')

    return X


# ------------------------------------------------------------------ #


def select_right_classification(data_set, classes):

    X = data_set[data_set.flake_id.isin(classes.flake_id)].reset_index(drop = True)
    X_to_keep = X.apply(lambda row: keep_right_row(row, classes), axis = 1)
    X = X[X_to_keep]

    return X


# ------------------------------------------------------------------ #


def keep_right_row(row, classes):

    if row.cam in classes[classes.flake_id == row.flake_id].cam.to_list():
        return True
    else:
        return False


# ------------------------------------------------------------------ #


def remove_wrong_classifications(classes, classifier):

    # Get the exact duplicates in classes
    exact_duplicates = classes[classes.duplicated(subset = ['flake_id', 'class_id'], 
                                                    keep = False)]

    # Get all the duplicates having the same flake id in classes
    all_duplicates = classes[classes.duplicated(subset = ['flake_id'], 
                                                    keep = False)]

    # Get all the miss classification
    wrong_classification = pd.concat([exact_duplicates, all_duplicates]).drop_duplicates(subset = ['flake_id', 'class_id'], keep = False)
    save_wrong_classifications(wrong_classification, classifier)

    # Get the flake id of the wrong duplicates
    wrong_classification_id = wrong_classification.drop_duplicates(subset = ['flake_id'], 
                                                    keep = 'first')

    # Get all the flake id with classes
    classes_id = classes.drop_duplicates(subset = ['flake_id'], 
                                                    keep = 'first')

    # Remove the wrong flake id from all the flake id
    cleaned_classes_id = pd.concat([classes_id, wrong_classification_id]).drop_duplicates(subset = ['flake_id'], 
                                                                            keep = False)
    classes = classes[classes.flake_id.isin(cleaned_classes_id.flake_id)]

    # Return teh result
    return classes


# ------------------------------------------------------------------ #


def save_wrong_classifications(wrong_classification, classifier):
    path = 'Data/wrong_classifications'
    path = os.path.join(path, classifier+'.pkl')
    wrong_classification.to_pickle(path)
    

# ------------------------------------------------------------------ #


def column_remover(data_set, classes):
    # Get the columns to delete for our experiences
    black_list_words = ['roi', 'riming', 'melting', 'snowflake', 'hl']
    cols_to_delete = list(filter(lambda cols: any(word in cols for word in black_list_words), data_set.columns))
    cols_to_delete.extend(['datetime', 'pix_size', 'flake_number_tmp', 'cam_id'])
    data_set = data_set.drop(cols_to_delete, axis = 1)
    return data_set, classes


# ------------------------------------------------------------------ #