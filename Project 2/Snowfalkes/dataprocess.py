import pandas as pd
import os
from tqdm import tqdm

from dataloader import load_data_sets

tqdm.pandas()


# ------------------------------------------------------------------ #


def get_processed_data(classifier = 'hydro'):

    # Directly the dataset
    data_set, classes = load_data_sets(classifier = classifier)

    # Remove all the columns that are not interesting for us
    data_set, classes = column_remover(data_set, classes)

    # Be sure that all teh falke id class are consistent
    data_set, classes = clean(data_set, classes, classifier)

    return data_set, classes


# ------------------------------------------------------------------ #


def clean(data_set, classes, classifier):

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
    cleaned_classes_id = pd.concat([classes_id, wrong_classification_id]).drop_duplicates(subset = ['flake_id', 'class_id'], 
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