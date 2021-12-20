import pandas as pd

from dataloader import load_data_sets


# ------------------------------------------------------------------ #


def get_processed_data(classifier = 'hydro'):

    # Directly the dataset
    data_set, classes = load_data_sets(classifier = classifier)

    # Remove all the columns that are not interesting for us
    data_set, classes = column_remover(data_set, classes)

    # Be sure that all teh falke id class are consistent
    data_set, classes = clean(data_set, classes)

    return data_set, classes


# ------------------------------------------------------------------ #


def clean(data_set, classes):

    return None


# ------------------------------------------------------------------ #


def column_remover(data_set, classes):
    # Get the columns to delete for our experiences
    black_list_words = ['roi', 'riming', 'melting', 'snowflake', 'hl']
    cols_to_delete = list(filter(lambda cols: any(word in cols for word in black_list_words), data_set.columns))
    cols_to_delete.extend(['datetime', 'pix_size', 'flake_number_tmp'])
    data_set = data_set.drop(cols_to_delete, axis = 1)
    return data_set, classes


# ------------------------------------------------------------------ #