import pandas as pd

from dataloader import load_data_sets


# ------------------------------------------------------------------ #


def get_processed_data(classifier = 'hydro'):
    data_set, classes = load_data_sets(classifier = classifier)

    data_set, classes = process(data_set, classes)

    return data_set, classes


# ------------------------------------------------------------------ #


def process(data_set, classes):
    # Get the columns to delete for our experiences
    black_list_words = ['roi', 'riming', 'melting', 'snowflake', 'hl']
    cols_to_delete = list(filter(lambda cols: any(word in cols for word in black_list_words), data_set.columns))
    cols_to_delete.extend(['datetime', 'pix_size', 'flake_number_tmp'])
    data_set = data_set.drop(cols_to_delete, axis = 1)
    return data_set, classes