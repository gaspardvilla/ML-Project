# Import the used libraries
from helpers import features_selection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from IPython.display import display
from sklearn.pipeline import Pipeline

from sklearn.datasets import *
from sklearn.ensemble import *
from sklearn.experimental import *
from sklearn.model_selection import *

# Import sklearn librairies
from sklearn.feature_selection import *
from sklearn.model_selection import *
from sklearn.ensemble import *
from sklearn.neural_network import *
from sklearn.pipeline import *
from sklearn.preprocessing import *
from sklearn.linear_model import *
from yellowbrick.model_selection import *
from sklearn.svm import *
from sklearn.decomposition import *
from sklearn.metrics import *



# --------------------------------------------------------------------------------------- #



def fs_param_update(feature_select_param):

    feature_select_param_ = feature_select_param.copy()
    for key in feature_select_param:
        new_key = 'fs__' + str(key)
        feature_select_param_[new_key] = feature_select_param_.pop(key)

    return feature_select_param_



# --------------------------------------------------------------------------------------- #



def tm_param_update(train_param):

    train_param_ = train_param.copy()
    for key in train_param:
        new_key = 'tm__' + str(key)
        train_param_[new_key] = train_param_.pop(key)
        
    return train_param_



# --------------------------------------------------------------------------------------- #



def cross_validation_general(feature_select_method, feature_select_param, train_method, train_param):
    
    # 
    steps_method = Pipeline([('fs', feature_select_method),
            ('tm', train_method)])

    # Update the parameters to optimize
    param_grid = fs_param_update(feature_select_param)
    train_param = tm_param_update(train_param)
    param_grid.update(train_param)

    cross_valodation_method = GridSearchCV(steps_method, param_grid, verbose = 1)

    return cross_valodation_method



# --------------------------------------------------------------------------------------- #