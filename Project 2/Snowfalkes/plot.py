from helpers import *

# Import the used libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from IPython.display import display
import warnings


# --------------------------------------------------------------------------------------- #

#list of features giving geometric or textural information
geometry = ['D90_r', 'Dmax', 'Dmax_90', 'Dmax_ori', 'Dmean', 'area', 'area_porous',
       'area_porous_r', 'bbox_len', 'bbox_width', 'compactness',
       'convexity', 'ell_fit_A', 'ell_fit_B', 'ell_fit_a_r',
       'ell_fit_area', 'ell_fit_ecc', 'ell_fit_ori', 'ell_in_A', 'ell_in_B',
       'ell_in_area', 'ell_out_A', 'ell_out_B', 'ell_out_area', 'eq_radius',
       'frac_dim_boxcounting', 'frac_dim_theoretical', 'har_contrast',
       'har_corr', 'har_energy', 'har_hom', 'hull_n_angles',
       'nb_holes', 'p_circ_out_r', 'p_circ_r',
       'perim', 'quality_xhi', 'rect_aspect_ratio', 'rect_eccentricity',
       'rect_perim_ratio', 'rectangularity', 'roundness', 'skel_N_ends',
       'skel_N_junc', 'skel_area_ratio', 'skel_perim_ratio', 'solidity',
       'sym_P1', 'sym_P2', 'sym_P3', 'sym_P4', 'sym_P5', 'sym_P6',
       'sym_P6_max_ratio', 'sym_Pmax_id', 'sym_mean', 'sym_std',
       'sym_std_mean_ratio']
texture = ['intensity_max', 'intensity_mean','intensity_std', 'local_intens',
            'contrast', 'hist_entropy', 'complexity','wavs', 'local_std', 'lap_energy']


# --------------------------------------------------------------------------------------- #


def plot_feature_importance (model, threshold = 0.5):
    """
    Plot the importance of the feature

    Args:
        model : model for which feature importance will be plot
        threshold : if the importance of feature is below the thrashod the feature will be removed
    
    Return a dataframe with feature id assigned to the name of the feature

    """

    importance = np.abs(model.coef_[0])
    feature_names = model.feature_names_in_

    #create a dataframe that assign to each importance value its corresponding feature name
    importance = pd.DataFrame(importance, feature_names)

    #sort importance in decreasing order
    importance = importance.sort_values(by=0,ascending=False)

    #add a column to the sorted_importance dataframe to assign the color of the feature according to its type (geometric or texture)
    colors = np.empty(importance.shape[0], dtype=str)
    importance['color'] = colors

    #select features present in the datasset
    geometry_list = list(set(importance.T.columns).intersection(geometry))
    texture_list = list(set(importance.T.columns).intersection(texture))

    #assign color in function of the type of the feature
    importance['color'][geometry_list] = 'orange'
    importance['color'][texture_list] = 'green'

    #remove feature with importance less than threshold
    importance = importance.where(importance[0] > threshold, np.nan)
    importance = importance.dropna()

    # plot feature importance       
    labels = list(['Geometry', 'Texture'])
    colors = {'Geometry':'orange', 'Texture':'green'}
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.bar([x for x in range(len(importance))], 
            importance[0], 
            color=importance['color'])
    plt.title('Riming degree')
    plt.xlabel('Feature ID'), plt.ylabel('Feature importance')
    plt.legend(handles, labels)
    plt.show()

    #assign name of the feature to its corresponding id
    print('Feature ID')
    print(pd.DataFrame(np.linspace(0, len(importance), num = len(importance), dtype=int), importance.T.columns, columns=['FeatureID']))


# --------------------------------------------------------------------------------------- #


def plot_conf_matrix(model, X_test, y_test):
    """
    Plot the confusion matrix for the model

    Args:
        model : model for which the confusion will be plot
        X_test : dataset used for predicting the target using the model
        y_test : true target  
    """

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    sns.heatmap(cm, annot=True, cmap = 'hot',
                xticklabels=['none', 'rimed', 'densily rimed', 'graupel-like', 'graupel'], 
                yticklabels=['none', 'rimed', 'densily rimed', 'graupel-like', 'graupel'])
    plt.title('Confusion Matrix')


# --------------------------------------------------------------------------------------- #


def plot_cv_results(cv, hyperparam, x_max):
    """
    Plot the accuracy score of both test and train set as a function of the hyperpar
    
    Args:
        cv: the returned cross-validation model after its evaluation by evaluate_model
        hyperparam: hyperparameter that was tuned by cross-validation and that we want to plot (a string) 
        x_max: the maximum value the hyperparameter can take
    """

    # Get the regular numpy array from the MaskedArray
    results = cv.cv_results_
    scoring = 'Accuracy'
    X_axis = np.array(results['param_%s' % hyperparam].data, dtype=float)
    best_index = np.nonzero(results['rank_test_%s' % scoring] == 1)[0][0]
    best_score = results['mean_test_%s' % scoring][best_index]

    plt.figure(figsize=(13, 13))
    plt.title('GridSearchCV evaluating using accuracy scoring', fontsize=16)

    plt.xlabel('%s' % hyperparam)
    plt.ylabel('Score')

    ax = plt.gca()
    ax.set_xlim(0, x_max)
    ax.set_ylim(best_score-0.05, best_score+0.05)

    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = results['mean_%s_%s' % (sample, scoring)]
        sample_score_std = results['std_%s_%s' % (sample, scoring)]
        ax.fill_between(
            X_axis,
            sample_score_mean - sample_score_std,
            sample_score_mean + sample_score_std,
            alpha=0.1 if sample == 'test' else 0,
            )
        ax.plot(
            X_axis,
            sample_score_mean,
            style,
            alpha=1 if sample == 'test' else 0.7,
            label='%s (%s)' % (scoring, sample),
            )

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot(
        [
            X_axis[best_index],
        ]
        * 2,
        [0, best_score],
        linestyle='-.',
        marker='x',
        markeredgewidth=3,
        ms=8,
    )

    # Annotate the best score for that scorer
    ax.annotate('%0.2f' % best_score, (X_axis[best_index], best_score + 0.005))

    plt.legend(loc='best')
    plt.grid(False)
    plt.show()


# --------------------------------------------------------------------------------------- #
