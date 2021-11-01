import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def correlation_plot(data_set):
    # Compute the correlation matrix
    Corr = np.corrcoef(data_set.T)

    # Highlight the highly correlated features
    Corr[np.where(Corr > 0.95)] = 1
    Corr[np.where(Corr < -0.95)] = -1

    # Plot the correlation matrix in colormap
    plt.figure(figsize=(8, 8))
    plt.matshow(Corr, 0)
    plt.colorbar()

def density_plot(data_set, feature_x, feature_y):
    # Initialize x and y
    x = data_set[:,feature_x]
    y = data_set[:,feature_y]

    # Set the number of bins
    b = 20

    # Visualization
    plt.figure(figsize=(16,6))
    plt.subplot(121)
    plt.hist2d(x, y, bins = (b, b), cmap=plt.cm.Reds)
    plt.subplot(122)
    plt.scatter(x, y)

def histogram_plot(y, data_set, feature_idx=None, ylabel=''):
    # Initialization
    zero_ind = np.array(np.where(y == 0)[0])
    unit_ind = np.array(np.where(y == 1)[0])

    if feature_idx==None:
        feature_idx = np.arange(data_set.shape[1])

    # Visualization
    for col in feature_idx:
        plt.figure(figsize=(8,5))
        sns.distplot(data_set[zero_ind, col], bins=100,  kde=False, label='y = 0', norm_hist=True)
        sns.distplot(data_set[unit_ind, col], bins=100,  kde=False, label='y = 1', norm_hist=True)
        plt.title('Title: Density plot of the feature %d' % (col))
        plt.ylabel('Density')
        plt.xlabel('Value after standardization')
        plt.legend()
        plt.show()

def cross_validation_visualization(error_tr, error_te, parameters, indice):
    train_error = np.reshape(error_tr, [len(error_tr),])
    test_error = np.reshape(error_te, [len(error_te),])

    tab = {'yolo': parameters.range(indice-1), 'Train error': train_error, \
         'Test error': test_error}
    error_df = pd.DataFrame({
        parameters.names[indice-1]: parameters.range(indice-1), 
        'Train error': train_error, 
        'Test error': test_error})

    lambds = parameters.range(indice)

    g = sns.relplot(data=pd.melt(error_df, [parameters.names[indice-1]]), x=parameters.names[indice-1],
     y='value', hue='variable', kind = 'line', err_style="bars", ci=68) 
    g.set(xscale="log")
    plt.figure(figsize=(15,8))
    g.set_ylabels('error [%]')
    return error_df