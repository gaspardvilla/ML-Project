import numpy as np
import matplotlib.pyplot as plt

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

def density_plot(data_set):
    # Initialize x and y
    x = data_set[:,1]
    y = data_set[:,3]

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
        plt.hist(data_set[zero_ind, col], bins=100, histtype = 'step', color = 'green', density=True)
        plt.hist(data_set[unit_ind, col], bins=100, histtype = 'step', color = 'red', density=True)
        plt.title('Title: Density plot of the feature %d' % (col))
        plt.ylabel('Density')
        plt.xlabel(ylabel)
        plt.legend(['y = 0', 'y = 1'])
        plt.show()