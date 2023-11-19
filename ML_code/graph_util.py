import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


def plot_hist(results, x_start = 0, x_end = None, logscale = 1):
    if x_end is None:
        x_end = len(results['accuracy'])
    
    metric = results['accuracy']
    val_metric = results['val_accuracy']
    loss = results['loss']
    val_loss = results['val_loss']
    if logscale == 1:
        # only for loss
        loss = np.log10(loss)
        val_loss = np.log10(val_loss)

    epochs = range(len(metric))

    (width, height) = (16, 4)
    fig = plt.figure(figsize = (width, height))

    plt.rc('font', size=15)          # controls default text sizes
    plt.rc('axes', titlesize=15)     # fontsize of the axes title
    plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
    plt.rc('legend', fontsize=15)    # legend fontsize
    plt.rc('figure', titlesize=15)  # fontsize of the figure title
    
    (n_row, n_col) = (1, 2)
    gs = GridSpec(n_row, n_col)

    ax = fig.add_subplot(gs[0])
    plt.plot(epochs, loss, 'b-', label='Training Loss', linewidth = 3)
    plt.plot(epochs, val_loss, 'r--', label='Validation Loss', linewidth = 2)
    if logscale == 1:
        plt.title('Log$_{10}$(Loss) vs. Epochs', fontweight='bold')
    else:
        plt.title('Loss vs. Epochs', fontweight='bold')
    ax.set_xlim(x_start, x_end)
    plt.legend()    
    
    ax = fig.add_subplot(gs[1])
    plt.plot(epochs, metric, 'b-', label='Training Accuracy', linewidth = 3)
    plt.plot(epochs, val_metric, 'r--', label='Validation Accuracy', linewidth = 2)
    plt.title('Accuracy vs. Epochs', fontweight='bold')  
    ax.set_xlim(x_start, x_end)
    plt.legend() 

    # plt.subplots_adjust(wspace=0.2)