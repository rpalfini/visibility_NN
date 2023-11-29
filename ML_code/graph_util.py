import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


def plot_hist(results, x_start = 0, x_end = None, logscale = 1):
    if x_end is None:
        x_end = len(results['all_outputs_correct'])
    
    metric = results['all_outputs_correct']
    val_metric = results['val_all_outputs_correct']
    metric_2 = results['binary_accuracy']
    val_metric_2 = results['val_binary_accuracy']
    loss = results['loss']
    val_loss = results['val_loss']
    if logscale == 1:
        # only for loss
        loss = np.log10(loss)
        val_loss = np.log10(val_loss)

    epochs = range(len(metric))

    # (width, height) = (16, 4)
    (width, height) = (12,8)
    fig = plt.figure(figsize = (width, height))

    plt.rc('font', size=15)          # controls default text sizes
    plt.rc('axes', titlesize=15)     # fontsize of the axes title
    plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
    plt.rc('legend', fontsize=15)    # legend fontsize
    plt.rc('figure', titlesize=15)  # fontsize of the figure title
    
    # (n_row, n_col) = (1, 2)
    (n_row, n_col) = (3, 1)
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
    plt.plot(epochs, metric, 'b-', label='Training Sample Acc', linewidth = 3)
    plt.plot(epochs, val_metric, 'r--', label='Validation Sample Acc', linewidth = 2)
    plt.title('Sample_Accuracy vs. Epochs', fontweight='bold')  
    ax.set_xlim(x_start, x_end)
    plt.legend() 

    ax = fig.add_subplot(gs[2])
    plt.plot(epochs, metric_2, 'b-', label='Training Binary Acc', linewidth = 3)
    plt.plot(epochs, val_metric_2, 'r--', label='Validation Binary Acc', linewidth = 2)
    plt.title('Binary_Accuracy vs. Epochs', fontweight='bold')  
    ax.set_xlim(x_start, x_end)
    plt.legend()
    plt.tight_layout()

def plot_2_hist(results, x_start = 0, x_end = None, logscale = 1):
    '''Updating code but this one just plots loss and accuracy as it did originally.'''
    if x_end is None:
        x_end = len(results['all_outputs_correct'])
    
    metric = results['all_outputs_correct']
    val_metric = results['val_all_outputs_correct']
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
    plt.title('Sample_Accuracy vs. Epochs', fontweight='bold')  
    ax.set_xlim(x_start, x_end)
    plt.legend() 


def old_plot_hist(results, x_start = 0, x_end = None, logscale = 1):
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

def save_loss_acc_plot(results, output_folder):
    model_results_path, model_number = os.path.split(output_folder)
    plot_hist(results)
    fig_path = os.path.join(output_folder,f"{model_number}_loss_acc.png")
    plt.savefig(fig_path)