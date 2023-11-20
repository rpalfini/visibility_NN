import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.python.client import device_lib
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle

import util
import graph_util as g_util

'''This script adds plots to already trained files that do not have a loss and accuracy plot'''

def add_plot_to_files(fpath):
    model_numbers = util.get_completed_model_list(fpath)

    for model_num in model_numbers:
        pickle_file = os.path.join(fpath,model_num,f"{model_num}_results.pkl")

        results = pickle.load(open(pickle_file,'rb'))
        g_util.plot_hist(results)
        fig_path = os.path.join(fpath,model_num,f"{model_num}_loss_acc.png")
        plt.savefig(fig_path)
        

if __name__ == "__main__":
    fpath = "C:/Users/Robert/git/visibility_NN/main_train_results/main_data_file_courses20"
    if os.name == 'nt':
        fpath = fpath.replace('/','\\')
    add_plot_to_files(fpath)

