import numpy as np
import os
import sys
import scipy.io as sio


def import_guess(guess_file):
    guess = np.load(guess_file)
    return guess

def load_guess_from_mat(guess_mat_file):
    # if you want to use guess file created in matlab ginput interface
    data = sio.loadmat(guess_mat_file)
    x_span = data['x_span'][0]
    x_out = data['x_out'][0]
    y_out = data['y_out'][0]
    y_span_guess = data['y_span_guess'][0] 
    y_guess = data['y_guess'][0]
    solution_cost = data['solution_cost'][0]

    return x_span, x_out, y_out, y_span_guess, y_guess, solution_cost