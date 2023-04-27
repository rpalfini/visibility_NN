import numpy as np
import os
import sys



def import_guess(guess_file):
    guess = np.load(guess_file)
    return guess
