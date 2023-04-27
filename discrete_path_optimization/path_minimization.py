import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sp
import min_utils as utils
import plotting
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatters

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../visibility_NN")
import visibility_graph as vg

'''Python port of matlab shortest distance guessing code'''

def create_guess(obs_file,start,end,out_fname="initial_guess"):
    #TODO this code will create valid guess that satisfies obstacle labels
    x0 = start[0]
    xf = end[0]
    y0 = start[0]
    yf = end[0]
    dx = 0.5
    N = (xf-x0)/dx

    np.save(out_fname,guess)
    pass

def import_guess(guess_file):
    guess = np.load(guess_file)
    return guess

def short_dist_multi_obs(obs_file,guess_file):
    obstacles = vg.read_obstacle_list(obs_file)
    guess = import_guess(guess_file) # guess is 2xN array with x values in first row and y values in second row of guess

    x0 = guess[0,0]
    xf = guess[0,-1]
    y0 = guess[1,0]
    yf = guess[1,-1]
    dx = guess[0,1]-guess[0,0]
    N = (xf-x0)/dx
    x_span = guess[0,:]
    x_span_guess = guess[0,1:-1]
    y_span = guess[1,:]
    y_span_guess = guess[1,1:-1]
    
    input_point_idx = np.zeros(1,)


def arg_parse():
    parser = ArgumentParser(description="Discrete Path Minimization.")
    parser.add_argument("-f","--fname", help="specify the file name to be used.  This changes based on run option selected")
    parser.add_argument("--run_create_guess", action="store_true", help="Creates guess based on labels input and problem description")
    parser.add_argument("--run_path_minimizer", action="store_true", help="Minimizes guess")
    parser.add_argument("--guess", help="file that contains guess")
    parser.add_argument("-s", "--start", type=float, default = [0,3], nargs=2, help='course start point')
    parser.add_argument("-e", "--end", type=float, default = [30,15], nargs=2, help='course end point')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    default_args={'dx'}
    args = arg_parse()
    if args.run_create_guess:
        guess = create_guess(args.fname,args.start,args.end)
    elif args.run_path_minimizer:
        short_dist_multi_obs(args.fname)
    else:
        print("No argument chosen")