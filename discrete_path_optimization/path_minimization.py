import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sp
import min_utils as utils
import plotting
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

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

def distance_objective(y0,yf,y,dx,N):
    # defines distance based objective for minimization
    result = np.sqrt(dx ** 2 + (y[0] - y0) ** 2)
    for ii in range(0, N - 2):
        result += np.sqrt(dx ** 2 + (y[ii + 1] - y[ii]) ** 2)
    result += np.sqrt(dx ** 2 + (yf - y[N - 2]) ** 2)
    return result

def circle_constraint(y_vals, x, obstacles):
    # defines constraints for minimization
    num_obst = obstacles.shape[0]
    num_steps = len(y_vals)-2

    c = np.array([])
    for jj in range(num_obst):
        c_inter = np.zeros(num_steps)
        for ii in range(num_steps):
            c_inter[ii] = -(x[ii+1]-obstacles[jj,1])**2 - (y_vals[ii]-obstacles[jj,2])**2 + obstacles[jj,0]**2
        c = np.concatenate((c, c_inter))
    ceq = np.array([])
    return c, ceq

def find_path_cost(points):
    """
    Calculates the cost of the given path represented by a sequence of points.
    
    Args:
    - points: a 2D NumPy array of shape (n, 2) representing the coordinates of the n points
    
    Returns:
    - cost: a float representing the total cost of the path
    """
    n = points.shape[0]
    distances = np.zeros(n)
    for i in range(n-1):
        distances[i] = np.linalg.norm(points[i,:] - points[i+1,:])
    cost = np.sum(distances)
    return cost

def short_dist_multi_obs(obs_file,guess_file):
    obstacles = vg.read_obstacle_list(obs_file)
    guess = utils.import_guess(guess_file) # guess is 2xN array with x values in first row and y values in second row of guess

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
    distance_objective(y0,yf,y_span_guess,dx,N)

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