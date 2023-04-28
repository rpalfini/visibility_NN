import numpy as np
import path_minimization as pm
import visibility_graph as vg
import min_utils as utils
import scipy as sp


def constraint_test():
    x0 = 3
    y0 = 17
    xf = 28
    yf = 2
    dx = 0.5
    obs_file = "C:/Users/Robert/git/visibility_NN/obs_courses/1_courses_5_obstacles_normal.txt"
    obstacles = vg.read_obstacle_list(obs_file)
    N = (xf-x0)/dx
    # load a created guess
    mat_guess_file = './discrete_path_optimization/distance_test.mat'
    x_out, y_out, y_span_guess, y_guess, solution_cost_truth = utils.load_guess_from_mat(mat_guess_file)

    # Define constraints
    cons = {'type': 'ineq', 'fun': pm.circle_constraint, 'args': (x,obstacles)}

    # Use interior-point algorithm for optimization
    method = 'SLSQP'

    res = minimize(pm.distance_objective, 






if __name__ == "__main__":
    constraint_test()