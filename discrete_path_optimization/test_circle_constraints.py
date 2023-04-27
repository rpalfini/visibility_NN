import numpy as np
import path_minimization as pm
import min_utils as utils



def constraint_test():
    x0 = 3
    y0 = 17
    xf = 28
    yf = 2
    dx = 0.5
    guess_file = "C:/Users/Robert/git/visibility_NN/obs_courses/1_courses_5_obstacles_normal.txt"
    obstacles = utils.import_guess(guess_file)
    N = (xf-x0)/dx
    pm.circle_constraint(x, obstacles, y_vals)

    # Define constraints
    cons = {'type': 'ineq', 'fun': pm.circle_constraint, 'args': (x,obstacles)}

    # Use interior-point algorithm for optimization
    method = 'SLSQP'

    res = minimize(pm.distance_objective(y0, yf, y, dx, N))






if __name__ == "__main__":
    constraint_test()