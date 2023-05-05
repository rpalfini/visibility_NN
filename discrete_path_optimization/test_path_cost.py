import numpy as np
import path_minimization as pm
import scipy as sp
import min_utils as utils



def distance_test():

    mat_guess_file = './discrete_path_optimization/distance_test.mat'
    x_span, x_out, y_out, y_span_guess, y_guess, solution_cost_truth = utils.load_guess_from_mat(mat_guess_file)
    solution_cost = pm.find_path_cost(x_out,y_out)
    
    if round(solution_cost_truth[0],5) == round(solution_cost,5):
        print('distance test passed')
    else:
        print('distance test failed')
    print(f'solution_cost_python = {solution_cost:.5f}; true cost = {solution_cost_truth[0]:.5f}')

if __name__ == "__main__":
    distance_test()