import numpy as np
import path_minimization as pm
import scipy as sp



def distance_test():

    data = sp.io.loadmat('./discrete_path_optimization/distance_test.mat')
    x_out = data['x_out'][0]
    y_out = data['y_out'][0]
    y_span_guess = data['y_span_guess'][0]
    solution_cost_truth = data['solution_cost'][0]
    
    solution_cost = pm.find_path_cost(x_out,y_out)

    if round(solution_cost_truth[0],5) == round(solution_cost,5):
        print('distance test passed')
    else:
        print('distance test failed')
    print(f'solution_cost_python = {solution_cost:.5f}; true cost = {solution_cost_truth[0]:.5f}')

if __name__ == "__main__":
    distance_test()