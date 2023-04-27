import numpy as np
import math
import path_minimization as pm



def optimization_func_test():
    x = np.arange(0,6)
    dx = 1
    G = (x[-1]-x[0])/dx
    y0 = 0
    yd = np.arange(1,5)
    yf = 5

    N = len(yd)+1 #is N supposed to be one less than usual?

    # define optimization function

    path_length = round(pm.distance_objective(y0,yf,yd,dx,N),4)
    if path_length == 7.0711:
        print('test passed')
    else:
        print('test failed')

    print(path_length)
    print(f'G = {G}')


if __name__ == "__main__":
    optimization_func_test()