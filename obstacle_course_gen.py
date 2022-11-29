import numpy as np
import random
import matplotlib.pyplot as plt


def round_radius(r_vec,r_bound):
    return [1 if x<r_bound[0] or x>r_bound[1] else x for x in r_vec]

def sample_radius(mu,sigma):
    r = np.random.normal(mu,sigma,1)
    r = round_radius(r)
    return r


print(s)
d = round_radius(s)
print(d)

s = np.random.uniform(0,20,100000)
count, bins, ignored = plt.hist(s, 15, density=True)
plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
plt.show()



num_obstacle = 8
bound_x = 20
bound_y = 20
r_bound = (0.5,6)
mu, sigma = 4, 2


# Creates obstacle course 
