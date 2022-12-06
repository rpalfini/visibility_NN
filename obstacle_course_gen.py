import numpy as np
import random
import matplotlib.pyplot as plt


def round_radius(r_vec,r_bound):
    # return [1 if x<r_bound[0] or x>r_bound[1] else x for x in r_vec]
    return 1 if r_vec<r_bound[0] or r_vec>r_bound[1] else r_vec[0]

def sample_radius(mu,sigma,r_bound):
    r = np.random.normal(mu,sigma,1)
    r = round_radius(r,r_bound)
    return r

def sample_point_uniform(radius,bound_x,bound_y):
    x = np.random.uniform(0+radius,bound_x-radius,1)
    y = np.random.uniform(0+radius,bound_y-radius,1)
    return x[0],y[0]

def sample_point_normal(radius,bound_x,bound_y,mu,sigma):
    valid = False
    while not valid:
        x = np.random.normal(mu,sigma,1)
        y = np.random.normal(mu,sigma,1)
        if not (x[0]<0+radius or x[0]>bound_x-radius):
            valid = True
        if not (y[0]<0+radius or y[0]>bound_y-radius):
            valid = True
    return x[0],y[0]

def circle_intersect(circle1,circle2,gap=2):
    d = np.sqrt((circle1[1]-circle2[1])**2 + (circle1[2]-circle2[2])**2)
    if d <= abs(circle1[0]-circle2[0]) or d <= circle1[0] + circle2[0] + gap:
        circles_intersect = True
    else:
        circles_intersect = False

    return circles_intersect

def format_circle(r,x,y):
    return (r,x,y)

def check_placement(cand_obs,placed_obstacles):
    if len(placed_obstacles) == 0:
        return True

    valid = True
    for o in placed_obstacles:
        intersects = circle_intersect(cand_obs,o)
        if intersects:
            valid = False
            break
    return valid

def plot_obstacles(obs_list,axs):
    for obstacle in obs_list:
        obst_x, obst_y = make_circle_points(obstacle)
        # self.vis_axs.plot(obst_x, obst_y,color='blue',linewidth=self.line_width,label="obstacle")
        axs.plot(obst_x, obst_y,color='blue',linewidth=2)

def make_circle_points(obstacle):
    r = obstacle[0]
    x = obstacle[1]
    y = obstacle[2]
    thetas = np.linspace(0,2*np.pi,100)
    bound_x = r*np.cos(thetas) + x
    bound_y = r*np.sin(thetas) + y
    return bound_x, bound_y

def gen_obs(num_obstacles = 6,show_result = False,fname = "obstacle_locations.txt"):
    
    output_result = True
    obstacles = []
    bound_x = 20
    bound_y = 20
    r_bound = (0.5,6)
    mu, sigma = 4, 2
    mu_circle, sigma_circle = 10, 5
    max_attempts = 20

    for i in range(num_obstacles):
        cand_r = sample_radius(mu,sigma,r_bound)
        placed = False
        place_attempts = 0
        while not placed:
            # cand_x,cand_y = sample_point_uniform(cand_r,bound_x,bound_y)
            cand_x,cand_y = sample_point_normal(cand_r,bound_x,bound_y,mu_circle,sigma_circle)
            cand_obs = format_circle(cand_r,cand_x,cand_y)
            valid = check_placement(cand_obs,obstacles)
            if valid:
                obstacles.append(cand_obs)
                placed = True
            else:
                place_attempts += 1
                if place_attempts < max_attempts:
                    place_attempts = 0
                    cand_r = sample_radius(mu,sigma,r_bound)
            
    if output_result:
        with open(fname,"a") as file:
            file.write("New Obstacle Set:\n")
            file.write("radius,x,y\n")
            for obs in obstacles:
                file.write(f"{obs[0]},{obs[1]},{obs[2]}\n")

    if show_result:
        fig,axs = plt.subplots()
        plot_obstacles(obstacles,axs)
        axs.set_aspect('equal')
        plt.show()

def gen_multi_courses(num_obs):
    fname = f'{num_obs}_obstacle_locations.txt'
    for ii in range(num_obs):
        gen_obs(fname=fname)

if __name__ == "__main__":
    batch = False
    courses = 100
    fname = "100_obstacle_locations_uniform.txt"
    if not batch:
        gen_obs(show_result=True)
    else:
        for ii in range(courses):
            gen_obs(fname=fname)
