import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import date
import math
import time


# This file can be used to generate obstacle courses input info can be obtained by running with -h option

def round_radius(r_vec,r_bound):
    # return [1 if x<r_bound[0] or x>r_bound[1] else x for x in r_vec]
    return 1 if r_vec<r_bound[0] or r_vec>r_bound[1] else r_vec[0]

def sample_radius_normal(mu,sigma,r_bound):
    is_valid = False
    while not is_valid:
        r = np.random.normal(mu,sigma,1)[0]
        if r > 0.1 and r <= r_bound:
            is_valid = True
        # r = round_radius(r,r_bound)
    return r

def sample_radius_uniform(r_bound):
    r = np.random.uniform(0.1,r_bound)
    return r

def sample_point_uniform(radius,bound_x,bound_y,x_start,y_start):
    x = np.random.uniform(x_start+radius,x_start+bound_x-radius,1)
    y = np.random.uniform(y_start+radius,y_start+bound_y-radius,1)
    return x[0],y[0]

def sample_point_normal(radius,bound_x,bound_y,mu,sigma,x_start,y_start):
    valid = False
    while not valid:
        x = np.random.normal(mu,sigma,1)
        y = np.random.normal(mu,sigma,1)
        if not (x[0]<x_start+radius or x[0]>x_start+bound_x-radius):
            if not (y[0]<y_start+radius or y[0]>y_start+bound_y-radius):
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

def gen_obs(num_obstacles = 6,show_result = False, start_x=5, start_y=5, bound_x=25, bound_y=25, fname="obstacle_locations.txt", mode='normal'):
    ''' main function loop that attempts to place objects in course field'''
    output_result = True
    obstacles = []
    bound_x = bound_x
    bound_y = bound_y
    r_bound = 10
    mu, sigma = 4, 1 # these are used for radius normal distirbution
    mu_circle = (bound_x + 2*start_x)/2
    sigma_circle = bound_x/4
    max_attempts = 20

    for i in range(num_obstacles):
        
        placed = False
        place_attempts = 0
        if mode == 'normal':
            cand_r = sample_radius_normal(mu,sigma,r_bound)
            while not placed:
                # cand_x,cand_y = sample_point_uniform(cand_r,bound_x,bound_y,start_x,start_y)
                cand_x,cand_y = sample_point_normal(cand_r,bound_x,bound_y,mu_circle,sigma_circle,start_x,start_y)
                cand_obs = format_circle(cand_r,cand_x,cand_y)
                valid = check_placement(cand_obs,obstacles)
                if valid:
                    obstacles.append(cand_obs)
                    placed = True
                    if False: #change to True if debugging positions of obstacles w.r.t. bounds
                        print(f"x0={cand_obs[1]} lb={start_x+cand_obs[0]} ub={start_x+bound_x-cand_obs[0]}; y0={cand_obs[2]} lb={start_y+cand_obs[0]} ub={start_y+bound_y-cand_obs[0]}")
                else:
                    place_attempts += 1
                    if place_attempts > max_attempts:
                        place_attempts = 0
                        cand_r = sample_radius_normal(mu,sigma,r_bound)
            
            

        elif mode == 'uniform':
            max_attempts = 10
            cand_r = sample_radius_uniform(r_bound)
            while not placed:
                # cand_x,cand_y = sample_point_uniform(cand_r,bound_x,bound_y,start_x,start_y)
                cand_x,cand_y = sample_point_uniform(cand_r,bound_x,bound_y,start_x,start_y)
                cand_obs = format_circle(cand_r,cand_x,cand_y)
                valid = check_placement(cand_obs,obstacles)
                if valid:
                    obstacles.append(cand_obs)
                    placed = True
                    if False: #change to True if debugging positions of obstacles w.r.t. bounds
                        print(f"x0={cand_obs[1]} lb={start_x+cand_obs[0]} ub={start_x+bound_x-cand_obs[0]}; y0={cand_obs[2]} lb={start_y+cand_obs[0]} ub={start_y+bound_y-cand_obs[0]}")
                else:
                    place_attempts += 1
                    if place_attempts > max_attempts:
                        place_attempts = 0
                        cand_r = sample_radius_uniform(r_bound)
        
        else:
            raise Exception(f"invalid mode: {mode} inputted")
        
    if mode == 'normal':
        if output_result:
            with open(fname,"a") as f:
                f.write("New Obstacle Set:\n")
                f.write(f"# obs = {num_obstacles},\n")
                f.write(f"radius bounds, mu, sigma = ({0.1}-{r_bound},{mu},{sigma})\n")
                f.write(f"x bounds, mu, sigma = ({start_x}-{start_x + bound_x},{mu_circle},{sigma_circle})\n")
                f.write(f"y bounds, mu, sigma = ({start_y}-{start_y + bound_y},{mu_circle},{sigma_circle})\n")
                # output file requirement is that radius,x,y is written line before the obstacles
                f.write("radius,x,y\n")
                for obs in obstacles:
                    f.write(f"{obs[0]},{obs[1]},{obs[2]}\n")
    
    if mode == 'uniform':
        if output_result:
            with open(fname,"a") as f:
                f.write("New Obstacle Set:\n")
                f.write(f"# obs = {num_obstacles},\n")
                f.write(f"radius bounds = ({0.1}-{r_bound})\n")
                f.write(f"x bounds = ({start_x}-{start_x + bound_x})\n")
                f.write(f"y bounds, mu, sigma = ({start_y}-{start_y + bound_y},{mu_circle},{sigma_circle})\n")
                # output file requirement is that radius,x,y is written line before the obstacles
                f.write("radius,x,y\n")
                for obs in obstacles:
                    f.write(f"{obs[0]},{obs[1]},{obs[2]}\n")

    if show_result:
        fig,axs = plt.subplots()
        plot_obstacles(obstacles,axs)
        axs.set_xlim(start_x,start_x+bound_x)
        axs.set_ylim(start_y,start_y+bound_y)
        axs.set_aspect('equal')
        plt.show()

def gen_multi_courses(num_obs):
    fname = f'{num_obs}_obstacle_locations.txt'
    for ii in range(num_obs):
        gen_obs(fname=fname)

def parse_input():
    parser = ArgumentParser(description="obstacle course generator",formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-b", "--batch", dest="batch", action='store_const', const=True, default = False, help="Creates unique file name and does not display courses via plot")
    parser.add_argument("-s", "--start", type=float, default = [5,0], nargs=2, help='specify start boundaries for x and y of obstacle course')
    parser.add_argument("-r", "--range", type=float, default = [20,30], nargs=2, help="range of obstacle course in x and y")
    parser.add_argument("-no","--num_obstacles", type=int, default = 10, help="Number of obstacles per course")
    parser.add_argument("-nc","--num_courses", type=int, default = 10, help="Number of courses to make")
    parser.add_argument("-f","--fname_out", default = "obstacle_locations.txt", help="argument to specify custom file name")
    parser.add_argument("-o", "--opt_thread", dest="opt_divide", action='store_const', const=True, default=False, help='Divides files to match number of threads on running.  Assumes 16 threads')
    parser.add_argument("-u","--uniform_mode", dest="mode", action='store_const', const='uniform', default='normal',help='Changes mode placement to be done with uniform distribution instead of normal distribution')
    
    args = parser.parse_args()
    args = vars(args)
    return args

if __name__ == "__main__":

    args = parse_input()
    batch = args["batch"]
    obstacles = args["num_obstacles"]
    start_x = args["start"][0] #TODO update this to not use int conversion
    start_y = args["start"][1]
    bound_x = args["range"][0]
    bound_y = args["range"][1]
    gen_multi_courses = True
    divide_by_course = False
    output_folder = "./obs_courses/"

    tic = time.perf_counter()
    if not batch:
        courses = args["num_courses"]
        today = date.today()
        formatted_date = today.strftime("%y_%m_%d")
        if args["fname_out"] == "obstacle_locations.txt":
            fname = f"{formatted_date}_{courses}_courses_{obstacles}_obstacles_normal.txt"
        else:
            fname = args["fname_out"] + '.txt'
        for ii in range(courses):
            gen_obs(num_obstacles=args["num_obstacles"],fname=fname,start_x=start_x,start_y=start_y,bound_x=bound_x,bound_y=bound_y,mode=args["mode"])
        
    elif gen_multi_courses: 
        courses = args["num_courses"]
        today = date.today()
        formatted_date = today.strftime("%y_%m_%d")
        num_files = 0 # number of files created
        
        for obs_num in range(obstacles):
            
                if divide_by_course:
                    for ii in range(courses):
                        if ii % 5 == 0: 
                            if args["fname_out"] == "obstacle_locations.txt":
                                fname = f"{formatted_date}_{courses}_courses_{obs_num+1}_obstacles_normal.txt"
                            else:
                                fname = f'{args["fname_out"]}_{num_files}.txt'
                                num_files += 1

                        gen_obs(num_obstacles=obs_num+1,fname=output_folder+fname,start_x=start_x,start_y=start_y,bound_x=bound_x,bound_y=bound_y,mode=args["mode"])
                else:
                    if args["opt_divide"]:
                        threads = 16
                        obs_per_file = math.floor(obstacles/threads)
                        num_extras = obstacles % threads
                        if num_files <= num_extras:
                            file_divider = obs_per_file + 1
                        else:
                            file_divider = obs_per_file
                    else:
                        file_divider = 1
                    obs_per_file = file_divider
                    if obs_num % obs_per_file == 0: # every third file after 1st is new_file
                        if args["fname_out"] == "obstacle_locations.txt":
                            fname = f"{formatted_date}_{courses}_courses_{obs_num+1}_obstacles_normal.txt"
                        else:
                            fname = f'{args["fname_out"]}_{num_files}.txt'
                            num_files += 1
                    
                    for ii in range(courses):
                        gen_obs(num_obstacles=obs_num+1,fname=output_folder+fname,start_x=start_x,start_y=start_y,bound_x=bound_x,bound_y=bound_y,mode=args["mode"])
    else:
        gen_obs(num_obstacles=args["num_obstacles"],show_result=True,start_x=start_x,start_y=start_y,bound_x=bound_x,bound_y=bound_y,mode=args["mode"])
        # courses = args["num_courses"]
        # today = date.today()
        # formatted_date = today.strftime("%y_%m_%d")
        # if args["fname_out"] == "obstacle_locations.txt":
        #     fname = f"{formatted_date}_{courses}_courses_{obstacles}_obstacles_normal.txt"
        # else:
        #     fname = args["fname_out"] + '.txt'
        # for ii in range(courses):
        #     gen_obs(num_obstacles=args["num_obstacles"],fname=fname,start_x=start_x,start_y=start_y,bound_x=bound_x,bound_y=bound_y)


toc = time.perf_counter()
print(f"created the data in {toc - tic:0.4f} seconds for file: {fname}")
