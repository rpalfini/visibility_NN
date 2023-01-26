# from visibility_graph import *
import visibility_graph as vg
from matplotlib import pyplot as plt
import os
import time
import WindowsInhibitor as wi #prevents computer from sleeping while generating data
from datetime import date
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from obstacle_course_gen import convert2bool


# example python vis_main.py 1_courses_5_obstacles_normal.txt 
args = vg.arg_parse()

#TODO fully add batch code which involves loading start/end values, maybe make it a function in vis_graph where it can generate a range of start/end values we want to test  
# batch = convert2bool(args["batch"])
batch = args["batch"]

if batch:
    osSleep = None
    if os.name == 'nt':
        osSleep = wi.WindowsInhibitor()
        osSleep.inhibit()


# start_vals = [(0,3)]
# end_vals = [(30,15)]
start_vals = args["start"]
end_vals = args["end"]


# obs_file_path = args["obs_path"] + args["fname"]
obs_file_path = args["obs_fpath"]
obs_courses_dict = vg.read_obstacle_list(obs_file_path)
obstacle_list = obs_courses_dict[args["course"]]



# columns = ['start_x','start_y']

# create start/end points
start_list = vg.init_points(start_vals)
end_list = vg.init_points(end_vals)

# create obstacle list
# obstacle_list = vg.init_obs(obst_locations,radius1)

tic = time.perf_counter()

vg_gen = vg.visibility_graph_generator()
if args["test_mode"]:
    vg_gen.run_test(start_list,end_list,obstacle_list,algorithm="dijkstra")
    vg_gen.run_test(start_list,end_list,obstacle_list,algorithm="AStar")
    plt.show()
    vg_gen.plot_solution(0,"dijkstra")
    vg_gen.plot_solution(1,"AStar")
else:
    vg_gen.run_test(start_list,end_list,obstacle_list,algorithm=args["solve_option"])
    vg_gen.plot_solution(0,"env 3_0")

toc = time.perf_counter()
print(f"created the data in {toc - tic:0.4f} seconds")

vg_gen.output_csv(f'{args["fname"]}_obs_data')
vg_gen.save_plot_image(f'{args["fname"]}_obs_fig')

# today = date.today()
# vg_gen.output_csv(today.strftime("%Y_%m_%d")+'three_obst data_large_1')

plt.show()

if batch:
    if osSleep:
        osSleep.uninhibit()

