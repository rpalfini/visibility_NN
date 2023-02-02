import visibility_graph as vg
from matplotlib import pyplot as plt
import os
import time
import WindowsInhibitor as wi #prevents computer from sleeping while generating data
from datetime import date
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from obstacle_course_gen import convert2bool

# This file tests all the plotting commands in visibility_graph

start_vals = [(0,3)]
end_vals = [(30,15)]

# course 1
obs_file_path = "./obs_courses/23_01_20_1_courses_5_obstacles_normal.txt"
obs_courses_dict = vg.read_obstacle_list(obs_file_path)
obstacle_list = obs_courses_dict[0]

# course 2
obs_file_path2 = "./obs_courses/23_01_20_1_courses_6_obstacles_normal.txt"
obs_courses_dict2 = vg.read_obstacle_list(obs_file_path2)
obstacle_list2 = obs_courses_dict2[0]

# create start/end points
start_list = vg.init_points(start_vals)
end_list = vg.init_points(end_vals)

vg_course1 = vg.visibility_graph_generator(is_ion=True)
vg_course1.run_test(start_list,end_list,obstacle_list)
vg_course2 = vg.visibility_graph_generator(is_ion=True)
vg_course2.run_test(start_list,end_list,obstacle_list2)

vg_course1.plot_env(0,'env')
vg_course2.plot_env(0,'env')
vg_course1.plot_solution(0,'sol')
vg_course2.plot_solution(0,'sol')
vg_course1.plot_full_vis_graph(0)
vg_course2.plot_full_vis_graph(0)
vg_course1.plot_network(0)
vg_course2.plot_network(0)







