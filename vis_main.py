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
parser = ArgumentParser(description="obstacle testing file",formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-b", "--batch", default = False, help="Creates unique file name and does not display courses")
parser.add_argument("-p", "--obs_path", default = r"./obs_courses/",help="path for finding obstacle course")
parser.add_argument("-c","--course", default=0, help="specify obstacle course number when multiple courses available")
#TODO figure out better way to specify start and end points as sometimes we want to use a vector of start/end points
parser.add_argument("-s", "--start", default = [0,3], nargs=2, help='course start point')
parser.add_argument("-e", "--end", default = [30,15], nargs=2, help='course end point')
parser.add_argument("-a", "--astar", dest='solve_option', action='store_const', const='AStar', default='djikstra',help='change shortest path solver from djikstra to AStar')
parser.add_argument("fname", help="Obstacle course file to test")

args = parser.parse_args()
args = vars(args)

batch = convert2bool(args["batch"])

if batch:
    osSleep = None
    if os.name == 'nt':
        osSleep = wi.WindowsInhibitor()
        osSleep.inhibit()


# start_vals = [(0,3)]
# end_vals = [(30,15)]
start_vals = [[float(x) for x in args["start"]]] # double bracket as the input needs to be pair(x,y) in a list
#TODO make start/end point files I can load as start and end or have option that specifies if i am using the batch start/end list or if I am using a signle start end point specified by the user
end_vals = [[float(x) for x in args["end"]]]


obs_file_path = args["obs_path"] + args["fname"]
obs_courses_dict = vg.read_obstacle_list(obs_file_path)
obstacle_list = obs_courses_dict[int(args["course"])]

# obst_locations = [(6,5),(13,9),(20,6)]
# radius1 = [2,3,2] #obstacle radius

# columns = ['start_x','start_y']

# create start/end points
start_list = vg.init_points(start_vals)
end_list = vg.init_points(end_vals)

# create obstacle list
# obstacle_list = vg.init_obs(obst_locations,radius1)

tic = time.perf_counter()

vg_gen = vg.visibility_graph_generator()
# vg_gen.run_test(start_list,end_list,obstacle_list, sys.argv[1] if (len(sys.argv) > 1) else "Djikstra")
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

