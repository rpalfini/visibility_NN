from visibility_graph import *
import time
import WindowsInhibitor as wi #prevents computer from sleeping while generating data
from datetime import date
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(description="obstacle testing file",formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-b", "--batch", default = False, help="Creates unique file name and does not display courses")
parser.add_argument("-p", "--obs_path", default = ".\\obs_courses\\",help="path for finding obstacle course")
parser.add_argument("fname", default = 10, help="Number of obstacles per course")
parser.add_argument("num_courses", default = 10, help="Number of courses to make")
args = parser.parse_args()
args = vars(args)

batch = args["batch"]

if batch:
    osSleep = None
    if os.name == 'nt':
        osSleep = wi.WindowsInhibitor()
        osSleep.inhibit()


start_vals = [(2,3)]
end_vals = [(24,14)]

obst_locations = [(6,5),(13,9),(20,6)]
radius1 = [2,3,2] #obstacle radius

# columns = ['start_x','start_y']

# create start/end points
start_list = init_points(start_vals)
end_list = init_points(end_vals)

# create obstacle list
obstacle_list = init_obs(obst_locations,radius1)

tic = time.perf_counter()

vg_gen = visibility_graph_generator()
vg_gen.run_test(start_list,end_list,obstacle_list, sys.argv[1] if (len(sys.argv) > 1) else "Djikstra")
vg_gen.plot_solution(0,"env 3_0")

toc = time.perf_counter()
print(f"created the data in {toc - tic:0.4f} seconds")

vg_gen.output_csv('three_obs_data')
vg_gen.save_plot_image('three_obs_fig')

# today = date.today()
# vg_gen.output_csv(today.strftime("%Y_%m_%d")+'three_obst data_large_1')

plt.show()

if osSleep:
    osSleep.uninhibit()

