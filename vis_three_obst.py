from visibility_graph import *
import time
import WindowsInhibitor as wi #prevents computer from sleeping while generating data
from datetime import date

import sys

osSleep = None

if os.name == 'nt':
    osSleep = wi.WindowsInhibitor()
    osSleep.inhibit()

batch = False

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
vg_gen.run_test(start_list,end_list,obstacle_list, sys.argv[1] if (len(sys.argv) > 1) else "Dijkstra")
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

