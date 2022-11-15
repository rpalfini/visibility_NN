from visibility_graph import *
import time
import WindowsInhibitor as wi #prevents computer from sleeping while generating data
from datetime import date


osSleep = None

if os.name == 'nt':
    osSleep = wi.WindowsInhibitor()
    osSleep.inhibit()

batch = False


start_vals = [(1,9)]
end_vals = [(29,9)]

# locs = [(13,10),(12,4),(20,16),(9,15),(7,7),(21,2.5),(19,9),(25,13)]
obst_locations = [[12,10],[11,4],[19,16],[8,15],[6,7],[24,6],[18,9],[24,13]]
radius1 = 2*np.ones(len(obst_locations)) #obstacle radius

# columns = ['start_x','start_y']

# create start/end points
start_list = init_points(start_vals)
end_list = init_points(end_vals)

# create obstacle list
obstacle_list = init_obs(obst_locations,radius1)

tic = time.perf_counter()

vis_graph_eight_obst = visibility_graph_generator()
vis_graph_eight_obst.run_test(start_list,end_list,obstacle_list)
vis_graph_eight_obst.plot_solution(0,"env 8_0")

toc = time.perf_counter()
print(f"created the data in {toc - tic:0.4f} seconds")

# vis_graph_eight_obst.output_csv('test_out')
# vis_graph_eight_obst.save_plot_image('eight_obs_fig2')

today = date.today()
vis_graph_eight_obst.output_csv(today.strftime("%Y_%m_%d")+'one_obst data_large_1')

# plt.show()

if osSleep:
    osSleep.uninhibit()

