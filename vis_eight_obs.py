from visibility_graph import *
import pandas as pd

start_vals = [(1,9)]
end_vals = [(29,9)]
radius1 = 2 #obstacle radius
# locs = [(13,10),(12,4),(20,16),(9,15),(7,7),(21,2.5),(19,9),(25,13)]
obst_locations = [[12,10],[11,4],[19,16],[8,15],[6,7],[24,6],[18,9],[24,13]]

# columns = ['start_x','start_y']

# create start/end points
start_list = init_points(start_vals)
end_list = init_points(end_vals)
obst_list = init_points(obst_locations)

# create obstacle list
obstacle_list = init_obs(obst_list,radius1)

vis_graph_eight_obst = visibility_graph_generator(obstacle_list)
vis_graph_eight_obst.run_test(start_list,end_list, obstacle_list)
vis_graph_eight_obst.plot_solution(0,"env 8_0")

# vis_graph_eight_obst.output_csv('test_out')
vis_graph_eight_obst.save_plot_image('eight_obs_fig2')

plt.show()



