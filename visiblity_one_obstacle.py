from visibility_graph import *

start_vals = [(3,9)]
# end_vals = [(23,16)]
end_vals = [(23,9)]
radius1 = 5
# radius1 = 1 #obstacle radius
obst_locations = [(13,10)]

# initialize point lists
start_list = init_points(start_vals)
end_list = init_points(end_vals)
obst_list = init_points(obst_locations)

# create obstacle list
obstacle_list = init_obs(obst_list,radius1)

vis_graph_one_obst = visibility_graph_generator(obstacle_list)

vis_graph_one_obst.run_test(start_list,end_list)

vis_graph_one_obst.plot_solution(0,"env 1_0")

# vis_graph_one_obst.show_graph()
# plt.show()



# vis_graph_one_obst.output_csv('test_out')
vis_graph_one_obst.save_plot_image('test_r=5')





