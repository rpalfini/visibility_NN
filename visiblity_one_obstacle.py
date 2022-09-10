from visibility_graph import *

start_vals = [(3,9)]
end_vals = [(23,9)]
radius1 = 5 #obstacle radius
loc1 = [(13,10)]

start_list = init_points(start_vals)
end_list = init_points(end_vals)

# create obstacle list
obstacle_list = init_obs(loc1,radius1)

vis_graph_one_obst = visibility_graph(obstacle_list)
vis_graph_one_obst.run_test(start_vals,end_vals)
vis_graph_one_obst.plot_solution(0,"env 1_0")

# vis_graph_one_obst.output_csv('test_out')
vis_graph_one_obst.save_plot_image('one_obs_fig')


plt.show()



