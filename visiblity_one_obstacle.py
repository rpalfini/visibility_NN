from visibility_graph import *

# start_vals = [(3,9)]
start_vals = [(3,9),(7,17)]
# end_vals = [(23,16)]
# radius1 = 1 #obstacle radius
# end_vals = [(23,9)]
end_vals = [(23,9),(19,5)]
radius1 = 5

obst_locations = [(13,10)]

#TODO what does specigying the dtype=object when creating the ndarray
# trying out large data file
# x bounds should be from number to right edge of obstacle location



# initialize point lists
start_list = init_points(start_vals)
end_list = init_points(end_vals)
# obst_list = init_points(obst_locations)

# create obstacle list
obstacle_list = init_obs(obst_locations,radius1)

vis_graph_one_obst = visibility_graph_generator(obstacle_list)

vis_graph_one_obst.run_test(start_list,end_list,obstacle_list)

# vis_graph_one_obst.plot_env(0,"env 1_0")
vis_graph_one_obst.plot_solution(0,"env 1_0")
# vis_graph_one_obst.plot_full_vis_graph(0,"env 1_0")
# vis_graph_one_obst.show_graph()

vis_graph_one_obst.output_csv('2022_10_16one_obst data')
# vis_graph_one_obst.save_plot_image('sol_r=1')

plt.show()



