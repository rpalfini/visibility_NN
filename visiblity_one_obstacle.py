from visibility_graph import *
import WindowsInhibitor as wi #prevents computer from sleeping while generating data

from datetime import date
import time
import sys

osSleep = None

if os.name == 'nt':
    osSleep = wi.WindowsInhibitor()
    osSleep.inhibit()

batch = False

# create obstacle list
# radius1 = 5
radius1 = [6]
obst_locations = [(13,10)]
obstacle_list = init_obs(obst_locations,radius1)
if not batch:
    record_on = True
    start_vals = [(3,9)]
    # start_vals = [(3,9),(7,17)]
    # end_vals = [(23,16)]
    # radius1 = 1 #obstacle radius
    end_vals = [(23,9)]
    # end_vals = [(23,9),(19,5)]
    

    #TODO what does specifying the dtype=object when creating the ndarray
    # trying out large data file
    # x bounds should be from number to right edge of obstacle location

    # initialize point lists
    start_list = init_points(start_vals)
    end_list = init_points(end_vals)
    # obst_list = init_points(obst_locations)

    
else:
    # Trying batch start/end vals
    record_on = False
    tol = 0.01
    num_points = 60
    min_x,max_x = find_test_range(obstacle_list)
    start_x = np.linspace(min_x-tol-50,min_x-tol,num_points) # could try using np.arange
    start_y = np.linspace(-10,50,num_points)
    end_x = np.linspace(max_x+tol,max_x+tol+50,num_points)
    end_y = start_y
    start_x = np.linspace(-10,min_x-tol,num_points)
    start_y = np.linspace(-10,50,num_points)
    end_x = np.linspace(max_x+tol,50,num_points)
    end_y = start_y

    start_vals = get_list_points(start_x,start_y)
    end_vals = get_list_points(end_x,end_y)

    start_list = init_points(start_vals)
    end_list = init_points(end_vals)


tic = time.perf_counter()
vis_graph_one_obst = visibility_graph_generator(record_on=record_on)

vis_graph_one_obst.run_test(start_list,end_list,obstacle_list, sys.argv[1] if (len(sys.argv) > 1) else "Djikstra")
toc = time.perf_counter()
print(f"created the data in {toc - tic:0.4f} seconds")
# vis_graph_one_obst.plot_env(0,"env 1_0")
# vis_graph_one_obst.plot_solution(0,"env 1_0")
# vis_graph_one_obst.plot_full_vis_graph(0,"env 1_0")
# vis_graph_one_obst.show_graph()
today = date.today()
vis_graph_one_obst.output_csv(today.strftime("%Y_%m_%d")+'one_obst data_large_1')
# vis_graph_one_obst.save_plot_image('sol_r=1')

if osSleep:
    osSleep.uninhibit()
# plt.show()




