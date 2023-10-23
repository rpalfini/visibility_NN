import numpy as np
import matplotlib.pyplot as plt
import pickle
import vis_main
import visibility_graph as vg
import matplotlib.patches as mpatches

def fixed_x_r_visual(course_path,use_default_course=True):
    # this code visiualizes the decision boundary for problem with fixed values except for y_start and y_end

    # Define the circle parameters
    circle_radius = 5
    circle_center = (15, 10)

    # Define the fixed x values for start and end points
    x_start = 2
    x_end = 28

    # Create a mesh grid of y coordinates for the start and end points
    start = 0
    end = 25
    grid_size = 100
    y_start = np.linspace(start,end, grid_size)
    y_end = np.linspace(start,end, grid_size)

    # Create an empty grid to store the decision class
    decision_grid = np.zeros((len(y_start), len(y_end)))

    # load needed arguments for calling vis_main
    pickle_file_path = "boundary_args.pickle"
    with open(pickle_file_path, 'rb') as file:
        vm_args = pickle.load(file) #vis_main args

    if not use_default_course:
        vm_args = replace_course_file(vm_args,course_path)


    # Calculate the decision class for each combination of y_start and y_end
    for ii,y_s in enumerate(y_start):
        for jj,y_e in enumerate(y_end):
            new_args = replace_start_end(vm_args,x_start,y_s,x_end,y_e)
            vg_gen = vis_main.main(new_args)
            label_dir = vg_gen.vis_data[0,-2]
            # Set the decision class based on the distance from the circle center
            decision_grid[ii, jj] = label_dir
            # if label_dir:
            #     decision_grid[ii, jj] = 1
            # else:
            #     decision_grid[ii, jj] = 0

    # Create a contour plot to visualize the decision boundary
    

    plt.close('all')
    fig, ax = plt.subplots()
    contour = plt.contourf(y_start, y_end, decision_grid, levels=[-0.5, 0.5, 1.5], colors=['blue', 'red'])
    # plt.matshow(decision_grid,cmap='bwr')
    # Create custom proxy artists for the legend
    proxy_down = mpatches.Patch(color='blue', label='Down')
    proxy_up = mpatches.Patch(color='red', label='Up')

    # Add the legend with custom proxy artists
    plt.legend(handles=[proxy_down, proxy_up], title="Direction", loc='upper right')
    plt.xlabel("Start Y Coordinate")
    plt.ylabel("End Y Coordinate")
    plt.title("Decision Boundary for Fixed x and Obstacle Radius")
    ax.set_aspect('equal')
    # # Custom labels for the legend
    # labels = ['Down', 'Up']

    # # Add the legend with custom labels
    # plt.legend(labels, title="Direction", loc='upper right')
    # plt.legend([contour],['levels'],loc='upper right')
    plt.show()
    # plt.close('all')
    print('hello')

def replace_start_end(args,start_x,start_y,end_x,end_y):
    args["start"] = [[start_x,start_y]]
    args["end"] = [[end_x,end_y]]
    return args
    
def replace_course_file(args,new_fname):
    args["obs_fpath"] = new_fname
    return args

if __name__ == "__main__":
    use_default_course = False # this defaults to using the file saved in the default arguments
    f_test = "./obs_courses/boundary_visual_2.txt"
    fixed_x_r_visual(f_test,use_default_course=use_default_course)