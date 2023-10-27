import numpy as np
import matplotlib.pyplot as plt
import pickle
import vis_main
import visibility_graph as vg
import matplotlib.patches as mpatches
import time


def create_3d_decision_boundary(obs_file,num_obs):
    tic = time.perf_counter()
    # Define the fixed x values for start and end points
    x_range_start = 0
    x_range_end = 4.9
    x_end = 28

    # Create a mesh grid of y coordinates for the start and end points
    start = 0
    end = 25
    # r_small = 0.1
    # r_large = 10
    grid_size = 5
    print(f'grid size is {grid_size}')
    x_range = np.linspace(x_range_start,x_range_end, grid_size)
    y_start = np.linspace(start,end, grid_size)
    y_end = np.linspace(start,end, grid_size)


    # Create an empty grid to store the decision class
    d_grid_list = []
    for ii in range(num_obs):
        d_grid_list.append(np.zeros((len(y_start), len(y_end), len(x_range))))    


    # load needed arguments for calling vis_main
    pickle_file_path = "boundary_args.pickle"
    with open(pickle_file_path, 'rb') as file:
        vm_args = pickle.load(file) #vis_main args

    # change course file from default to desired
    vm_args = replace_course_file(vm_args,obs_file)


    # Calculate the decision class for each combination of y_start and y_end
    for ii,y_s in enumerate(y_start):
        for jj,y_e in enumerate(y_end):
            for ll,x_s in enumerate(x_range):
                new_args = replace_start_end(vm_args,x_s,y_s,x_end,y_e)
                vg_gen = vis_main.main(new_args)
                for kk in range(num_obs):
                    label_idx = -(num_obs+1-kk)
                    label_dir = vg_gen.vis_data[0,label_idx]
                    d_grid_list[kk][ii,jj,ll] = label_dir

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    data = d_grid_list[0]
    x, y, z = np.indices(data.shape)
    # Filter data to get the boundary values
    # boundary_data = data ^ np.roll(data, shift=1, axis=0)
    # boundary_data |= data ^ np.roll(data, shift=1, axis=1)
    # boundary_data |= data ^ np.roll(data, shift=1, axis=2)
    threshold = 0.5
    boundary_data = data > threshold
    mask = boundary_data
    # Plot the decision boundary surface
    ax.plot_surface(x[mask], y[mask], z[mask], cmap='coolwarm', alpha=0.5)
    # Customize labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.title('3D Decision Boundary Plot')

    # subplots = create_subplots(num_obs,is3D=True)
    
    # for ii in range(num_obs):
    #     subplot = subplots[ii]
    #     # subplot.contourf(y_start,y_end,d_grid_list[ii],levels=[-0.5, 0.5, 1.5], colors=['blue', 'red'])
    #     # subplot.set_xlabel("Start Y Coordinate")
    #     # subplot.set_ylabel("End Y Coordinate")
    #     # subplot.set_title(f'Obstacle {ii+1}')
    #     # subplot.set_aspect('equal')
    #     # Create a meshgrid to represent the data indices
    #     data = d_grid_list[ii]
    #     x, y, z = np.indices(data.shape)

    #     # Filter data to get the boundary values
    #     # boundary_data = data ^ np.roll(data, shift=1, axis=0)
    #     # boundary_data |= data ^ np.roll(data, shift=1, axis=1)
    #     # boundary_data |= data ^ np.roll(data, shift=1, axis=2)
    #     threshold = 0.5
    #     boundary_data = data > threshold

    #     mask = boundary_data

    #     # Plot the decision boundary surface
    #     subplot.plot_surface(x[mask], y[mask], z[mask], cmap='coolwarm', alpha=0.5)

    #     # Customize labels and title
    #     subplot.set_xlabel('X-axis')
    #     subplot.set_ylabel('Y-axis')
    #     subplot.set_zlabel('Z-axis')
    #     plt.title('3D Decision Boundary Plot')
    
    # for ii in range(num_obs, len(subplots)):
    #     subplots[ii].axis('off')
        
    # plt.suptitle("Decision Boundary for Fixed x and Obstacle Radius")
    # plt.tight_layout()
    toc = time.perf_counter()
    print(f"created the data in {toc - tic:0.4f} seconds")
    plt.show()


def create_decision_boundaries(obs_file,num_obs):
    # this code visiualizes the decision boundary for problem with fixed values except for y_start and y_end

    tic = time.perf_counter()
    # Define the fixed x values for start and end points
    x_start = 2
    x_end = 28

    # Create a mesh grid of y coordinates for the start and end points
    start = 0
    end = 25
    grid_size = 5
    print(f'grid size is {grid_size}')
    y_start = np.linspace(start,end, grid_size)
    y_end = np.linspace(start,end, grid_size)

    # Create an empty grid to store the decision class
    d_grid_list = []
    for ii in range(num_obs):
        d_grid_list.append(np.zeros((len(y_start), len(y_end))))    
    # decision_grid = np.zeros((len(y_start), len(y_end)))

    # load needed arguments for calling vis_main
    pickle_file_path = "boundary_args.pickle"
    with open(pickle_file_path, 'rb') as file:
        vm_args = pickle.load(file) #vis_main args

    # change course file from default to desired
    vm_args = replace_course_file(vm_args,obs_file)

    # Calculate the decision class for each combination of y_start and y_end
    for ii,y_s in enumerate(y_start):
        for jj,y_e in enumerate(y_end):
            new_args = replace_start_end(vm_args,x_start,y_s,x_end,y_e)
            vg_gen = vis_main.main(new_args)
            for kk in range(num_obs):
                label_idx = -(num_obs+1-kk)
                label_dir = vg_gen.vis_data[0,label_idx]
                d_grid_list[kk][ii,jj] = label_dir

    subplots = create_subplots(num_obs)
    
    for ii in range(num_obs):
        subplot = subplots[ii]
        subplot.contourf(y_start,y_end,d_grid_list[ii],levels=[-0.5, 0.5, 1.5], colors=['blue', 'red'])
        subplot.set_xlabel("Start Y Coordinate")
        subplot.set_ylabel("End Y Coordinate")
        subplot.set_title(f'Obstacle {ii+1}')
        subplot.set_aspect('equal')
    
    for ii in range(num_obs, len(subplots)):
        subplots[ii].axis('off')
        
    plt.suptitle("Decision Boundary for Fixed x and Obstacle Radius")
    plt.tight_layout()
    toc = time.perf_counter()
    print(f"created the data in {toc - tic:0.4f} seconds")
    plt.show()


def create_subplots(num_subplots,is3D=False):
    # Calculate the number of rows and columns needed for the subplots
    num_rows = int(np.ceil(np.sqrt(num_subplots)))
    num_cols = int(np.ceil(num_subplots / num_rows))

    # Create a list to store the subplots

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))
    if num_obs != 1:
        subplots = axes.flatten()
    else:
        subplots = [axes]

    if is3D:
        for subplot in subplots:
            subplot = fig.add_subplot(111, projection='3d')
    return subplots

    # return subplots

def replace_start_end(args,start_x,start_y,end_x,end_y):
    args["start"] = [[start_x,start_y]]
    args["end"] = [[end_x,end_y]]
    return args
    
def replace_course_file(args,new_fname):
    args["obs_fpath"] = new_fname
    return args

def default_decision_case():
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

    # Calculate the decision class for each combination of y_start and y_end
    for ii,y_s in enumerate(y_start):
        for jj,y_e in enumerate(y_end):
            new_args = replace_start_end(vm_args,x_start,y_s,x_end,y_e)
            vg_gen = vis_main.main(new_args)
            label_dir = vg_gen.vis_data[0,-2]
            # Set the decision class based on the distance from the circle center
            decision_grid[ii, jj] = label_dir

    # Create a contour plot to visualize the decision boundary
    fig, ax = plt.subplots()
    contour = plt.contourf(y_start, y_end, decision_grid, levels=[-0.5, 0.5, 1.5], colors=['blue', 'red'])

    # Create custom proxy artists for the legend
    proxy_down = mpatches.Patch(color='blue', label='Down')
    proxy_up = mpatches.Patch(color='red', label='Up')

    # Add the legend with custom proxy artists
    plt.legend(handles=[proxy_down, proxy_up], title="Direction", loc='upper right')
    plt.xlabel("Start Y Coordinate")
    plt.ylabel("End Y Coordinate")
    plt.title("Decision Boundary for Fixed x and Obstacle Radius")
    ax.set_aspect('equal')
    plt.show()

if __name__ == "__main__":
    # f_test = "./obs_courses/boundary_visual_4.txt"
    f_test = "./obs_courses/boundary_visual.txt"
    # f_test = "./obs_courses/23_10_24_1_courses_20_obstacles_normal.txt"
    num_obs = 1
    # create_decision_boundaries(f_test,num_obs)
    create_3d_decision_boundary(f_test,num_obs)
    # default_decision_case()