import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def init_points(point_list):
    point_obj = []
    for p in point_list:
        point_val = point(p)
        point_obj.append(point_val)
    return point_obj

def init_obs(obs_list,radius):
    obs_obj = []
    for o in obs_list:
        obs_val = obstacle(radius,o)
        obs_obj.append(obs_val)
    return obs_obj


class point:
    def __init__(self,point_coord):
        self.x = point_coord[0]
        self.y = point_coord[1]

class obstacle:
    def __init__(self,radius,center_loc):
        self.radius = radius
        # self.center_x = center_loc[0]
        # self.center_y = center_loc[1]
        self.center_x = center_loc.x
        self.center_y = center_loc.y
        self.center_loc = center_loc # this should be a point object
        self.node_list = []

    def add_node(self,point):
        self.node_list.append(point)

 
class visibility_graph:
    df_columns = ['start','end','obst1_dir']
    num_col = 5;
    vis_data = np.array([],dtype = np.double).reshape(0,num_col) # formatted start_x, start_y, end_x, end_y, direction label
    vis_df = pd.DataFrame(columns = df_columns) #unsure whether to use dataframe or array
    # variables for plotting
    axis_xlim = [0, 30] # fixed graph axis limits, will set to be size for floor
    axis_ylim = [0, 20]
    line_width = 3

    def __init__(self,obstacles=[]):
        # self.start = start # formatted (x,y)
        # self.end = end
        self.obstacles = obstacles
        # visibility viewer initialization
        self.fig, self.vis_axs = plt.subplots(1,1)
        self.vis_axs.set_xlim(self.axis_xlim)
        self.vis_axs.set_ylim(self.axis_ylim)
        self.vis_axs.grid(visible=True)
        self.vis_axs.set_aspect('equal')

    #TODO add method for updating obastcles list when needed

    def run_test(self,start_list,end_list):
        for start in start_list:
            for end in end_list:
                # method that calculates visibility graph
                # method that calculates shortest distance, djikstra algo
                # create labels from djikstra algo
                direction_label = 0
                self.record_result(start,end,direction_label)

    def vis_point_obst(self,point,obstacle):
        # calculates visibility graph point to obstacle
        center_dist = self.euclid_dist(point,obstacle.center_loc) # distance from obstacle center to point
        theta = np.arccos(obstacle.radius/center_dist)
        phi = self.rotation_to_horiz(obstacle.center_loc,point)
        node1 = point(self.direction_step(obstacle.center_loc,obstacle.radius,phi + theta))
        node2 = point(self.direction_step(obstacle.center_loc,obstacle.radius,phi + theta))
        return
    
    def direction_step(self,start,dist,angle):
        return

    def rotation_to_horiz(self,point1,point2):
        dy = point2.y - point1.y
        dx = point2.x - point1.x
        rotation_1_2 = np.arctan2(dy,dx)
        return rotation_1_2

    def euclid_dist(self,point1,point2):
        dist = np.sqrt((point1.x - point2.x)^2 + (point1.y - point2.y)^2)
        return dist

    def vis_obst_obst(self):
        # calculates visibility graph tangent lines between obstacles
        
        
        return

    def is_vis_valid(self):
        # checks if visibility line intersects other obstacles
        is_valid = True
        return is_valid

    def calc_vis_graph(self):
        return

    def record_result(self,start,end,direction_label):
        # result_df = pd.DataFrame()
        # self.vis_df = pd.concat([self.vis_df, result_df])
        results_array = np.array([start.x, start.y, end.x, end.y, direction_label]).reshape(1,self.num_col) # direction label of 1 is up and 0 is down
        self.vis_data = np.concatenate([self.vis_data, results_array])

    def output_csv(self,file_name):
        # output training data to file
        np.savetxt(file_name+'.csv', self.vis_data, delimiter=",")

    ## plot viewer methods
    def plot_solution(self,test_num,title=None):
        # plots obstacles and solution
        self.plot_start_end(test_num)
        self.plot_obstacles()
        self.plot_shortest_path(test_num)
        self.vis_axs.set_title(title)
        self.vis_axs.legend()

    # def plot_shortest_path(self):
    def plot_start_end(self,test_num):
        data = self.vis_data[test_num,:]
        self.vis_axs.scatter(data[0],data[1],color='red',marker="^",linewidth=self.line_width,label="start")
        self.vis_axs.scatter(data[2],data[3],color='green',marker="o",linewidth=self.line_width,label="end")

    def plot_obstacles(self):
        # plots obstacles
        for obstacle in self.obstacles:
            obst_x, obst_y = self.make_circle_points(obstacle)
            # self.vis_axs.plot(obst_x, obst_y,color='blue',linewidth=self.line_width,label="obstacle")
            self.vis_axs.plot(obst_x, obst_y,color='blue',linewidth=self.line_width)

    def make_circle_points(self,obstacle):
        thetas = np.linspace(0,2*np.pi,100)
        radius = obstacle.radius
        bound_x = radius*np.cos(thetas) + obstacle.center_x
        bound_y = radius*np.sin(thetas) + obstacle.center_y
        return bound_x, bound_y

    def plot_shortest_path(self,test_num):
        #TODO once I have visibility graph data I will make the plot
        return

    def clear_plot(self):
        self.vis_axs.cla()

    def save_plot_image(self,fig_name):
        self.fig.savefig(fig_name + '.png')
        # creates .png of visibility graph
