from tracemalloc import start
from turtle import st
from unicodedata import is_normalized
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dijkstra import *

def init_points(point_list):
    point_obj = []
    for p in point_list:
        point_val = point(p)
        point_obj.append(point_val)
    return point_obj

def init_obs(obs_list,radius):
    #TODO This only allows for one obstacle radius size
    # obs_list should be a list of point objects
    obs_obj = []
    for obs in obs_list:
        obs_val = obstacle(radius,obs)
        obs_obj.append(obs_val)
    return obs_obj

class point:
    def __init__(self,point_coord):
        self.x = point_coord[0]
        self.y = point_coord[1]

    def view(self):
        print("(" + str(self.x) + "," + str(self.y) + ")")

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

    def clear_nodes(self):
        self.node_list = []

class visibility_graph_generator:
    # variables for buidling vis graph
    node_dict = {} # stores the points associated with each key for a start/end pair
    vis_graph = {} # this is a graph that stores the nodes and their edge distances
    graphs_gen_memory = {} # this dictionary stores the start/end, graph created, and a node_point_dictionary, used for plotting

    # variables for outputting training data
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

    #vis graph methods
    def run_test(self,start_list,end_list):
        # main function that creates training data for start/end points
        for start in start_list:
            for end in end_list:
                self.build_vis_graph(start,end) # method that calculates visibility graph
                
                # method that calculates shortest distance, djikstra algo
                # create labels from djikstra algo
                direction_label = 0
                self.record_result(start,end,direction_label)
                # reset obstacles
                self.clear_node_dict()
                self.clear_obst_nodes()

    def build_vis_graph(self,start,end):
        # builds tangent visibility graph of obstacles for a start/end pair
        # add start and end to node dictionary
        self.add_node2dict(start,"start")
        self.add_node2dict(end,"end")
        # check if start/end is visible
        if self.is_node_vis

        for obstacle in self.obstacles:
            self.vis_point_obst(start,obstacle)
            self.vis_point_obst(end,obstacle)
        return

    def vis_point_obst(self,start_node,obstacle):
        # calculates visibility graph node to obstacle
        center_dist = self.euclid_dist(start_node,obstacle.center_loc) # distance from obstacle center to point
        theta = np.arccos(obstacle.radius/center_dist)
        phi = self.rotation_to_horiz(obstacle.center_loc,start_node)
        
        can_node1 = point(self.direction_step(obstacle.center_loc,obstacle.radius,phi + theta)) #candidate node1
        if self.is_node_vis(start_node,can_node1):
            self.add_node2dict(can_node1)
            self.add_edge2graph()
        
        can_node2 = point(self.direction_step(obstacle.center_loc,obstacle.radius,phi - theta)) #candidate node2
        if self.is_node_vis(start_node,can_node2):
            self.add_node2dict(can_node2)
            self.add_edge2graph()
        
        return
    
    def direction_step(self,start,dist,angle):
        # calculates tangent node location on an obstacle
        x = dist*np.cos(angle) + start.x
        y = dist*np.sin(angle) + start.y
        return (x,y)

    def rotation_to_horiz(self,point1,point2):
        # calculates rotation of connecting line to horizontal axis
        dy = point2.y - point1.y
        dx = point2.x - point1.x
        rotation_1_2 = np.arctan2(dy,dx)
        return rotation_1_2

    def euclid_dist(self,point1,point2):
        # calculates euclidean distance b/T two poitns
        dist = np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
        return dist

    def vis_obst_obst(self):
        # calculates visibility graph tangent lines between obstacles
        
        return

    def process_cand_node(self,start_node,cand_node):
        # this method adds edge to node dictionary and vis_graph, if the node is visible
        if self.is_node_vis(start_node,cand_node):
            self.add_node2dict(cand_node)
            edge_length = self.euclid_dist(start_node,cand_node)
            self.add_edge2graph()

    def is_node_vis(self,start_node,end_node):
        # checks if visibility line intersects other obstacles
        for obstacle in self.obstacles:
            d = 0

        is_valid = True
        return is_valid

    ## graph dict methods
    def add_graph(self):
        return

    ## node dict methods
    def add_node2dict(self,point,label=None):
        if label == None:
            num_keys = len(self.node_dict)
            self.node_dict[num_keys] = point
        else:
            self.node_dict[label] = point

    def clear_node_dict(self):
        self.node_dict = {}

    def clear_obst_nodes(self):
        for obstacle in self.obstacles:
            obstacle.clear_nodes()
   
    ## output methods
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
