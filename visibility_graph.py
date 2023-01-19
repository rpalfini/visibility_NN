import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dijkstra as dijk
import a_star as ast
from bidict import bidict
from vis_graph_enum import *
import param_func as pf
import os
import copy
import warnings

class point:
    key_precision = 6

    def __init__(self,point_coord):
        self.x = point_coord[0]
        self.y = point_coord[1]

    def coord_key(self):
        # outputs the points as a tuple to be used as a key in node dictionary
        return round(self.x,self.key_precision), round(self.y,self.key_precision)

    def view(self):
        print("(" + str(self.x) + "," + str(self.y) + ")")
        return(self.x,self.y)

class obstacle:
    def __init__(self,radius,center_loc):
        self.radius = radius
        self.center_x = center_loc.x
        self.center_y = center_loc.y
        self.center_loc = center_loc # this should be a point object
        self.node_list = set() #using set as there shouldn't be more than one of a given node

    def add_node(self,point):
        self.node_list.add(point)

    def output_prop(self):
        '''Outputs the obstacle properties (x,y,radius)'''
        return self.center_x, self.center_y, self.radius

    def clear_nodes(self): #TODO delete this method if copy code is working as intended
        self.node_list = set()

    def view_nodes(self):
        print(f'Nodes in {id(self)}:')
        for node in self.node_list:
            print(f'({node.x},{node.y})')

    def view(self):
        print(f'(x,y): ({self.center_x},{self.center_y}), radius: {self.radius}')
        return(self.radius,self.center_x,self.center_y)

class vis_graph:
    debug = True

    def __init__(self,obstacles):
        self.node_dict = bidict({}) # stores the nodes as associated with each point, bidict allows lookup in both directions
        self.vis_graph = {} #TODO this shouldn't have the class name # this is a graph that stores the nodes and their edge distances
        self.h_graph = {} # Distance from Key Node to Final Node, normally. May also be used for any other heuristic cost
        self.edge_type_dict = {} # matches the format of vis_graph but only records if edge is surfing or hugging for plotting purposes
        self.node_obst_dict = {} # keeps track of which obstacle each node is on, adding this to make plotting arcs
        self.pw_opt_func = {} # record parameters of piecewise function for label evaulation
        self.obstacles = obstacles

    def view_vis_graph(self):
        for key,value in recursive_items(self.vis_graph):
            print(key,value)

    def init_start_end(self,start,end):
        self.start = start
        self.end = end
        # add start and end to node dictionary
        self.add_node2dict(start,"start")
        self.add_node2dict(end,"end")

    def make_obs_vis_graph(self):
        '''creates surfing edges for obstacle to obstacle connections'''
        remaining_obst = copy.copy(self.obstacles) # using this to prevent calculating nodes between obstacles twice
        for obstacle in self.obstacles:
            remaining_obst.remove(obstacle)
            self.vis_obst_obst(obstacle,remaining_obst)

    def build_vis_graph(self,start,end):
        '''builds tangent visibility graph of obstacles for a start/end pair'''
        self.init_start_end(start,end)
        # check if start/end is visible and add to graph if it is
        self.process_cand_node(self.start,self.end,obstacle=None) #TODO if start/end is visible, no need to create the rest of graph, just go to creating labels
        # creates surfing edges for point to obstacle connections
        for obstacle in self.obstacles:
            self.vis_point_obst(self.start,obstacle)
            self.vis_point_obst(self.end,obstacle,is_end_node = True)

        # creates hugging edges on obstacles, this must be done after start/end nodes added
        for obstacle in self.obstacles:
            self.make_hugging_edges(obstacle)
        return

    def build_h_graph(self):
        for node_id in self.node_dict:
            self.h_graph[node_id] = self.euclid_dist(self.get_node_obj(node_id), self.end)

    def update_node_props(self,cand_node,obstacle):
        if self.is_node_new(cand_node):
            self.add_node2dict(cand_node)
            self.add_node2obstacle(obstacle,cand_node)

    def process_cand_node(self,start_node,cand_node,obstacle,is_end_node = False):
        '''this method adds cand_node, and edge to node dictionary and vis_graph, if the node is visible.  
        It also attaches cand_node to obstacle'''
        if True:
            viewer = graph_viewer(self)
            viewer.plot_obstacles(0)
            viewer.plot_cand_edge(start_node,cand_node)
        if self.is_node_vis(start_node,cand_node):
            self.update_node_props(cand_node,obstacle)
            # if start_node is an end_node, then add to graph vertically
            edge_length = self.euclid_dist(start_node,cand_node)
            if is_end_node is False: 
                self.add_edge2graph(start_node,cand_node,edge_length)
            else:
                self.add_edge2graph(cand_node,start_node,edge_length)

    def process_cand_edge(self,node_obst1,node_obst2):
        '''tags nodes to appropriate obstacles, inputs are tuple of cand_node and its obstacle'''
        if self.is_node_vis(node_obst1[0],node_obst2[0]):
            self.update_node_props(node_obst1[0],node_obst1[1]) #tags node1 to obst 1
        self.process_cand_node(node_obst1[0],node_obst2[0],node_obst2[1]) #tags node 2 to obst2 and creates edge in graph
        
    def vis_point_obst(self,start_node,obstacle,is_end_node = False):
        '''calculates visibility graph node to obstacle,
        is_end_node sets the order of edge storage'''
        center_dist = self.euclid_dist(start_node,obstacle.center_loc) # distance from obstacle center to point
        theta = np.arccos(obstacle.radius/center_dist)
        phi = self.rotation_to_horiz(obstacle.center_loc,start_node)
        
        cand_node1 = point(self.direction_step(obstacle.center_loc,obstacle.radius,phi + theta)) #candidate node1
        cand_node2 = point(self.direction_step(obstacle.center_loc,obstacle.radius,phi - theta)) #candidate node2
        self.process_cand_node(start_node,cand_node1,obstacle,is_end_node)
        self.process_cand_node(start_node,cand_node2,obstacle,is_end_node)
    
    def vis_obst_obst(self,obst,remaining_obst):
        '''calculates visibility graph tangent lines between obstacles'''
        for next_obst in remaining_obst:
            diff = vec_sub(obst.center_loc,next_obst.center_loc)
            if diff.x < 0:
                obst_A = obst
                obst_B = next_obst
            elif diff.x > 0:
                obst_A = next_obst
                obst_B = obst
            else:
                warnings.warn('multiple obstacles have same x-coordinate, skipping vis graph')
                continue
            
            if id(obst) == id(next_obst): # dont calculate visibility with itself
                raise Exception('Obstacle not deleted correctly from list')
            
            self.internal_bitangents(obst_A,obst_B)
            self.external_bitangents(obst_A,obst_B)
            # loops through the new nodes with process_cand_node
        

    def internal_bitangents(self,obst_L,obst_R):
        '''Finds internal bitangents, obstacle L should be to the left of obstacle R on x axis'''
        center_dist = self.euclid_dist(obst_L.center_loc,obst_R.center_loc)
        theta = np.arccos((obst_L.radius+obst_R.radius)/center_dist)
        phi_L = self.rotation_to_horiz(obst_L.center_loc,obst_R.center_loc)
        phi_R = self.rotation_to_horiz(obst_R.center_loc,obst_L.center_loc) # this will be phi_L - pi
              
        cand_nodeL1 = point(self.direction_step(obst_L.center_loc,obst_L.radius,phi_L + theta))
        cand_nodeL2 = point(self.direction_step(obst_L.center_loc,obst_L.radius,phi_L - theta))
        cand_nodeR1 = point(self.direction_step(obst_R.center_loc,obst_R.radius,phi_R - theta))
        cand_nodeR2 = point(self.direction_step(obst_R.center_loc,obst_R.radius,phi_R + theta))


        self.process_cand_edge((cand_nodeL1,obst_L),(cand_nodeR2,obst_R))    
        self.process_cand_edge((cand_nodeL2,obst_L),(cand_nodeR1,obst_R))
        
    def external_bitangents(self,obst_L,obst_R):
        
        center_dist = self.euclid_dist(obst_L.center_loc,obst_R.center_loc)
        theta = np.arccos(abs(obst_L.radius-obst_R.radius)/center_dist)

        # need to compare obstacle radii to determine angle directions
        if obst_L.radius > obst_R.radius:
            phi = self.rotation_to_horiz(obst_L.center_loc,obst_R.center_loc)
        else:
            phi = self.rotation_to_horiz(obst_R.center_loc,obst_L.center_loc)

        cand_nodeL1 = point(self.direction_step(obst_L.center_loc,obst_L.radius,phi + theta))
        cand_nodeL2 = point(self.direction_step(obst_L.center_loc,obst_L.radius,phi - theta))
        cand_nodeR1 = point(self.direction_step(obst_R.center_loc,obst_R.radius,phi + theta))
        cand_nodeR2 = point(self.direction_step(obst_R.center_loc,obst_R.radius,phi - theta))
        
        self.process_cand_edge((cand_nodeL1,obst_L),(cand_nodeR1,obst_R))    
        self.process_cand_edge((cand_nodeL2,obst_L),(cand_nodeR2,obst_R))


    def is_node_vis(self,start_node,end_node):        
        # checks if visibility line intersects other obstacles
        is_valid = True
        a,b,c = planar_line_form(start_node,end_node)
        #TODO make sure that obstacle we are touching doesnt result in non-visibility
        
        if start_node.x < end_node.x:
            left_node = start_node
            right_node = end_node
        elif start_node.x > end_node.x:
            left_node = end_node
            right_node = start_node
        else:
            raise Exception('start_node and end_node have same x coordinate')
        
        for obstacle in self.obstacles:
            if self.is_obst_between_points(left_node,right_node,obstacle):
                if check_collision(a,b,c,obstacle.center_x,obstacle.center_y,obstacle.radius):
                    is_valid = False
                    break
        return is_valid

    def is_obst_between_points(self,start_node,end_node,obstacle):
        if obstacle.center_x > start_node.x and obstacle.center_x < end_node.x:
            is_between = True
        else:
            is_between = False
        if self.debug:
            print(f'vis_graph.is_obst_between_points()')
            print(f'obstacle at ({obstacle.center_x},{obstacle.center_y})')
            print(f'start_node at ({start_node.x},{start_node.y})')
            print(f'end_node at ({end_node.x},{end_node.y}')
            print(f'is_between = {is_between}\n')
        return is_between

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
        # calculates euclidean distance b/T two points
        dist = np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
        return dist

    def make_hugging_edges(self,obstacle):
        # this function goes through obstacles to create hugging edge node connections
        angs = []
        # calculate angles of points to horizontal
        for node in obstacle.node_list:
            angs.append(find_point_angle(node,obstacle))
        # sort in ascending order
        # sorted_nodes = [x for _, x in sorted(zip(angs,obstacle.node_list),key=lambda pair: pair[0])]
        sorted_angles,sorted_nodes = zip(*sorted(zip(angs,obstacle.node_list),key=lambda pair: pair[0]))
        # calculate hugging edge lengths 
        for idx,node in enumerate(sorted_nodes):
            # calculate connection to 2 adjacent node neighbors(above and below in sorted list)
            # connect above
            if idx+1 == len(sorted_nodes): # if end of list, loop back to beginning
                next_idx = 0
            else:
                next_idx = idx+1
            ang_diff = find_ang_diff(sorted_angles[next_idx],sorted_angles[idx])
            if next_idx == 0: # connection from last to first node needs to be 2pi - ang 
                ang_diff = 2*np.pi-ang_diff
            if self.debug == True:
                print('start node ' + str(node.coord_key())+ ' next node ' + str(sorted_nodes[next_idx].coord_key()))
                print('above ang dif ' + str(ang_diff) + ' aka ' + str(ang_diff*180/np.pi))
            arc_length = ang_diff*obstacle.radius
            self.add_edge2graph(node,sorted_nodes[next_idx],arc_length,is_hugging=True,is_edge_CW=False,ang_rot=ang_diff)
            
            # connect below
            ang_diff = find_ang_diff(sorted_angles[idx-1],sorted_angles[idx])
            if (idx-1) == -1: # connecting start to end node needs to be 2*pi - ang
                ang_diff = 2*np.pi-ang_diff
            arc_length = ang_diff*obstacle.radius
            if self.debug == True:
                print('start node ' + str(node.coord_key())+' next node ' +str(sorted_nodes[idx-1].coord_key()))
                print('below ang dif ' + str(ang_diff) + ' aka ' + str(ang_diff*180/np.pi))
                print()
            self.add_edge2graph(node,sorted_nodes[idx-1],arc_length,is_hugging=True,is_edge_CW=True,ang_rot=ang_diff)

    def add_node2obstacle(self,obstacle,cand_node):
        # this method adds candidate node to obstacle object and adds obstacle to obs_node_dict
        if obstacle != None: # if start/end connection visible #TODO: test if this is needed...
            obstacle.add_node(cand_node)
            self.tag_obstacle2node(obstacle,cand_node)
    
    def tag_obstacle2node(self,obstacle,cand_node):
        cand_node_id = self.get_node_id(cand_node)
        self.node_obst_dict[cand_node_id]['obstacle'] = obstacle

    def find_shortest_path(self):
        start_id = self.get_node_id(self.start)
        end_id = self.get_node_id(self.end)
        self.opt_path = dijk.shortestPath(self.vis_graph,start_id,end_id)
        if self.debug == True:
            print(self.opt_path)

    def find_shortest_path_a_star(self):
        start_id = self.get_node_id(self.start)
        end_id = self.get_node_id(self.end)
        self.opt_path = ast.shortestPath(self.vis_graph, self.h_graph, start_id, end_id)
        if self.debug == True:
            print(self.opt_path)
    
    def create_pw_opt_path_func(self):
        def decide_zero_slope_sign(node_id):
            # this function is to deal with case where we have 0 slope line
            obstacle = self.get_node_obs(node_id)
            node_diff = vec_sub(obstacle.center_loc,self.get_node_obj(node_id))
            if node_diff.y > 0:
                # case where 0 slope at center of circle cannot be the optimal solution so do not need to account for that case
                is_pos = False
            else:
                is_pos = True
            return is_pos

        '''Creates a piecewise function of the optimal path to use for creating object labels'''
        for idx,node_id in enumerate(self.opt_path):
            if node_id == 'start': # equation is up to a node, and there are no points before the start
                continue
            x_key,_ = self.get_node_xy(node_id)
            node_before = self.opt_path[idx-1] #node before node_id
            # we can store
            edge_key = self.get_edge_type(self.is_edge_hugging(node_before,node_id)) # finding edge type between this and previous node
            if edge_key == edge_type.line:
                slope,y_int = slope_int_form(self.get_node_obj(node_before),self.get_node_obj(node_id))
                if slope > 0:
                    is_pos = True
                elif slope < 0:
                    is_pos = False
                elif slope == 0:
                    is_pos = decide_zero_slope_sign(node_id)
                self.pw_opt_func[x_key] = pf.line(slope,y_int,is_pos)
            elif edge_key == edge_type.circle:
                is_slope_pos = self.pw_opt_func[self.get_x_key_smaller(x_key)].is_slope_pos() # determine if previous segment has pos or neg slope
                obstacle = self.get_node_obs(node_id)
                self.pw_opt_func[x_key] = pf.circle(*obstacle.output_prop(),is_slope_pos)

    # functions for working with pw_opt_func
    def gen_obs_labels(self):
        # this needs to be called after a shortest_path is found
        self.obstacle_labels = [] # records the labels for the obstacles
        for obstacle in self.obstacles:
            self.obstacle_labels.append(self.create_label(obstacle))
        return self.obstacle_labels

    def get_obs_prop(self):
        '''Outputs list of obs centers in order of labels, in order x,y,r for each object'''
        # used for recording results
        prop_list = [x.output_prop() for x in self.obstacles]
        output_list = []
        # unpacking tuples to make property list for outputting in record results
        for prop in prop_list:
            for att in prop:
                output_list.append(att)
        return output_list

    def get_x_key_smaller(self,x):
        ''' finds largest x_key in pw_opt_func that is less than the input x'''
        # x_keys = [*self.pw_opt_func]
        x_keys = [key for key in self.pw_opt_func.keys() if key<x]
        return max(x_keys)

    # used when evaluating pw_opt_func
    def get_x_key_larger(self,x):
        ''' finds smallest x_key in pw_opt_func that is greater than the input x'''
        x_keys = [key for key in self.pw_opt_func.keys() if key>x]
        return min(x_keys)
    
    def create_label(self,obstacle):
        x_key = self.get_x_key_larger(obstacle.center_x)
        y_path = self.pw_opt_func[x_key].evaluate(obstacle.center_x)
        return dir_label.up if y_path > obstacle.center_y else dir_label.down

    ## node dict methods
    def add_node2dict(self,point,label=None):
        if label == None:
            num_keys = len(self.node_dict)
            self.init_graph_dict_entry(point,num_keys)
        else:
            self.init_graph_dict_entry(point,label)

    def init_graph_dict_entry(self,point,node_id):
        self.node_dict[node_id] = point.coord_key() # add node to dictionary
        self.vis_graph[node_id] = {} # initialize node in graph
        self.edge_type_dict[node_id] = {} # type of edges for plotting, not included in vis_graph so that I wouldn't have to mess with dijkstra or path finding interface
        self.node_obst_dict[node_id] = {} # needed for creating hugging edges when plotting solution
        self.node_obst_dict[node_id]['point'] = point # use this to store the node point objects for use with plotting, may not need this object

    def add_edge2graph(self,start_node,end_node,distance,is_hugging=False,is_edge_CW=False,ang_rot=0):
        '''creates entry in edge graph as well as records properties needed for plotting hugging edges'''
        start_id = self.get_node_id(start_node)
        end_id = self.get_node_id(end_node)
        self.vis_graph[start_id][end_id] = distance
        self.edge_type_dict[start_id][end_id] = {'is_hugging' : is_hugging}
        if is_hugging == True:
            self.edge_type_dict[start_id][end_id]['is_CW'] = is_edge_CW # this is used to determine plotting direction
            self.edge_type_dict[start_id][end_id]['ang'] = ang_rot

    def is_node_new(self,point):
        '''checks that node hasn't been added to the dictionary'''
        return not point.coord_key() in self.node_dict.inverse

    # returns node id for a node point    
    def get_node_id(self,point):
        node_id = self.node_dict.inverse[point.coord_key()]
        return node_id

    def get_node_xy(self,node_id):
        return self.node_dict[node_id]

    def get_node_obj(self,node_id):
        return self.node_obst_dict[node_id]['point']

    def get_node_obs(self,node_id):
        return self.node_obst_dict[node_id]['obstacle']

    def get_edge_ang(self,start_id,end_id):
        return self.edge_type_dict[start_id][end_id]['ang']

    def is_edge_hugging(self,start_id,end_id):
        return self.edge_type_dict[start_id][end_id]['is_hugging']

    def get_edge_type(self,is_hugging):
        # temporary method to bridge gap between allowing multiple types of edges besides hugging and surfing
        if is_hugging:
            return edge_type.circle
        elif not is_hugging:
            return edge_type.line
        else:
            raise Exception('invalid edge type')

    def is_edge_CW(self,start_id,end_id): 
        return self.edge_type_dict[start_id][end_id]['is_CW']

    def clear_obs_nodes(self): #TODO delete this method and move to vis_graph class
        for obstacle in self.obstacles:
            obstacle.clear_nodes()

    def clear_node_dict(self): #TODO delte this method, no need as graph objects are reinitialized
        self.node_dict = bidict({})
        self.vis_graph = {}
    
    def warn_close_nodes(self,point1,point2):
        diff = vec_sub(point1,point2)
        if abs(diff.x) < 0.001 or abs(diff.y) < 0.001:
            warnings.warn('Nodes are very close in value')

class visibility_graph_generator:
    #TODO look into pickle for saving vis_graph_gen objects
    debug = True # guess i could have super class to inherit this as well as any debug routines

    def __init__(self,obstacles=None,record_on = True):
        # variables for buidling vis graph
        self.record_graph_objects = record_on
        self.graphs_memory = {} # this dictionary stores the graph created start/end, graph created, and a node_point_dictionary, used for plotting

        # variables for outputting training data
        self.df_columns = ['start','end','obst1_dir']
        # self.num_col = 16 #TODO this should be equal to 4*num_obs + 4 (start_x start_y end_x end_Y (radius center_x center_y label))
        #TODO determine if using np array is the best way to ouptut the data
        #TODO num_col should be determined based on the graph and how many obstacles it has or it should be based on the maximum size we want for our neural net
        # self.vis_data = np.array([],dtype = np.double).reshape(0,self.num_col) #TODO delete if new data storage method is faster
        # self.vis_df = pd.DataFrame(columns = self.df_columns) #unsure whether to use dataframe or array

        # variables for plotting
        self.axis_xlim = [0, 30] # fixed graph axis limits, will set to be size for floor
        self.axis_ylim = [0, 20]
        self.line_width = 3
        # self.start = start # formatted (x,y)
        # self.end = end
        if obstacles is None:
            self.obstacles = []
        else:
            self.obstacles = obstacles #TODO determine if i want to store obstacle list in this object
        # visibility viewer initialization
        self.fig, self.vis_axs = plt.subplots(1,1)
        self.init_graph_props()

    def init_graph_props(self):
        self.vis_axs.set_xlim(self.axis_xlim)
        self.vis_axs.set_ylim(self.axis_ylim)
        self.vis_axs.grid(visible=True)
        self.vis_axs.set_aspect('equal')

    #vis graph methods
    def run_test(self,start_list,end_list,obstacle_list,algorithm="djikstra"):
        # main function that creates training data for start/end points
        num_obs = len(obstacle_list)
        self.init_data_memory(start_list,end_list,num_obs)
        ii = 0
        base_graph = vis_graph(obstacle_list)
        base_graph.make_obs_vis_graph()
        for start in start_list:
            for end in end_list:
                graph = copy.deepcopy(base_graph)
                # graph.clear_obs_nodes() #TODO verify removing this doesnt break the code
                graph.build_vis_graph(start,end)
                # graph.build_h_graph()
                # method that calculates shortest distance, djikstra algo
                if (algorithm == "AStar"):
                    print("Utilizing A-Star...")
                    graph.find_shortest_path_a_star()
                else: # Default Djikstra
                    print("Utilizing Djikstra...")
                    graph.find_shortest_path()
                graph.create_pw_opt_path_func()
                labels = graph.gen_obs_labels()
                obs_att = graph.get_obs_prop()
                self.record_result(start,end,obs_att,labels,ii)
                if self.record_graph_objects == True:
                    self.store_vis_graph(graph)
                if self.debug:
                    if ii % 1000 == 0: print(f'completed {ii} our of {len(start_list)*len(end_list)}')
                    ii += 1

    def store_vis_graph(self,graph): #TODO move this to child class that is generator + plotter
        # when debugging different graph configurations
        append_dict(self.graphs_memory,graph)
   
    ## output methods
    def init_data_memory(self,start_list,end_list,num_obs):
        nstart = len(start_list)
        nend = len(end_list)
        ndata = nstart*nend
        # self.vis_data = np.array([],dtype = np.double).reshape(ndata,self.num_col)
        self.num_col = 4*num_obs + 4
        self.vis_data = np.empty((ndata,self.num_col),dtype = np.double)

    def record_result(self,start,end,obstacle_att,direction_labels,idx):
        # result_df = pd.DataFrame()
        # self.vis_df = pd.concat([self.vis_df, result_df])
        label_values = [label.value for label in direction_labels] #use value as labels are enums
        # results_array = np.array([start.x, start.y, end.x, end.y,*obstacle_att, *label_values]).reshape(1,self.num_col) # direction label of 1 is up and 0 is down
        # self.vis_data = np.concatenate([results_array,self.vis_data])
        results_array = [start.x, start.y, end.x, end.y,*obstacle_att, *label_values]# direction label of 1 is up and 0 is down
        self.vis_data[idx,:] = results_array

    def output_csv(self,file_name):
        # output training data to file
        response_needed = True
        fname = file_name+'.csv'
        # attempt to prevent accidental file overwrites
        if os.path.exists(fname):
            while response_needed:
                ans = input('file name already exists, overwrite? (y/n)')
                if ans == 'y':
                    np.savetxt(fname, self.vis_data, delimiter=",")
                    response_needed = False
                elif ans == 'n':
                    new_name = input('new file name?')
                    new_fname = new_name+'.csv'
                    if os.path.exists(new_fname):
                        print('that name exists already too...')
                    else:
                        np.savetxt(new_name+'.csv',self.vis_data, delimiter=",")
                        response_needed = False
                else:
                    print('invalid response try again')

        else:
            np.savetxt(fname, self.vis_data, delimiter=",")

    ## plot viewer methods    
    def plot_env(self,test_num,title=None):
        self.plot_start_end(test_num)
        self.plot_obstacles(test_num)
        self.finish_plot(title)

    def plot_solution(self,test_num,title=None):
        # plots obstacles and solution
        self.plot_start_end(test_num)
        self.plot_obstacles(test_num)
        self.plot_shortest_path(test_num) 
        self.finish_plot(title)

    def plot_full_vis_graph(self,test_num,title=None):
        self.plot_start_end(test_num)
        self.plot_obstacles(test_num)
        self.plot_vis_graph(test_num) #TODO add test_num to be plotted
        self.finish_plot(title)

    def plot_just_obstacles(self,test_num,title=None):
        #TODO update axis limits so that all obstacles can be seen
        self.plot_obstacles(test_num)
        self.finish_plot(title)

    ## "private" plot methods
    def get_start_end_data(self,test_num):
        data = self.vis_data[test_num,:]
        start = (data[0],data[1])
        end = (data[2],data[3])
        return start, end

    def update_axis_lim(self,test_num):
        # TODO this only works for updating start_end axis limits
        start,end = self.get_start_end_data(test_num)
        if self.axis_xlim[0] > start[0]-2:
            self.axis_xlim[0] = start[0]-2
        if self.axis_xlim[1] < end[0]+2:
            self.axis_xlim[1] = end[0]+2
        #TODO update y axis lim by finding which obstacle, or start end point, is highest and lowest y and compare to axis limits
        if self.axis_ylim[0] > start[1]-2:
            self.axis_ylim[0] = start[1]-2
        if self.axis_ylim[1] < end[1]+2:
            self.axis_ylim[1] = end[1]+2
        self.init_graph_props()

    def plot_start_end(self,test_num):
        self.update_axis_lim(test_num)
        start,end = self.get_start_end_data(test_num)
        self.vis_axs.scatter(start[0],start[1],color='red',marker="^",linewidth=self.line_width,label="start")
        self.vis_axs.scatter(end[0],end[1],color='green',marker="o",linewidth=self.line_width,label="end")

    def plot_obstacles(self,test_num):
        # plots obstacles
        graph = self.graphs_memory[test_num]
        for obstacle in graph.obstacles:
            obst_x, obst_y = self.make_circle_points(obstacle)
            # self.vis_axs.plot(obst_x, obst_y,color='blue',linewidth=self.line_width,label="obstacle")
            self.vis_axs.plot(obst_x, obst_y,color='blue',linewidth=self.line_width)

    def make_circle_points(self,obstacle):
        #TODO could remove this function and use make_arc_points instead
        thetas = np.linspace(0,2*np.pi,100)
        radius = obstacle.radius
        #TODO replace this with direction_step()
        bound_x = radius*np.cos(thetas) + obstacle.center_x
        bound_y = radius*np.sin(thetas) + obstacle.center_y
        return bound_x, bound_y

    def make_arc_points(self,start_id,end_id,graph):
        start_point = graph.get_node_obj(start_id) #TODO this could also be point(graph.get_node_xy(start_id) so we dont have to store the points in the node data dict)
        # end_point = graph.get_node_obj(end_id)
        obstacle = graph.get_node_obs(start_id) # TODO decide if I need to add a check that both the start_id and end_id are tagged to the same obstacle (would need to add an obstacle bi,dictionary)
        ang_start = find_point_angle(start_point,obstacle)
        ang_diff = graph.get_edge_ang(start_id,end_id)
        if graph.is_edge_CW(start_id,end_id) == True:
            ang_diff = -ang_diff
        if self.debug == True:
            print('start id ' + str(start_id) + ' end id ' + str(end_id))
            # print('ang start ' + str(ang_start*180/np.pi) + ' ang end ' + str(ang_end*180/np.pi))
            print('ang_diff ' + str(ang_diff*180/np.pi) + ' is_CW ' + str(graph.is_edge_CW(start_id,end_id)))
        thetas = np.linspace(ang_start,ang_start+ang_diff,100)
        bound_x = obstacle.radius*np.cos(thetas) + obstacle.center_x
        bound_y = obstacle.radius*np.sin(thetas) + obstacle.center_y
        return list(zip(bound_x, bound_y))

    def plot_vis_graph(self,test_num):
        # this method plots the visibility graph generated
        node_points = []
        graph = self.graphs_memory[test_num]
        for node_id in graph.vis_graph:
            node_points.append(graph.get_node_xy(node_id))
            for adj_node_id in graph.vis_graph[node_id]:
                #TODO replace bleow with method plot_graph_edge()
                if graph.is_edge_hugging(node_id,adj_node_id) == False: # 
                    node_points.append(graph.get_node_xy(adj_node_id))
                    self.vis_axs.plot(*zip(*node_points),color='purple',linewidth=self.line_width)
                    node_points.pop()
                else:
                    arc_points = self.make_arc_points(node_id,adj_node_id,graph)
                    for point in arc_points: node_points.append(point)
                    self.vis_axs.plot(*zip(*node_points),color='purple',linewidth=self.line_width) # plot formatted points
                    # node_points.remove(arc_points) #TODO verify this code removes all points except for the root node id
                    del node_points[1:]
            if len(node_points) > 1:
                raise Exception("sorry, node_points was not reset properly during plotting")
            node_points.pop() #reset node_points

    def finish_plot(self,title=None):
        self.vis_axs.set_title(title)
        self.vis_axs.legend()

    def plot_shortest_path(self,test_num):
        graph = self.graphs_memory[test_num]
        for idx,node_id in enumerate(graph.opt_path):
            if node_id == 'end': # reached end of path
                break
            else:
                next_node_id = graph.opt_path[idx+1]
                self.plot_graph_edge(graph,node_id,next_node_id)

    def plot_graph_edge(self,graph,start_id,end_id):
        node_points = []
        node_points.append(graph.get_node_xy(start_id))
        if graph.is_edge_hugging(start_id,end_id) == False: # 
            node_points.append(graph.get_node_xy(end_id))
            self.vis_axs.plot(*zip(*node_points),color='purple',linewidth=self.line_width)
        else:
            arc_points = self.make_arc_points(start_id,end_id,graph)
            for point in arc_points: node_points.append(point)
            self.vis_axs.plot(*zip(*node_points),color='purple',linewidth=self.line_width) # plot formatted points
            # node_points.remove(arc_points) #TODO verify this code removes all points except for the root node id

    def plot_all_node_labels(self,test_num):
        graph = self.graphs_memory[test_num]
        for node_id in graph.node_dict:
            node = graph.node_dict[node_id]
            self.vis_axs.text(node[0],node[1],str(node_id))
    
    def plot_opt_path_node_labels(self,test_num):
        graph = self.graphs_memory[test_num]
        for node_id in graph.opt_path:
            node = graph.get_node_obj(node_id)
            self.vis_axs.text(node.x,node.y,str(node_id))

    def plot_labels(self,test_num):
        graph = self.graphs_memory[test_num]
        for label,obstacle in zip(graph.obstacle_labels,graph.obstacles):
            if label == dir_label.up:
                self.vis_axs.scatter(obstacle.center_x,obstacle.center_y,color='purple',marker=6)
            elif label == dir_label.down:
                self.vis_axs.scatter(obstacle.center_x,obstacle.center_y,color='purple',marker=7)
            else:
                raise Exception('invalid label type')


    def clear_plot(self):
        self.vis_axs.cla()
        # self.vis_axs.grid(visible=True)
        self.init_graph_props()

    def save_plot_image(self,fig_name):
        self.fig.savefig(fig_name + '.png')
        # creates .png of visibility graph

class graph_viewer(visibility_graph_generator):
    # This class allows us to look at the vis graph before it is finished
    def __init__(self, vis_graph_obj, obstacles=None, record_on=True):
        super().__init__(obstacles, record_on) # this is needed so we can reuse plot methods from parent
        self.store_vis_graph(vis_graph_obj) 

    #TODO create method that gets obstacle data here and in parent class

    def get_start_end_data(self, test_num=0):
        graph = self.graphs_memory[test_num]
        start = (graph.start.x, graph.start.y)
        end = (graph.end.x, graph.end.y)
        return start,end

    def plot_network(self,test_num=0):
        self.plot_start_end(test_num)
        self.plot_obstacles(test_num)
        self.plot_vis_graph(test_num)
    
    def plot_cand_edge(self,node1,node2):
        x_points = [node1.x,node2.x]
        y_points = [node1.y,node2.y]
        self.vis_axs.plot(x_points,y_points,color='red')
        
# global methods
# for reading new file list
def read_obstacle_list(fname):
    def read_obstacle(obs_string):
        data = obs_string.split(",")
        r = float(data[0])
        center = point((float(data[1]),float(data[2])))
        return obstacle(r,center)
    obstacle_courses = {}
    obstacles = []
    obs_file = open(fname,"r")
    action = 0
    while obs_file:
        line = obs_file.readline()
        
        if (line.strip() == "New Obstacle Set:" and len(obstacles) > 0) or line == '':
            append_dict(obstacle_courses,obstacles)
            action = 0
            obstacles = []
        elif line.strip() == "radius,x,y":
            action = 1
            line = obs_file.readline()

        if line == "":
            break
        
        if action == 1:
            new_obs = read_obstacle(line.strip())
            obstacles.append(new_obs)
        
    obs_file = obs_file.close()
    return obstacle_courses

def append_dict(dict_in,item):
    num_keys = len(dict_in)
    dict_in[num_keys] = item

def remove_list_item(item,list_in):
    #TODO this function might not work
    # removes obstacle from obstacle list without modifying original obstacle_list
    new_obs_list = list_in[:]
    new_obs_list.remove(item)
    return new_obs_list

def check_collision(a,b,c,x,y,radius):
    dist = round((abs(a * x + b * y + c)) / np.sqrt(a * a + b * b),4) # rounding for numerical errors
    if radius > dist:
        collision = True
    else:
        collision = False
    return collision

def slope_int_form(start_node,end_node):
    slope = (end_node.y-start_node.y)/(end_node.x-start_node.x)
    y_int = end_node.y-slope*end_node.x
    return slope, y_int

def planar_line_form(start_node,end_node):
    # returns line connecting nodes in planar line form for check_collision
    slope,y_int = slope_int_form(start_node,end_node) # TODO replace with decorator?
    # slope = (end_node.y-start_node.y)/(end_node.x-start_node.x)
    # y_int = end_node.y-slope*end_node.x
    a = -slope
    b = 1
    c = -y_int
    return a,b,c

def find_point_angle(node,obstacle):
    # finds the angle of a point w.r.t the x axis of the obstacle, postiive angle returned
    node_bar = vec_sub(node,obstacle.center_loc)
    ang_val = np.arctan2(node_bar.y,node_bar.x)
    if ang_val < 0:
        ang_val = ang_val + 2*np.pi
    return ang_val

def find_ang_diff(ang1,ang2):
    '''calculates absolute angle difference between two angles'''
    ang_diff = np.abs(ang1-ang2)
    return ang_diff
    
def vec_acc(p1,p2):
    p3x = p1.x + p2.x
    p3y = p1.y + p2.y
    p3 = point((p3x,p3y))
    return p3

def vec_sub(p1,p2):
    '''returns p1-p2'''
    p3x = p1.x - p2.x
    p3y = p1.y - p2.y
    p3 = point((p3x,p3y))
    return p3

def vec_dot(p1,p2):
    p3 = p1.x*p2.x + p1.y*p2.y
    return p3

def recursive_items(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict:
            yield (key, value)
            yield from recursive_items(value)
        else:
            yield (key, value)

# initialization functions
def init_points(point_list):
    point_obj = []
    for p in point_list:
        point_val = point(p)
        point_obj.append(point_val)
    return point_obj

def init_obs(obs_list,radius_list):
    #TODO This only allows for one obstacle radius size
    if len(obs_list) != len(radius_list):
        raise Exception('obstacle list and radius list different lengths')
    obs_loc = init_points(obs_list)
    obs_data = zip(obs_loc,radius_list)
    obs_obj = []
    for obs,radius in obs_data:
        obs_val = obstacle(radius,obs)
        obs_obj.append(obs_val)
    return obs_obj

def find_test_range(obstacle_list):
    prop_list = [obs.output_prop() for obs in obstacle_list]
    obs_min_x = [prop[0]-prop[2] for prop in prop_list]
    obs_max_x = [prop[0]+prop[2] for prop in prop_list]
    return min(obs_min_x),max(obs_max_x)

def get_list_points(x,y):
    grid_x,grid_y = np.meshgrid(x,y)
    grid = np.vstack([grid_x.ravel(),grid_y.ravel()])
    grid = grid.tolist()
    grid = list(zip(*grid))
    return grid