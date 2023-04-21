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
import pickle
import scipy.io as sio
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

class point:
    key_precision = 10

    def __init__(self,point_coord):
        self.x = point_coord[0]
        self.y = point_coord[1]

    def coord_key(self):
        # outputs the points as a tuple to be used as a key in node dictionary
        return round(self.x,self.key_precision), round(self.y,self.key_precision)

    def output(self):
        # this is used by check_collision()
        return(self.x,self.y)

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
        #TODO would this exception help find an error in the algorithm?
        # if point in self.node_list:
        #     raise Exception("point already in obstacle definition")
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
    debug = False

    def __init__(self,obstacles):
        self.node_dict = bidict({}) # stores the nodes as associated with each point, bidict allows lookup in both directions
        self.vis_graph = {} #TODO this shouldn't have the class name # this is a graph that stores the nodes and their edge distances
        self.h_graph = {} # Distance from Key Node to Final Node, normally. May also be used for any other heuristic cost
        self.edge_type_dict = {} # matches the format of vis_graph but only records if edge is surfing or hugging for plotting purposes
        self.node_obst_dict = {} # keeps track of which obstacle each node is on, adding this to make plotting arcs
        self.pw_opt_func = {} # record parameters of piecewise function for label evaulation
        self.opt_path_cost = 0
        self.obstacles = obstacles
        self.debug_view_enabled = False

    def attach_graph_viewer(self):
        if not self.debug_view_enabled:
            self.viewer = graph_viewer(self)    
            self.debug_view_enabled = True

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
        self.process_cand_node(self.start,self.end,obstacle=None)
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
        # def check_start_end(start_node,cand_node):
        #     is_start_end = False
        #     if start_node == 'start' or cand_node == 'start':
        #         is_start_end = True
        #     if start_node == 'end' or cand_node == 'end':
        #         is_start_end = True
        #     return is_start_end

        if self.is_node_vis(start_node,cand_node):
            self.update_node_props(cand_node,obstacle)
            # if start_node is an end_node, then add to graph vertically
            edge_length = self.euclid_dist(start_node,cand_node)
            # if check_start_end(start_node,cand_node):
            #     if is_end_node is False: 
            #         self.add_edge2graph(start_node,cand_node,edge_length)
            #     else:
            #         self.add_edge2graph(cand_node,start_node,edge_length)
            # else:
            self.add_edge2graph(start_node,cand_node,edge_length)
            self.add_edge2graph(cand_node,start_node,edge_length)
        
        cat = 0 # so i have a line for breakpoint


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
        phi = self.rotation_from_horiz(obstacle.center_loc,start_node)
        
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
                # if obstacles have same x coordinate, compare diff of their y-coords
                if diff.y > 0:
                    obst_A = obst 
                    obst_B = next_obst
                elif diff.y < 0:
                    obst_A = next_obst
                    obst_B = obst
                else:
                    #TODO I don't think this will catch something that the following exception wouldnt have already caught
                    msg = "obstacles have same x and y coordinate: vis_obst_obst()"
                    self.record_exception_data(obst=obst,remaining_obst=remaining_obst,function='vis_obst_obst',exception=msg)
                    raise Exception(msg)
            
            if id(obst) == id(next_obst): # dont calculate visibility with itself
                msg = 'Obstacle not deleted correctly from list'
                self.record_exception_data(obst=obst,remaining_obst=remaining_obst,function='vis_obst_obst',exception=msg)
                raise Exception(msg)
            
            self.internal_bitangents(obst_A,obst_B)
            self.external_bitangents(obst_A,obst_B)
            # loops through the new nodes with process_cand_node
        

    def internal_bitangents(self,obst_L,obst_R):
        '''Finds internal bitangents, obstacle L should be to the left of obstacle R on x axis'''
        center_dist = self.euclid_dist(obst_L.center_loc,obst_R.center_loc)
        theta = np.arccos((obst_L.radius+obst_R.radius)/center_dist)
        phi_L = self.rotation_from_horiz(obst_L.center_loc,obst_R.center_loc)
        phi_R = self.rotation_from_horiz(obst_R.center_loc,obst_L.center_loc) # abs(phi_R) is supplementary angle of abs(phi_L)
              
        cand_nodeL1 = point(self.direction_step(obst_L.center_loc,obst_L.radius,phi_L + theta))
        cand_nodeL2 = point(self.direction_step(obst_L.center_loc,obst_L.radius,phi_L - theta))
        cand_nodeR1 = point(self.direction_step(obst_R.center_loc,obst_R.radius,phi_R - theta))
        cand_nodeR2 = point(self.direction_step(obst_R.center_loc,obst_R.radius,phi_R + theta))


        self.process_cand_edge((cand_nodeL1,obst_L),(cand_nodeR2,obst_R))    
        self.process_cand_edge((cand_nodeL2,obst_L),(cand_nodeR1,obst_R))
        
    def external_bitangents(self,obst_L,obst_R):
        '''Finds external bitangents, obstacle being left or right doesnt affect this calculation'''
        center_dist = self.euclid_dist(obst_L.center_loc,obst_R.center_loc)
        theta = np.arccos(abs(obst_L.radius-obst_R.radius)/center_dist)

        # need to compare obstacle radii to determine angle directions
        if obst_L.radius > obst_R.radius:
            phi = self.rotation_from_horiz(obst_L.center_loc,obst_R.center_loc)
        else: # if radii are the same size, it doesnt matter the order you calculate rotation_from_horiz
            phi = self.rotation_from_horiz(obst_R.center_loc,obst_L.center_loc)

        cand_nodeL1 = point(self.direction_step(obst_L.center_loc,obst_L.radius,phi + theta))
        cand_nodeL2 = point(self.direction_step(obst_L.center_loc,obst_L.radius,phi - theta))
        cand_nodeR1 = point(self.direction_step(obst_R.center_loc,obst_R.radius,phi + theta))
        cand_nodeR2 = point(self.direction_step(obst_R.center_loc,obst_R.radius,phi - theta))
        
        self.process_cand_edge((cand_nodeL1,obst_L),(cand_nodeR1,obst_R))    
        self.process_cand_edge((cand_nodeL2,obst_L),(cand_nodeR2,obst_R))


    def is_node_vis(self,start_node,end_node):        
        # checks if visibility line intersects other obstacles
        is_valid = True
        
        check = False #option to enable when debugging, should be off normally
        if check:
            self.attach_graph_viewer()
            self.viewer.reinit_vis_graph(self)
            self.viewer.plot_obstacles(0)
            self.viewer.plot_cand_edge(start_node,end_node)
            self.viewer.update_axis_lim(0)
            
        for obstacle in self.obstacles:
            if check_collision(start_node,end_node,obstacle):
                if self.debug:
                    print(f'non visible edge found:')
                    print(f'start=({start_node.x},{start_node.y})')
                    print(f'end_node=({end_node.x},{end_node.y})')
                    print(f'obstacle(r,x,y) = ({obstacle.view()}\n')
                is_valid = False

        if check:
            self.viewer.set_title(f'is_valid = {is_valid}')

        return is_valid

    def is_obst_between_points(self,start_node,end_node,obstacle):
        '''returns if obstace is between horizontally and vertically.'''
        if obstacle.center_x > start_node.x and obstacle.center_x < end_node.x:
            is_between = True
        else:
            is_between = False #TODO add an output that says if obstacle is between points when start_node and end_node have same x coord.  Maybe we want to check y coordinate also
        if self.debug:
            print(f'vis_graph.is_obst_between_points()')
            print(f'obstacle at ({obstacle.center_x},{obstacle.center_y})')
            print(f'start_node at ({start_node.x},{start_node.y})')
            print(f'end_node at ({end_node.x},{end_node.y}')
            print(f'is_between = {is_between}\n')
        return is_between

    def direction_step(self,start,dist,angle):
        '''calculates tangent node location on an obstacle'''
        x = dist*np.cos(angle) + start.x
        y = dist*np.sin(angle) + start.y
        return (x,y)

    def rotation_from_horiz(self,point1,point2):
        '''calculates rotation from positive horizontal axis at point1.y to line defined by point1, point2.  Direction of rotation is smaller of absolute value of two rotations from pos horizontal axis to line, i.e. the one <= to pi rotation'''
        dy = point2.y - point1.y
        dx = point2.x - point1.x
        rotation_1_2 = np.arctan2(dy,dx)
        return rotation_1_2

    def euclid_dist(self,point1,point2):
        '''calculates euclidean distance b/T two points'''
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
        self.algo = 'Dijkstra'
        start_id = self.get_node_id(self.start)
        end_id = self.get_node_id(self.end)
        self.opt_path = dijk.shortestPath(self.vis_graph,start_id,end_id)
        if self.debug == True:
            print(self.opt_path)

    def find_shortest_path_a_star(self):
        self.algo = 'AStar'
        start_id = self.get_node_id(self.start)
        end_id = self.get_node_id(self.end)
        self.opt_path = ast.shortestPath(self.vis_graph, self.h_graph, start_id, end_id)
        if self.debug == True:
            print(self.opt_path)

    def eval_opt_path_cost(self):
        # cost = 0
        # for ii,node_id in enumerate(self.opt_path):
        #     if node_id == 'start':
        #         continue
        #     cost += self.vis_graph[self.opt_path[ii-1]][self.opt_path[ii]]
        # self.opt_path_cost = cost
        # return cost
        self.opt_path_cost = self.eval_path_cost(self.opt_path)
        return self.opt_path_cost
    
    def eval_path_cost(self,path_list):
        cost = 0
        for ii,node_id in enumerate(path_list):
            if node_id == 'start':
                continue
            cost += self.vis_graph[path_list[ii-1]][path_list[ii]]
        return cost

    def create_pw_opt_path_func(self):
        '''Creates a piecewise function of the optimal path to use for creating object labels'''
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

        is_slope_inf = False # this flag is used to handle specific edge case where first line segment is infinite slope
        for idx,node_id in enumerate(self.opt_path):
            if node_id == 'start': # equation is up to a node, and there are no points before the start
                continue
            x_key,_ = self.get_node_xy(node_id)
            node_before = self.opt_path[idx-1] #node before node_id
            edge_key = self.get_edge_type(self.is_edge_hugging(node_before,node_id)) # finding edge type between this and previous node
            
            x_before,_ = self.get_node_xy(node_before)
            if x_before >= x_key:
                msg = "invalid opt path found"
                self.record_exception_data(x_before=x_before,x_key=x_key,function='create_pw_opt_path_func',exception=msg)
                raise Exception(msg)

            if edge_key == edge_type.line:
                slope,y_int = slope_int_form(self.get_node_obj(node_before),self.get_node_obj(node_id))
                if slope > 0:
                    is_pos = True
                elif slope < 0:
                    is_pos = False
                elif slope == 0:
                    # for the case we have zero_slope_sign on the last segment to the end
                    if node_id == 'end':
                        # is_pos = self.pw_opt_func[self.get_x_key_smaller(x_key)].is_slope_pos()
                        is_pos = True # this value is not used again after the "end" node so it doesnt matter what it is
                    else:
                        is_pos = decide_zero_slope_sign(node_id)
                else:
                    msg = f'invalid slope value: {slope}'
                    self.record_exception_data(slope = slope,function='create_pw_opt_path_func',exception=msg)
                    raise Exception(msg)
                if slope == float("-inf") or slope == float("inf"):
                    is_slope_inf = True
                    inf_slope_sign = is_pos
                    continue # we can ignore these in pw_func becasue previous steps guarantee the obstacle is not intersecting with the infinite slope segment.  Thus we can just ignore it as the value is handled by the other parts of the pw_func.
                self.pw_opt_func[x_key] = pf.line(slope,y_int,is_pos)
            elif edge_key == edge_type.circle:
                if is_slope_inf == True and not self.pw_opt_func: # only covers the case where first line in slope is inf
                    is_slope_pos = inf_slope_sign
                else:
                    is_slope_pos = self.pw_opt_func[self.get_x_key_smaller(x_key)].is_slope_pos() # determine if previous segment has pos or neg slope
                obstacle = self.get_node_obs(node_id)
                self.pw_opt_func[x_key] = pf.circle(*obstacle.output_prop(),is_slope_pos)
            else:
                msg = f'invalid Edge key found: {edge_key}'
                self.record_exception_data(edge_key = edge_key,function='create_pw_opt_path_func',exception=msg)
                raise Exception(msg)

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
        # adding this code to detect when x_key gets added twice to pw_opt_func dictionary
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
        '''returns value of is_hugging from edge_type_dict'''
        return self.edge_type_dict[start_id][end_id]['is_hugging']

    def get_edge_type(self,is_hugging):
        '''temporary method to bridge gap between allowing multiple types of edges besides hugging and surfing'''
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

    def record_exception_data(self,**kwargs):
        filename = "vis_graph_data.pickle"
        folder_path = "./exception_data"
        create_directory(folder_path)
        if os.path.exists(os.path.join(folder_path,filename)):
            # If the file already exists, add a suffix to the filename
            ii = 1
            while True:
                new_filename = f"{os.path.splitext(filename)[0]}_{ii}{os.path.splitext(filename)[1]}"
                if os.path.exists(os.path.join(folder_path, new_filename)):
                    ii += 1
                else:
                    filename = new_filename
                    break
        with open(os.path.join(folder_path,filename),'wb') as f: #TODO change this to also output data regarding which test the error is
            kwargs['vis_graph_object'] = self
            pickle.dump(kwargs,f)
        f.close()

class visibility_graph_generator:
    #TODO look into pickle for saving vis_graph_gen objects
    debug = False # guess i could have super class to inherit this as well as any debug routines

    def __init__(self,obstacles=None,record_on=True,is_ion=False,obs_file=None):
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

        self.obs_file = obs_file
        self.is_plot_self = True # used by subplots function for writing methods on new plot

        if is_ion:
            plt.ion() 
        if record_on:
            self.fig = plt.figure()
        if is_ion:
            self.place_figure() #sets location of plot window to second monitor on the left
        if record_on:
            self._init_graph_props()    
        

    #vis graph methods
    def run_test(self,start_list,end_list,obstacle_list,algorithm="dijkstra",pad_list=True):
        # main function that creates training data for start/end points
        if pad_list:
            num_obs = 20
            if len(obstacle_list)>num_obs:
                raise Exception('number of obstacles greater than size of padded list')
        else:
            num_obs = len(obstacle_list)
        self.init_data_memory(start_list,end_list,num_obs)
        ii = 0
        base_graph = vis_graph(obstacle_list)
        base_graph.make_obs_vis_graph()
        for start in start_list:
            for end in end_list:
                graph = copy.deepcopy(base_graph)
                graph.build_vis_graph(start,end)
                graph.build_h_graph()
                # method that calculates shortest distance, dijkstra algo
                if (algorithm == "AStar"):
                    if self.debug:
                        print("Utilizing A-Star...")
                    graph.find_shortest_path_a_star()
                else: # Default Dijkstra
                    if self.debug:
                        print("Utilizing Dijkstra...")
                    graph.find_shortest_path()
                opt_cost = graph.eval_opt_path_cost()
                graph.create_pw_opt_path_func()
                labels = graph.gen_obs_labels()
                obs_att = graph.get_obs_prop()
                self.record_result(start,end,obs_att,labels,ii,pad_list,num_obs,opt_cost)
                if self.record_graph_objects == True:
                    self.store_vis_graph(graph)
                if self.debug:
                    if ii % 1000 == 0: print(f'completed {ii} out of {len(start_list)*len(end_list)}')
                ii += 1
        if self.debug:
            print(f'completed {ii} out of {len(start_list)*len(end_list)}')  

    def run_ginput_test(self,obstacle_list,algorithm="dijkstra"):
        pass

    def store_vis_graph(self,graph): #TODO move this to child class that is generator + plotter
        # when debugging different graph configurations
        append_dict(self.graphs_memory,graph)
   
    ## output methods
    def init_data_memory(self,start_list,end_list,num_obs):
        '''initializes the data memory for a course test'''
        nstart = len(start_list)
        nend = len(end_list)
        ndata = nstart*nend
        # self.vis_data = np.array([],dtype = np.double).reshape(ndata,self.num_col)
        self.num_col = calc_ndata_col(num_obs)
        self.vis_data = np.empty((ndata,self.num_col),dtype = np.double)
        self.is_memory_init = True

    def record_result(self,start,end,obstacle_att,direction_labels,idx,pad_list,num_obs,opt_path_cost):
        # result_df = pd.DataFrame()
        # self.vis_df = pd.concat([self.vis_df, result_df])
        label_values = [label.value for label in direction_labels] #use value as labels are enums
        # results_array = np.array([start.x, start.y, end.x, end.y,*obstacle_att, *label_values]).reshape(1,self.num_col) # direction label of 1 is up and 0 is down
        # self.vis_data = np.concatenate([results_array,self.vis_data])
        if pad_list:
            pad_length = num_obs-len(label_values)
            obs_pad = [0] * pad_length * 3 # 3 obstacle properties
            label_pad = [1] * pad_length
            results_array = [start.x, start.y, end.x, end.y,*obstacle_att, *obs_pad, *label_values, *label_pad, opt_path_cost]# direction label of 1 is up and 0 is down
        else:
            results_array = [start.x, start.y, end.x, end.y,*obstacle_att, *label_values, opt_path_cost]# direction label of 1 is up and 0 is down
        self.vis_data[idx,:] = results_array

    def output_csv(self,file_name,overwrite=False):
        # output training data to file
        response_needed = True
        fname = file_name+'.csv'
        # attempt to prevent accidental file overwrites
        if os.path.exists(fname):
            while response_needed:
                if overwrite:
                    ans = 'y'
                else:
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
    def plot_env(self,test_num,title=None,loc='best'):
        self.plot_start_end(test_num)
        self.plot_obstacles(test_num)
        self.finish_plot(title=title,loc=loc)

    def plot_solution(self,test_num,title=None,loc='best'):
        # plots obstacles and solution
        self.plot_start_end(test_num)
        self.plot_obstacles(test_num)
        self.plot_shortest_path(test_num) 
        self.finish_plot(title=title,loc=loc)

    def plot_full_vis_graph(self,test_num,title=None,loc='best'):
        self.plot_start_end(test_num)
        self.plot_obstacles(test_num)
        self.plot_vis_graph(test_num) #TODO add test_num to be plotted
        self.finish_plot(title=title,loc=loc)

    def plot_just_obstacles(self,test_num,title=None,loc='best'):
        #TODO update axis limits so that all obstacles can be seen
        self.update_axis_lim(test_num)
        self.plot_obstacles(test_num)
        self.finish_plot(title=title,loc=loc)

    def plot_network(self,test_num):
        #TODO decide if i want it to be called plot_network or plot_full_vis_graph
        # self.plot_start_end(test_num)
        self.update_axis_lim(test_num)
        self.plot_obstacles(test_num)
        self.plot_vis_graph(test_num)

    ## "private" plot methods
    def get_start_end_data(self,test_num):
        #TODO this is part of method so i can still plot things even if I dont save the graphs but i probably wont end up using it..
        try:
            data = self.vis_data[test_num,:]
            start = (data[0],data[1])
            end = (data[2],data[3])
        except:
            graph = self.graphs_memory[test_num]
            start = (graph.start.x, graph.start.y)
            end = (graph.end.x, graph.end.y)

        return start, end

    def update_axis_lim(self,test_num):
        
        x_points = []
        y_points = []
        # get start, end, and obstacles and add to compare list
        graph = self.graphs_memory[test_num]
        
        try:
            start,end = self.get_start_end_data(test_num)
        # add all critical x and y points
            x_points.append(start[0])
            x_points.append(end[0])
            y_points.append(start[1])
            y_points.append(end[1])
        except:
            print('no start/end point found, continuing axis lim update')
        for obstacle in graph.obstacles:
            x_low = obstacle.center_x - obstacle.radius
            x_high = obstacle.center_x + obstacle.radius
            x_points.append(x_low)
            x_points.append(x_high)
            y_low = obstacle.center_y - obstacle.radius
            y_high = obstacle.center_y + obstacle.radius
            y_points.append(y_low)
            y_points.append(y_high)

        x_min = min(x_points)
        x_max = max(x_points)
        y_min = min(y_points)
        y_max = max(y_points)

        # compare max/min's to axis limits and adjust if they are violated
        if self.axis_xlim[0] > x_min-2:
            self.axis_xlim[0] = x_min-2
        if self.axis_xlim[1] < x_max+2:
            self.axis_xlim[1] = x_max+2
        if self.axis_ylim[0] > y_min-2:
            self.axis_ylim[0] = y_min-2
        if self.axis_ylim[1] < y_max+2:
            self.axis_ylim[1] = y_max+2
        self._init_graph_props()

    def plot_start_end(self,test_num):
        self._act_fig()
        self.update_axis_lim(test_num)
        start,end = self.get_start_end_data(test_num)
        plt.scatter(start[0],start[1],color='green',marker="o",linewidth=self.line_width,label="start")
        plt.scatter(end[0],end[1],color='red',marker="o",linewidth=self.line_width,label="end")

    def plot_obstacles(self,test_num):
        # plots obstacles
        self._act_fig()
        graph = self.graphs_memory[test_num]
        for obstacle in graph.obstacles:
            obst_x, obst_y = self.make_circle_points(obstacle)
            # plt.plot(obst_x, obst_y,color='blue',linewidth=self.line_width,label="obstacle")
            plt.plot(obst_x, obst_y,color='black',linewidth=2)

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
            print('start id         ' + str(start_id) + ' end id ' + str(end_id))
            # print('ang start ' + str(ang_start*180/np.pi) + ' ang end ' + str(ang_end*180/np.pi))
            print('ang_diff         ' + str(ang_diff*180/np.pi) + ' is_CW ' + str(graph.is_edge_CW(start_id,end_id)))
            print('distance to node ' + str(graph.vis_graph[start_id][end_id]))
            print('distance to end  ' + str(graph.h_graph[end_id]))
        thetas = np.linspace(ang_start,ang_start+ang_diff,100)
        bound_x = obstacle.radius*np.cos(thetas) + obstacle.center_x
        bound_y = obstacle.radius*np.sin(thetas) + obstacle.center_y
        return list(zip(bound_x, bound_y))

    def plot_vis_graph(self,test_num):
        # this method plots the visibility graph generated
        self._act_fig()
        node_points = []
        graph = self.graphs_memory[test_num]
        for node_id in graph.vis_graph:
            node_points.append(graph.get_node_xy(node_id))
            for adj_node_id in graph.vis_graph[node_id]:
                #TODO replace bleow with method plot_graph_edge()
                if graph.is_edge_hugging(node_id,adj_node_id) == False: # 
                    node_points.append(graph.get_node_xy(adj_node_id))
                    plt.plot(*zip(*node_points),color='purple',linewidth=self.line_width)
                    node_points.pop()
                else:
                    arc_points = self.make_arc_points(node_id,adj_node_id,graph)
                    for point in arc_points: node_points.append(point)
                    plt.plot(*zip(*node_points),color='purple',linewidth=self.line_width) # plot formatted points
                    # node_points.remove(arc_points) #TODO verify this code removes all points except for the root node id
                    del node_points[1:]
            if len(node_points) > 1:
                raise Exception("node_points was not reset properly during plotting")
            node_points.pop() #reset node_points

    def set_title(self,title):
        self._act_fig()
        plt.title(title)

    def finish_plot(self,title=None,loc='best'):
        self._act_fig()
        plt.legend(fontsize='small',loc=loc)
        plt.title(title, wrap=True)

    def plot_shortest_path(self,test_num):
        graph = self.graphs_memory[test_num]
        for idx,node_id in enumerate(graph.opt_path):
            if node_id == 'end': # reached end of path
                break
            else:
                next_node_id = graph.opt_path[idx+1]
                self.plot_graph_edge(graph,node_id,next_node_id)

    def plot_graph_edge(self,graph,start_id,end_id):
        self._act_fig()
        node_points = []
        node_points.append(graph.get_node_xy(start_id))
        if graph.is_edge_hugging(start_id,end_id) == False: # 
            node_points.append(graph.get_node_xy(end_id))
            plt.plot(*zip(*node_points),color='purple',linewidth=self.line_width)
        else:
            arc_points = self.make_arc_points(start_id,end_id,graph)
            for point in arc_points: node_points.append(point)
            plt.plot(*zip(*node_points),color='purple',linewidth=self.line_width) # plot formatted points
            # node_points.remove(arc_points) #TODO verify this code removes all points except for the root node id

    def plot_all_node_labels(self,test_num):
        self._act_fig()
        graph = self.graphs_memory[test_num]
        for node_id in graph.node_dict:
            node = graph.node_dict[node_id]
            plt.text(node[0],node[1],str(node_id))
    
    def plot_obstacle_node_labels(self,test_num,obstacle):
        self._act_fig()
        graph = self.graphs_memory[test_num]
        for node in obstacle.node_list:
            node_id = graph.get_node_id(node)
            plt.text(node.x,node.y,str(node_id))

    def plot_opt_path_node_labels(self,test_num):
        self._act_fig()
        graph = self.graphs_memory[test_num]
        for node_id in graph.opt_path:
            node = graph.get_node_obj(node_id)
            plt.text(node.x,node.y,str(node_id))

    def plot_labels(self,test_num):
        '''plots up down labels for obstacles'''
        self._act_fig()
        graph = self.graphs_memory[test_num]
        is_first_up = True
        is_first_down = True
        label_width = 3
        for label,obstacle in zip(graph.obstacle_labels,graph.obstacles):
            
            if label == dir_label.up:
                # plt.scatter(obstacle.center_x,obstacle.center_y,color='coral',marker=6,linewidth=4) # marker 6 is an up arrow
                if is_first_up:
                    plt.scatter(obstacle.center_x,obstacle.center_y,color='coral',marker="^",linewidth=label_width,label="above") 
                    is_first_up = False
                else:
                    plt.scatter(obstacle.center_x,obstacle.center_y,color='coral',marker="^",linewidth=label_width)

            elif label == dir_label.down:
                if is_first_down:
                    plt.scatter(obstacle.center_x,obstacle.center_y,color='royalblue',marker="v",linewidth=label_width,label="below")
                    is_first_down = False # marker 7 is a down arrow
                else:
                    plt.scatter(obstacle.center_x,obstacle.center_y,color='royalblue',marker="v",linewidth=label_width)
            else:
                raise Exception('invalid label type')
            
    #these two methods were for generating plots for report, dont call them
    def plot_sub_plot(self,plot_name):
        fig, axs = plt.subplots(1,3,sharex=True, sharey=True, figsize=(10,4))
        self.is_plot_self = False #turn off plotting on self to make subplot
        plt.sca(axs[0])
        self._init_graph_props()
        self.plot_env(0,title='Initial env')
        plt.sca(axs[1])
        self._init_graph_props()
        self.plot_env(0,title="Initial conditions predicted \n by classifier")
        self.plot_labels(0)
        plt.sca(axs[2])
        self._init_graph_props()
        self.plot_solution(0,title="Shortest path \n from minimization")
        self.plot_labels(0)
        fig.savefig(plot_name + '.png')
        self.is_plot_self = True

    def _plot_4_pane_sub_plot(self,plot_name):
        # this is used to generate 4 pane image for reports
        def plot_guess(x,y):
            # plt.plot(x,y,linestyle='-.',color=(0.3010, 0.7450, 0.9330),label="guess")
            plt.plot(x,y,color=(0.3010, 0.7450, 0.9330),linewidth=self.line_width,label="guess")
        def plot_solution(x,y):
            plt.plot(x,y,color='purple',linewidth=self.line_width,label='solution')
        
        #loading data from matlab for making a specific plot
        data = sio.loadmat('for_graphic.mat')
        x_out = data['x_out'][0]
        y_span_guess = data['y_span_guess'][0]
        y_out = data['y_out'][0]
    
        fig, axs = plt.subplots(2,2,sharex=True, sharey=True, figsize=(10,10))
        self.is_plot_self = False #turn off plotting on self to make subplot
        plt.sca(axs[0,0])
        self._init_graph_props()
        self.plot_start_end(0)
        self.plot_obstacles(0)
        plt.title('1. Example Problem Formulation')
        # self.plot_env(0,title='1. Environment',loc='upper left')
        plt.sca(axs[0,1])
        self._init_graph_props()
        self.plot_start_end(0)
        self.plot_obstacles(0)
        plt.title('2. Direction labels predicted by classifier')
        # self.plot_env(0,title="2. Initial conditions (ICs) \npredicted by classifier",loc='upper left')
        self.plot_labels(0)
        plt.sca(axs[1,0])
        self._init_graph_props()
        # self.plot_env(0)
        self.plot_start_end(0)
        self.plot_obstacles(0)
        plt.title('3. Initial guess for optimizer \nbased on direction labels')
        self.plot_labels(0)
        plot_guess(x_out,y_span_guess)
        # self.finish_plot(title="3. Guess based on ICs",loc='upper left')
        plt.sca(axs[1,1])
        self._init_graph_props()
        # self.plot_solution(0,title="Shortest path \n from minimization")
        self.plot_env(0)
        self.plot_labels(0)
        plot_guess(x_out,y_span_guess)
        plot_solution(x_out,y_out)
        self.finish_plot(title="4. Global, shortest path found \nfrom minimization",loc='upper left')
        axs[1,1].legend(loc='lower center', bbox_to_anchor=(-0.2,-0.25), ncol=3)
        fig.savefig(plot_name + '.png')
        self.is_plot_self = True


    def clear_plot(self):
        plt.cla()
        self._init_graph_props()

    def save_plot_image(self,fig_name):
        '''creates .png of visibility graph'''
        self.fig.savefig(fig_name + '.png')

    def place_figure(self,location=(-1500,600)):
        self.fig.canvas.manager.window.move(*location)

    def _init_graph_props(self):
        self._act_fig()
        plt.xlim(self.axis_xlim)
        plt.ylim(self.axis_ylim)
        plt.grid(visible=True)
        ax = plt.gca()
        ax.set_aspect('equal')

    def _act_fig(self):
        '''This method makes sure correct figure is being plotted on'''
        if self.is_plot_self:
            plt.figure(self.fig.number)

class graph_viewer(visibility_graph_generator):
    # This class allows us to look at the vis graph before it is finished
    def __init__(self, vis_graph_obj, obstacles=None, record_on=True,is_ion=True):
        super().__init__(obstacles, record_on, is_ion) # this is needed so we can reuse plot methods from parent
        self.store_vis_graph(vis_graph_obj) 
        # self.previous_segment() # this is used only for debugging to track what the 

    #TODO create method that gets obstacle data here and in parent class

    def get_start_end_data(self, test_num=0):
        graph = self.graphs_memory[test_num]
        start = (graph.start.x, graph.start.y)
        end = (graph.end.x, graph.end.y)
        return start,end

    def plot_network(self, test_num=0):
        return super().plot_network(test_num)
    
    def store_last_plotted():
        '''store_last_plotted is used to store last thing plotted to prevent plot from replotting'''
        pass

    def reinit_vis_graph(self,vis_graph_obj):
        self.clear_graph_memory()
        self.store_vis_graph(vis_graph_obj)
        self._act_fig()
        self.clear_plot()
    
    def clear_graph_memory(self):
        self.graphs_memory = {}

    def plot_cand_edge(self,node1,node2):
        '''used for debugging new edges and plotting as they are found'''
        self._act_fig()
        x_points = [node1.x,node2.x]
        y_points = [node1.y,node2.y]
        plt.plot(x_points,y_points,color='red')
        
# global methods
def arg_parse():
    parser = ArgumentParser(description="obstacle testing file",formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-b", "--batch", type=bool, default = False, help="Creates unique file name and does not display courses")
    parser.add_argument("-p", "--base_path", default = "./obs_courses/",help="path to folder containing obstacle course files")
    parser.add_argument("-c","--course", type=int, default=0, help="specify obstacle course number when multiple courses available")
    parser.add_argument("-s", "--start", type=float, default = [0,3], nargs=2, help='course start point')
    parser.add_argument("-e", "--end", type=float, default = [30,15], nargs=2, help='course end point')
    parser.add_argument("-a", "--astar", dest='solve_option', action='store_const', const='AStar', default='dijkstra',help='Change shortest path solver from dijkstra to AStar')
    parser.add_argument("-t", "--path_test", dest='test_mode', action='store_const', const=True, default=False,help='Changes to test mode to compare dijkstra and AStar solutions.  Only works if batch mode if off.')
    parser.add_argument("-i", "--ion_on", dest='is_ion',action='store_const', const=True, default=False,help='initializes graph generator with plt.ion() enabling interactive mode')
    parser.add_argument("-gs","--graph_storage",dest='record_on',action='store_const',const=False,default=True,help="Turns off storage of graph objects in vis_obs generator")
    parser.add_argument("-f","--output_file", default = None, help="sets the output directory for generated data in batch mode")
    parser.add_argument("-nt","--no_title", dest='no_title',action='store_const', const=True, default=False,help='Turns off title for graphs outputted by function')
    parser.add_argument("fname", help="Obstacle course file to test")
    
    args = parser.parse_args()
    args = vars(args)
    args["start"] = [args["start"]] #run test expects start and end as list of points
    args["end"] = [args["end"]]
    args["obs_fpath"] = args["base_path"] + args["fname"]
    return args

# for reading new file list
def read_obstacle_list(fname):
    #TODO move this function to the obstacle_course_gen.py file
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

def create_obs_file_from_vgobj(vg_obj: vis_graph, fname: str):
    obstacle_data = [x.view() for x in vg_obj.obstacles]
    with open(fname+".txt",'w') as f:
        f.write("New Obstacle Set:\n")
        f.write(f"# obs = {len(obstacle_data)},\n")
        f.write("radius,x,y\n")
        for line in obstacle_data:
            f.write(','.join(str(x) for x in line)+'\n')

def calc_ndata_col(num_obs):
    return 4*num_obs + 5

def create_start_end(obstacle_list,npoints):
    '''Creates start and end lists for batch testing'''
    tol = 0.01
    num_points_x = npoints[0]
    num_points_y = npoints[1]
    min_x,max_x = find_test_range(obstacle_list)
    range_start = point((0,0))
    range_bound = point((30,30))

    if min_x <= 5 or max_x >= 25:
        raise Exception('Obstacles are not in x bounds (5,25)')
    
    # if we want dynamic bounds based on obstacle locations
    start_x = np.linspace(range_start.x,min_x-tol,num_points_x) # could try using np.arange
    start_y = np.linspace(range_start.y,range_bound.y,num_points_y)
    end_x = np.linspace(max_x+tol,range_bound.x,num_points_x)
    end_y = start_y
    # if we want fixed bounds
    # start_x = np.linspace(range_start.x,5,num_points)
    # start_y = np.linspace(range_start.y,range_bound.y,num_points)
    # end_x = np.linspace(25,range_bound.x,num_points)
    # end_y = start_y

    start_vals = get_list_points(start_x,start_y)
    end_vals = get_list_points(end_x,end_y)

    start_list = init_points(start_vals)
    end_list = init_points(end_vals)

    return start_list, end_list

def compare_solutions(sol1,sol2):
    '''returns if two lists are the same'''
    same = True
    for i1,i2 in zip(sol1,sol2):
        if i1 != i2:
            same = False
            break
    return same

def append_dict(dict_in,item):
    num_keys = len(dict_in)
    dict_in[num_keys] = item

# def remove_list_item(item,list_in):
#     #TODO this function Does not work dont use it
#     # removes obstacle from obstacle list without modifying original obstacle_list
#     new_obs_list = list_in[:]
#     new_obs_list.remove(item)
#     return new_obs_list

def check_collision(start_node,end_node,obstacle):
    '''checks if line collides with circle, check intersection can see if the line segments are intersecting'''
    # first check if the entire line defined by start_node,end_node intersects with the obstacle
    a,b,c = planar_line_form(start_node,end_node)
    x0 = obstacle.center_x
    y0 = obstacle.center_y
    radius = round(obstacle.radius,4)
    dist = round((abs(a * x0 + b * y0 + c)) / np.sqrt(a * a + b * b),4) # rounding for numerical errors; from https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    if radius <= dist:
        return False
    else:
        # if the line intersects the obstacle, now check if the intersection occurs in the segment between start_node and end_node
        x_col = (b*(b*x0-a*y0)-a*c) / (a*a + b*b)
        y_col = (a*(-b*x0+a*y0)-b*c) / (a*a + b*b)
        collision_point = point((x_col,y_col))
        if check_intersection((collision_point,obstacle.center_loc),(start_node,end_node)):
            collision = True
        else:
            collision = False
        return collision

def check_intersection(seg1,seg2):
    '''checks if two line segments are intersecting.  Line segment should be tuple of 2 point objects'''
    dbg = False
    # based on chapter 4, slide 172 of Lectures on Robotic Planning and Kinematics by Francessco Bullo and Stephen L. Smith available here http://motion.me.ucsb.edu/book-lrpk/
    x1,y1 = seg1[0].output()
    x2,y2 = seg1[1].output()
    x3,y3 = seg2[0].output()
    x4,y4 = seg2[1].output()
    tol = 0.00002

    a = y1-y3
    b = x4-x3
    c = y4-y3
    d = x1-x3
    e = x2-x1
    f = y2-y1

    Sa = (a*b-c*d)/(c*e-f*b)
    Sb = (a*e-d*f)/(e*c-f*b)
    A_in_range = in_range(Sa,(0,1),tol)
    B_in_range = in_range(Sb,(0,1),tol)
    if A_in_range and B_in_range:
        intersects = True
    else:
        intersects = False
    if dbg:
        print(f'Sa = {Sa}, in_range = {A_in_range}, tol = {tol}')
        print(f'Sb = {Sb}, in_range = {B_in_range}, tol = {tol}')

    return intersects

def in_range(var,bounds,tol):
    '''checks if variable is in range specified by tuple bounds(lower,higher), with given tol'''
    if bounds[0]-tol <= var <= bounds[1]+tol:
        return True
    else:
        return False

def slope_int_form(start_node,end_node):
    tol = 0.0000002
    y_diff = end_node.y-start_node.y
    x_diff = end_node.x-start_node.x
    if -tol <= x_diff <= tol:
        if y_diff < 0:
            slope = float('-inf')
        elif y_diff > 0:
            slope = float('inf')
        else:
            raise Exception("start_node and end_node are the same point")
        y_int = None
    else:
        slope = (end_node.y-start_node.y)/(end_node.x-start_node.x) 
        y_int = end_node.y-slope*end_node.x
 
    return slope, y_int

def planar_line_form(start_node,end_node):
    x1,y1 = start_node.output()
    x2,y2 = end_node.output()
    a = y1-y2
    b = x2-x1
    c = y1*x1-y1*x2+y2*x1-y1*x1
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
    prop_list = [obs.output_prop() for obs in obstacle_list] #note for some reason output_prop outputs obstacles in the fromat x,y,r instead of the r, x,y used elsewhere...
    obs_min_x = [prop[0]-prop[2] for prop in prop_list]
    obs_max_x = [prop[0]+prop[2] for prop in prop_list]
    return min(obs_min_x),max(obs_max_x)

def get_list_points(x,y):
    grid_x,grid_y = np.meshgrid(x,y)
    grid = np.vstack([grid_x.ravel(),grid_y.ravel()])
    grid = grid.tolist()
    grid = list(zip(*grid))
    return grid

def create_directory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print(f"Directory '{directory_name}' created successfully.")