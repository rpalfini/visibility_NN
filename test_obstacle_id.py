import visibility_graph as vg
import numpy as np
import copy

# testing if identical obstacles can be identified by id

start_vals = [(1,9)]
end_vals = [(29,9)]

# locs = [(13,10),(12,4),(20,16),(9,15),(7,7),(21,2.5),(19,9),(25,13)]
obst_locations = [[12,10],[11,4],[19,16],[8,15],[6,7],[24,6],[18,9],[24,13]]
radius1 = 2*np.ones(len(obst_locations)) #obstacle radius

start_list = vg.init_points(start_vals)
end_list = vg.init_points(end_vals)

# create obstacle list
obstacle_list = vg.init_obs(obst_locations,radius1)

graph = vg.vis_graph(copy.deepcopy(obstacle_list))

# print(id(x)) for x in graph.obstacles]
foo = [id(x) for x in graph.obstacles]

for item in foo:
    for item2 in foo:
        if item == item2:
            print(f'matching id found {item} and {item2}')

print(len(foo))