import visibility_graph as vg

foo = set()

start_vals = [(2,3)]
end_vals = [(24,14)]

obst_locations = [(6,5),(13,9),(20,6)]
radius1 = [2,3,2] #obstacle radius

start_list = vg.init_points(start_vals)
end_list = vg.init_points(end_vals)

# create obstacle list
obstacle_list = vg.init_obs(obst_locations,radius1)


graph = vg.vis_graph(obstacle_list)
list2 = graph.obstacles

for obst in graph.obstacles:
    # [print(f'obst in set2 {x.view()} with id {id(x)}') for x in set2]
    # print(f'obst to remove {obst.view()} with id {id(obst)}')
    # set2.remove(obst)
    # [print(f'obst in set2 {x.view()} with id {id(x)}') for x in set2]
    # print('')
    [print(f'obst in set2 with id {id(x)}') for x in set2]
    print(f'obst to remove {obst.view()} with id {id(obst)}')
    list2.remove(obst)
    [print(f'obst in set2 with id {id(x)}') for x in set2]
    print('')




print('end')