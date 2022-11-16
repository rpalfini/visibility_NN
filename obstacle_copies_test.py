from visibility_graph import *

test_num = 2
print(f'test_num is {test_num}')
# testing how obstacle copies and references work for nested objects

radius1 = [6]
obst_locations = [(13,10)]
obstacle_list = init_obs(obst_locations,radius1)

if test_num == 0:
    # this is case where node is added to other graph unintentionally
    graph1 = vis_graph(obstacle_list)
    graph2 = vis_graph(obstacle_list)
    graph1.obstacles[0].add_node(point([1,2]))
    graph2.obstacles[0].add_node(point([2,4]))

elif test_num == 1:
    # deep copies of obstacle list prevent unintential node additions in other graphs
    graph1 = vis_graph(copy.deepcopy(obstacle_list))
    graph2 = vis_graph(copy.deepcopy(obstacle_list))
    graph1.obstacles[0].add_node(point([1,2]))
    graph2.obstacles[0].add_node(point([2,4]))

elif test_num == 2:
    # matches the order that code is assigned
    # we can see we dont need to make deep copy of obstacle list when we init base_graph
    base_graph = vis_graph(obstacle_list)
    base_graph.obstacles[0].add_node(point([5,6]))

    graph1 = copy.deepcopy(base_graph)
    graph1.obstacles[0].add_node(point([1,2]))
    graph2 = copy.deepcopy(base_graph)
    graph2.obstacles[0].add_node(point([2,4]))

elif test_num == 3:
    base_graph = vis_graph(copy.deepcopy(obstacle_list))
    base_graph.obstacles[0].add_node(point([5,6]))

    graph1 = copy.deepcopy(base_graph)
    graph1.obstacles[0].add_node(point([1,2]))
    graph2 = copy.deepcopy(base_graph)
    graph2.obstacles[0].add_node(point([2,4]))

else:
    print("you are too old to party")

graph1.obstacles[0].view_nodes()
graph2.obstacles[0].view_nodes()

print('end')