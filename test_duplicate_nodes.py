from visibility_graph import *
import copy

# this script needs to test if we add the same nodes multiple times
# to an obstacle, is that bad and is the vis_graph ok


#TODO: I need to modify obstacle so that it doesnt add more nodes to obstacle than there are in the node dictionary

test_num = 2
print(f'test_num is {test_num}')

radius1 = [6]
obst_locations = [(13,10)]
obstacle_list = init_obs(obst_locations,radius1)
start = point([3,4])
end = point((100,100))
cand_point = point((13.3434242341,4))
cand_point2 = point((13.3434242342,4))

if test_num == 0:
    graph1 = vis_graph(copy.deepcopy(obstacle_list))
    graph1.obstacles[0].add_node(cand_point)
    graph1.obstacles[0].add_node(cand_point2)
elif test_num == 1:
    graph1 = vis_graph(copy.deepcopy(obstacle_list))
    graph1.init_start_end(start,end)
    graph1.process_cand_node(start,cand_point,graph1.obstacles[0])
    graph1.process_cand_node(start,cand_point,graph1.obstacles[0])
elif test_num == 2:
    graph1 = vis_graph(copy.deepcopy(obstacle_list))
    graph1.init_start_end(start,end)
    graph1.process_cand_node(start,cand_point,graph1.obstacles[0])
    graph1.process_cand_node(start,cand_point2,graph1.obstacles[0])


graph1.obstacles[0].view_nodes()
print('node_dict items')
[print(item) for item in graph1.node_dict.inverse]
[print(item) for item in graph1.node_dict]
graph1.view_vis_graph()

print('end')



