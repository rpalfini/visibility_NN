# visibility_NN

## Desciption
This repo creates tangent visibility graph for data generation and implements a navigation decision NN for obstacle path finding.

It creates data files where each row describes a test performed and the optimal labels found for a set of obstacles given a start and end point.  Each row corresponds to one test.  One test is outputted in the form: 

[start_x, start_y,end_x,end_y,

obstacle_1_center_x, obstacle_1_center_y, obstacle_1_radius through obstacle_n_center_x, obstacle_n_center_y, 

obstacle_n_radius, obstacle_1_label through obstacle_n_label].

Meaning for the labels values is defined in vis_graph_enum.py but a down label corresponds to 0 and an up label corresponds to 1. 

**Dependencies:**
1. pip install bidict
2. numpy
3. pandas (not used currently)
4. Scipy
5. llook at requirements.txt

### Other capabilities
1. vis_main.py is main script to create data files for various obstacle courses.  -h option explains how to use
2. Random obstacle courses can be generated with obstacle_course_gen.py. -h option explains how to use
3. obstacle_viewer.py allows preview of obstacle course generated.  -h option has not been added yet, so just open file and change var to path to obstacle

### Sample Usage
python vis_main.py 1_courses_10_obstacles_normal.txt
### Open Items

(person working on task)
1. ~~Update code to work with multiple obstacles by implementing vis_graph.vis_obst_obst() (Robby)~~
2. Remove obstacles from vis_graph_generator attributes
3. ~~Update dijkstra and priodict to pop keys based on cost + euclidean distance heuristic (i.e. A*)~~
4. Delayed edge generation based on A* _stretch goal_
5. Many #TODOS
6. Implement NN for navigation prediction for one obstacle (Robby)
7. ~~Update vis_graph.build_vis_graph() to generate obstacle to obstacle surfing edges once before surfing edges between start and end point are evaluated.  Ideally when a new vis_graph is initialized, it should have obstacle objects that already have the surfing edge nodes and associated graph attributes initialized correctly.~~
8. Create test file with test_cases needed to verify the shortest_path calculation is working correctly. (I have been manually testing during implementation but do not have a file that can test if full algorithm is working with test cases)
9. Create obstacle superclass to generalize obstacle types, enabling us to use other shapes besides circles _stretch goal_
10. Adding functionality to enable overlapping obstacles
11. Any other helpful features mentioned in [circular obstacle pathfinding guide](https://redblobgames.github.io/circular-obstacle-pathfinding/)
12. Adding profiler to speed up data generation.
13. Testing if removing plot object from generator improves performance. Would move plot functionality to child class.
14. ~~Update path finding to deal with edges with infinite slope.~~
15. ~~Add code to prevent computer from sleeping until data generated.~~
16. ~~Code to combine created csv data files.~~
17. Implement NN for navigation prediction for multiple obstacles. (Robby)
18. ~~Add ability to save data as pickle files~~
19. ~~Update plotting methods to take the graph object as input, so that I can call the plotting methods during debugging by just calling with a graph object. (Robby)~~
20. ~~Fix bugs that occur when two points have same x coordinate~~
21. Move plot_network to parent class
22. Update vis_graph algorithm to allow start point in front of obstacles on the x-axis
23. ~~Fix collision check to check for obstacles vertically and horizontally.  Can we only consider obstacle centers or do we need to check their extrema?~~
24. ~~Change plot handles from subplots to just plots.~~
25. ~~Finish adding batch mode in vis_main.py~~
