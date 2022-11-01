# visibility_NN

This repo creates tangent visibility graph for data generation and implements a navigation decision NN for obstacle path finding.

It creates data files outputted in the form [start_x, start_y,end_x,end_y, obstacle_n_center_x, obstacle_n_center_y, obstacle_n_radius, obstacle_n_label], where we have 1:n obstacles.
Label meaning is defined in vis_graph_enum.py

**Dependencies:**
1. pip install bidict
2. numpy
3. pandas (not used currently)

**Open Items:**

(person working on task)
1. Update code to work with multiple obstacles by implementing vis_graph.vis_obst_obst() (Robby)
2. Remove obstacles from vis_graph_generator attributes
3. Update dijkstra and priodict to pop keys based on cost + euclidean distance heuristic (i.e. A*)
4. Delayed edge generation based on A* _stretch goal_
5. Many #TODOS
6. Implement NN for navigation prediction for one obstacle (Robby)
7. Update vis_graph.build_vis_graph() to generate obstacle to obstacle surfing edges once before surfing edges between start and end point are evaluated.  Ideally when a new vis_graph is initialized, it should have obstacle objects that already have the surfing edge nodes and associated graph attributes initialized correctly.
8. Create test file with test_cases needed to verify the shortest_path calculation is working correctly. (I have been manually testing during implementation but do not have a file that can test if full algorithm is working with test cases)
9. Create obstacle superclass to generalize obstacle types, enabling us to use other shapes besides circles _stretch goal_
10. Adding functionality to enable overlapping obstacles
11. Any other helpful features mentioned in [circular obstacle pathfinding guide](https://redblobgames.github.io/circular-obstacle-pathfinding/)
12. Adding profiler to speed up data generation.
13. Testing if removing plot object from generator improves performance. Would move plot functionality to child class.
14. Update path finding to deal with edges with infinite slope.
15. ~~Add code to prevent computer from sleeping until data generated.~~
16. Code to combine created csv data files.
17. Implement NN for navigation prediction for multiple obstacles.