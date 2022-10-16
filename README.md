# visibility_NN

This repo creates tangent visibility graph for data generation and implements a navigation decision NN for obstacle path finding.

It creates data files outputted in the form [start_x, start_y,end_x,end_y, obstacle_n_center_x, obstacle_n_center_y, obstacle_n_radius, obstacle_n_label]

where we have 1:n obstacles.

Label meaning is defined in vis_graph_enum.py