import visibility_graph as vg
from matplotlib import pyplot as plt

## This script allows you to preview a course

obs_file = r"./obs_courses/1_courses_5_obstacles_normal.txt"

course_num = 0
g_title = f"course {course_num}"
obs_dict = vg.read_obstacle_list(obs_file)
obs_list = obs_dict[course_num]
vis_viewer = vg.visibility_graph_generator()
graph = vg.vis_graph(obs_list)
graph.make_obs_vis_graph()
vis_viewer.store_vis_graph(graph)
vis_viewer.plot_just_obstacles(0,title=g_title)
plt.show()
