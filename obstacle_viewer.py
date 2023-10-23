import visibility_graph as vg
from matplotlib import pyplot as plt

## This script allows you to preview a course


args = vg.arg_parse()
obs_file = args["obs_fpath"]
course_num = args["course"]

g_title = f"course {course_num}"
obs_dict = vg.read_obstacle_list(obs_file)
obs_list = obs_dict[course_num]
vis_viewer = vg.visibility_graph_generator(is_ion=args["is_ion"])
graph = vg.vis_graph(obs_list)
graph.make_obs_vis_graph()
vis_viewer.store_vis_graph(graph)
# vis_viewer.plot_just_obstacles(0,title=g_title)
vis_viewer.plot_just_obstacles(0)
plt.show()
