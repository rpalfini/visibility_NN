import os
import pickle
from matplotlib import pyplot as plt
import visibility_graph as vg
## This script allows you to preview a course

def add_title(args,title):
    args["title"] = title
    return args

def load_default_args():
    with open('default_args.pickle', 'rb') as f:
        args = pickle.load(f)
    return args

def replace_file_tested(args,file_to_test):
    args["fname"] = os.path.basename(file_to_test)
    args["obs_fpath"] = file_to_test
    return args

def main(args):
    obs_file = args["obs_fpath"]
    course_num = args["course"]
    g_title = f"course {course_num}"
    obs_dict = vg.read_obstacle_list(obs_file)
    obs_list = obs_dict[course_num]
    vis_viewer = vg.visibility_graph_generator(is_ion=args["is_ion"])
    graph = vg.vis_graph(obs_list)
    graph.make_obs_vis_graph()
    vis_viewer.store_vis_graph(graph)
    if "title" in args:
        vis_viewer.plot_just_obstacles(0,args["title"])
    else:    
        vis_viewer.plot_just_obstacles(0)
        # vis_viewer.plot_just_obstacles(0,title=g_title)
    plt.show()

if __name__ == "__main__":
    args = vg.arg_parse()
    main(args)
    