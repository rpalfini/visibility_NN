import visibility_graph as vg
from matplotlib import pyplot as plt

args = vg.arg_parse()

obs_file_path = args["obs_path"] + args["fname"]
obs_courses_dict = vg.read_obstacle_list(obs_file_path)
obstacle_list = obs_courses_dict[int(args["course"])]

vg_gen = vg.visibility_graph_generator()
