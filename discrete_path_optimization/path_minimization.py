import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sp
import min_utils as utils
import plotting
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatters

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../visibility_NN")
import visibility_graph as vg



def short_dist_multi_obs(obs_file):
    obstacles = vg.read_obstacle_list(obs_file)
    


if __name__ == "__main__":
    short_dist_multi_obs()