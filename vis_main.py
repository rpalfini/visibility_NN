# from visibility_graph import *
import visibility_graph as vg
from matplotlib import pyplot as plt
import os
import time
import WindowsInhibitor as wi #prevents computer from sleeping while generating data
from datetime import date
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# example python vis_main.py 1_courses_5_obstacles_normal.txt 
args = vg.arg_parse()

#TODO fully add batch code which involves loading start/end values, maybe make it a function in vis_graph where it can generate a range of start/end values we want to test  
# batch = convert2bool(args["batch"])
batch = args["batch"]

if batch:
    osSleep = None
    if os.name == 'nt':
        osSleep = wi.WindowsInhibitor()
        osSleep.inhibit()

# obs_file_path = args["obs_path"] + args["fname"]
obs_file_path = args["obs_fpath"]
obs_courses_dict = vg.read_obstacle_list(obs_file_path)
obstacle_list = obs_courses_dict[args["course"]]

tic = time.perf_counter()
if batch:
    # this mode tests multiple courses
    npoints = (10,30)
    # npoints = (5,5)
    
    for ii in range(len(obs_courses_dict)):
        print(f'Testing course {ii} out of {len(obs_courses_dict)}')
        obstacle_list = obs_courses_dict[ii]
        start_list, end_list = vg.create_start_end(obstacle_list,npoints)
        vg_gen = vg.visibility_graph_generator(record_on=args["record_on"],is_ion=args["is_ion"])
        vg_gen.run_test(start_list,end_list,obstacle_list,algorithm=args["solve_option"])
        
        #output the results
        file_title = args["fname"].replace('.txt','')
        vg_gen.output_csv(f'{file_title}_course_{ii+1}_obs_data')

else:
    # start_vals = [(0,3)]
    # end_vals = [(30,15)]
    start_vals = args["start"]
    end_vals = args["end"]
    # create start/end points
    start_list = vg.init_points(start_vals)
    end_list = vg.init_points(end_vals)

    if args["test_mode"] and not batch:
        # this mode solves one course with dijkstra and AStar and compares their solutions
        vg_gen = vg.visibility_graph_generator(is_ion=args["is_ion"])
        vg_gen.run_test(start_list,end_list,obstacle_list,algorithm="dijkstra")
        vg_gen.run_test(start_list,end_list,obstacle_list,algorithm="AStar")
        if vg.compare_solutions(vg_gen.graphs_memory[0].opt_path,vg_gen.graphs_memory[1].opt_path):
            print('solution PATH is SAME')
        else:
            print('solution PATH is DIFFERENT')
            print(f'd = {vg_gen.graphs_memory[0].opt_path}')
            print(f'a = {vg_gen.graphs_memory[1].opt_path}')
        if vg_gen.graphs_memory[0].opt_path_cost == vg_gen.graphs_memory[1].opt_path_cost:
            print('solution COST is SAME')
        else:
            print('solution COST is DIFFERENT')
            print(f'd = {vg_gen.graphs_memory[0].opt_path_cost}, a* = {vg_gen.graphs_memory[1].opt_path_cost}')
        vg_gen.plot_solution(0,"dijkstra")
        vg_gen.plot_solution(1,"AStar")
    else:
        # this mode means we are just testing one course and getting an output
        vg_gen = vg.visibility_graph_generator(record_on=args["record_on"],is_ion=args["is_ion"])
        vg_gen.run_test(start_list,end_list,obstacle_list,algorithm=args["solve_option"])
        g_title = f"course {args['fname']}"
        vg_gen.plot_solution(0,g_title)
        
        #output the results
        file_title = args["fname"].replace('.txt','')
        vg_gen.output_csv(f'{file_title}_obs_data')
        vg_gen.save_plot_image(f'{file_title}_obs_fig')

toc = time.perf_counter()
print(f"created the data in {toc - tic:0.4f} seconds")

# file_title = args["fname"].replace('.txt','')
# vg_gen.output_csv(f'{file_title}_obs_data')
# vg_gen.save_plot_image(f'{file_title}_obs_fig')

# today = date.today()
# vg_gen.output_csv(today.strftime("%Y_%m_%d")+'three_obst data_large_1')

if batch:
    if osSleep:
        osSleep.uninhibit()

