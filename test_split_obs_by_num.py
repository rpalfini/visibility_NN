import csv_file_combiner as cfc
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import math


def determine_num_obs(line,max_num_obs,end_idx=3):
    '''determines number of obstacles by counting how many sets of 0,0,0 are on the input starting from last place of obstacles to first place of obstacles in data file'''
    max_num_obs = 20
    tokens = line.split(',')
    tokens = list(map(float,tokens))
    start_idx = end_idx+max_num_obs*3
    zero_counter = 0
    for ii in range(start_idx,end_idx,-1):
        if tokens[ii] == 0:
            zero_counter += 1
        else:
            # if we encounter a non zero that means we are done with place holder obstacles
            break
    num_empty_obs = math.floor(zero_counter/3)
    num_obs = max_num_obs - num_empty_obs
    print(f'num_empty_obs = {num_empty_obs}')
    print(f'num_empty_obs % 3 = {num_empty_obs%3}')
    print(f'num_obs = {num_obs}')

    return num_obs

def resize_row():
    pass

if __name__ == "__main__":
    resize = True
    fname = "single_course_set_merge.csv"
    fname_no_extension = os.path.splitext(fname)[0]
    # create a dictionary of file handles for the 20 output files
    output_files = {}
    for i in range(1, 21):
        filename = f"{fname_no_extension}_courses{i}.csv"
        output_files[i] = open(filename, "a")

    # read lines from the input file
    file_path = os.path.join("./results_merge",fname)
    data_gen = cfc.csv_reader(file_path)
    for row in data_gen:
        num_obs = determine_num_obs(row)
        if resize:
            new_row = resize_row(row)
        else:
            output_files[num_obs].write(row)

    # close all output files
    for file_handle in output_files.values():
        file_handle.close()
