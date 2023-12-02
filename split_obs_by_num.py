import math
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm

import csv_file_combiner as cfc

debug = False

def divide_line_by_entries(line):
    '''divides comma separated file and converts entries from string to float'''
    tokens = line.split(',')
    tokens = list(map(float,tokens))
    return tokens

def find_first_label_idx(max_num_obs,end_idx):
    first_label_idx = end_idx+max_num_obs*3+1
    return first_label_idx

def determine_num_obs(line,max_num_obs,end_idx):
    '''determines number of obstacles by counting how many sets of 0,0,0 are on the input starting from last place of obstacles to first place of obstacles in data file'''
    tokens = divide_line_by_entries(line)
    start_idx = find_first_label_idx(max_num_obs,end_idx)-1
    zero_counter = 0
    null_obs_idx = (None,None)
    for ii in range(start_idx,end_idx,-1):
        if tokens[ii] == 0:
            zero_counter += 1
        else:
            # if we encounter a non zero that means we are done with place holder obstacles
            null_obs_start_idx = ii+1
            null_obs_end_idx = start_idx+1
            null_obs_idx = (null_obs_start_idx,null_obs_end_idx)
            break
    
    num_empty_obs = math.floor(zero_counter/3)
    num_obs = max_num_obs - num_empty_obs
    if debug:
        print(f'num_empty_obs={num_empty_obs}, zero_counter % 3 = {zero_counter%3}, num_obs = {num_obs}')
    return num_obs, null_obs_idx

def find_null_label_idx(num_obs,max_num_obstacles,end_idx):
    first_label_idx = find_first_label_idx(max_num_obstacles,end_idx)
    # num_empty_obs = max_num_obstacles-num_obs
    first_null_label_idx = first_label_idx + num_obs
    end_null_label_idx = first_label_idx + max_num_obstacles
    null_label_idx = (first_null_label_idx,end_null_label_idx)
    return null_label_idx

def resize_line(line,num_obs,max_num_obstacles,null_obs_idx,null_label_idx):
    '''Use to remove the empty obstacle entries and resize data'''
    if num_obs < max_num_obstacles:
        tokens = divide_line_by_entries(line) 
        del tokens[null_label_idx[0]:null_label_idx[1]] # delete null labels
        del tokens[null_obs_idx[0]:null_obs_idx[1]] # delete null obstacles
        expected_num_tokens = 4+num_obs*3+num_obs+1
        actual_num_tokens = len(tokens)
        if actual_num_tokens != expected_num_tokens:
            raise Exception('new line has incorrect number of tokens')
        new_line = ','.join(str(x) for x in tokens) + "\n"
    elif num_obs == max_num_obstacles:
        new_line = line
    else:
        raise Exception('num_obstacles > max_num_obstacles')
    return new_line

if __name__ == "__main__":
    resize = False # makes outputted files not have the padded obstacles and labels to make the data fit the 20 size framework
    max_num_obs = 20
    first_obstacle_idx = 4
    fpath = "D:/Vis_network_data/data_file_by_course_padded"
    fname = "main_data_file.csv"
    # fname_no_extension = os.path.splitext(fname)[0] # want to append to our existing data file
    fname_no_extension = "main_data_file"
    # create a dictionary of file handles for the 20 output files
    output_files = {}
    num_lines_file = {}
    total_lines_processed = 0
    total_zero_lines = 0
    for i in range(1,max_num_obs+1):
        filename = f"{fname_no_extension}_courses{i}.csv"
        output_files[i] = open(os.path.join(fpath,filename), "a")
        num_lines_file[i] = 0

    # read lines from the input file
    # file_path = os.path.join("./results_merge",fname)
    # file_path = "D:/Vis_network_data/to_be_added_to_main_file/23_03_11_merge/23_03_11-03_14_merge.csv"
    file_path = "D:/Vis_network_data/main_data_file/4_17_main_data_file.csv"
    data_gen = cfc.csv_reader(file_path)
    for row in tqdm(data_gen, desc="Processing", unit="row"):
        num_obs, null_obs_idx = determine_num_obs(row,max_num_obs,first_obstacle_idx-1)
        if num_obs == 0:
            total_zero_lines += 1
            # print(f'found zero line at {total_lines_processed}')
            continue
        
        if resize:
            null_label_idx = find_null_label_idx(num_obs,max_num_obs,first_obstacle_idx-1)
            new_row = resize_line(row,num_obs,max_num_obs,null_obs_idx,null_label_idx)
            output_files[num_obs].write(new_row)
            num_lines_file[num_obs] += 1
        else:
            output_files[num_obs].write(row)
            num_lines_file[num_obs] += 1
        total_lines_processed += 1

    # close all output files
    print(f'Total Lines Processed = {total_lines_processed}')
    print(f'Zero Lines Found = {total_zero_lines}')
    for file_handle in output_files.values():
        file_handle.close()
        
