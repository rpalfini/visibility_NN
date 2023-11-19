import pandas as pd
import os
import glob
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from ML_code import util

'''This module contains code for working with large csv files created from data generation.'''


def append_file_csv():
    pass

def get_file_list(folder_path):
    '''Only returns names of csv files in folder'''
    # base_path = 'C:/Users/Robert/Documents/Vis_network_data/'
    # # base_path = 'C:/Users/Robert/git/visibility_NN/results_merge/'
    cur_dir = os.getcwd()
    # path = base_path + folder
    extension = 'csv'
    os.chdir(folder_path)
    result = glob.glob('*.{}'.format(extension))
    os.chdir(cur_dir)
    return result

def get_dir_list(path):
    result = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path,name))]
    return result

def csv_reader(fname):
    for row in open(fname):
        yield row

def row_counter(data_gen):
    '''Counts rows in a generator.  Data_gen should be a generator to a data file created by csv_reader()'''
    row_count = 0
    for row in data_gen:
        row_count += 1
    return row_count

def count_rows_in_dir(dir_path):
    total_rows = 0
    csv_files = get_file_list(dir_path)
    for csv_file in csv_files:
        row_count = 0
        with open(dir_path + f'/{csv_file}','r') as file:
            row_count = file.readlines()
            # for line in file:
            #     row_count += 1
        # row_gen = csv_reader(dir_path + f'/{csv_file}')
        # row_count = row_counter(row_gen)
        total_rows += len(row_count)
    return total_rows

def other_count_rows_in_dir(dir_path):
    total_rows = 0
    csv_files = get_file_list(dir_path)
    for csv_file in csv_files:
        row_count = 0
        with open(dir_path + f'/{csv_file}','r') as file:
            # row_count = file.readlines()
            for line in file:
                row_count += 1
        # row_gen = csv_reader(dir_path + f'/{csv_file}')
        # row_count = row_counter(row_gen)
        total_rows += row_count
    return total_rows

def data_check(file_path,ncolumns):
    '''checks if every row of file in file_path has ncolumns.'''
    data_gen = csv_reader(file_path)
    valid_file = True
    ii = 0
    n_invalid_rows = 0
    for row in data_gen:
        ii += 1
        tokens = row.split(',')
        if len(tokens) != ncolumns:
            valid_file = False
            print(f'row {ii} is invalid with {len(tokens)} tokens')
        for token in tokens:
            if token == '':
                # print(f'row {ii} is invalid')
                n_invalid_rows+=1
                break
    # print(f'nrows = {ii}')
    return valid_file,n_invalid_rows,ii

def calculate_num_columns(num_obstacles):
    num_feat = util.calc_num_features(num_obstacles)
    num_labels = num_obstacles
    num_extra_columns = 1 # this is because we are storing the optimal path cost found as the last row
    num_columns = num_feat + num_labels + num_extra_columns
    return num_columns

def remove_invalid_data(data_file_path):
    '''input should be absolute or relative path to the data file including the data file i.e. C:/Users/dog.csv'''
    data_file, data_path = split_fname_path(data_file_path)
    data_gen = csv_reader(data_file_path)
    output_file = data_file_path.rstrip(".csv") + "_fixed.csv"

    f = open(output_file,"a")
    n_invalid_rows = 0
    invalid_rows = []
    ii=0
    for row in data_gen:
        is_row_bad = False
        ii+=1
        tokens = row.split(",")
        for token in tokens:
            if token == '':
                n_invalid_rows += 1
                invalid_rows.append(ii)
                is_row_bad = True
                break
        if not is_row_bad:
            f.write(row)
    f.close()
    print(f'outputted corrected data file at {output_file}')
    print(f'{n_invalid_rows} invalid rows at rows: {invalid_rows}')
    is_valid,n_invalid_rows,total_rows = data_check(output_file,85)
    if not is_valid:
        raise Exception("Fixed file is invalid/has errors")
    else:
        print(f'{output_file} is fixed and valid')

def count_zero_lines(file_path):
    file_gen = csv_reader(file_path)
    num_zero_lines = 0
    for line in file_gen:
        if all(map(lambda x: float(x.strip()) == 0.0, line.split(','))):
            num_zero_lines += 1
    return num_zero_lines

def split_fname_path(data_path):
    tokens = data_path.split('/')
    fname = tokens[-1]
    fpath = "/".join(tokens[:-1])
    fpath += "/"
    return fname,fpath

def print_data_window(file_path,target_row,back_win,forward_win):
    ii = 0
    file_gen = csv_reader(file_path)
    for row in file_gen:
        ii+=1
        if ii >= (target_row-back_win) and ii <= (target_row+forward_win):
            print(row)
        
        if ii == target_row:
            print(row)
        
        if ii > target_row+forward_win:
            break


def _name_output(csv_folder,merge_out_folder):
    # if ".csv" in csv_folder:
    #     output_file = f"{merge_out_folder}/{csv_folder[0:21]}_merge.csv" # I dont remember why i have this code
    # else:
    output_file = f"{merge_out_folder}/{csv_folder}_merge.csv"
    return output_file

def _chunk_and_output(csv_file_list,csv_folder_path,output_file,biggest_file=None):
    
    for csv_file_name in csv_file_list:
        if csv_file_name == biggest_file:
            print('biggest file encountered')
            continue
        else:
            csv_fpath = csv_folder_path + "/" + csv_file_name
            chunk_container = pd.read_csv(csv_fpath, chunksize=CHUNK_SIZE, header=None)
            for chunk in chunk_container:
                chunk.to_csv(output_file, mode="a", index=False,header=False)

# def combin
def combine_csv(csv_folder):
    print(csv_folder)

    output_file = _name_output(csv_folder)
    csv_file_list = get_file_list(csv_folder)
    _chunk_and_output(csv_file_list,output_file)

def arg_parse():
    parser = ArgumentParser(description="csv file combiner.  Combines multiple csv data files into one file")
    parser.add_argument("csv_folder",help="path to csv folder to combine csv files in")
    parser.add_argument("-a","--append", default=None,help="specifies a file to append csv_folder data to instead of make a new folder")
    parser.add_argument("-o","--outputfolder", default="./results_merge",help="sets the folder for merged file to be outputted to.  This is not used if append mode is selected.")
    args = parser.parse_args()
    args = vars(args)
    return args


if __name__ == "__main__":
    # User options
    args = arg_parse()
    CHUNK_SIZE = 50000
    dir_mode = False # Leave as False
    # merge_out_folder = "C:/Users/Robert/git/visibility_NN/results_merge/"
    merge_out_folder = args["outputfolder"]

    # make_new_file = True # TODO: this mode doesnt work, and is activated by changing to false... this mode creates a new file for outputting.  Use this if all the files are approx the same size
    # base_path = 'C:/Users/Robert/Documents/Vis_network_data/'
    # base_path = 'C:/Users/Robert/git/visibility_NN/results_merge/'
    # base_path = 'C:/Users/Robert/git/visibility_NN/'
    # base_path = './data_out/'

    # csv_folder = '23_02_18_19_20'
    csv_folder = os.path.basename(os.path.normpath(args["csv_folder"])) #gets the name of the csv_folder
    if args["append"] != None:
        append_mode = True
    else:
        append_mode = False
 
    dir_exists = os.path.isdir(merge_out_folder)
    if not dir_exists:
        os.mkdir(merge_out_folder)

    if dir_mode:
        dir_list = get_dir_list('C:/Users/Robert/Documents/Vis_network_data')
        for csv_folder in dir_list:
            print(csv_folder)

            output_file = _name_output(csv_folder,merge_out_folder)
            folder_path = args["csv_folder"]
            csv_file_list = get_file_list(csv_folder,output_file)
            _chunk_and_output(csv_file_list)
    
    elif append_mode:
        print(csv_folder)
        output_file = args["append"]
        folder_path = args["csv_folder"]
        csv_file_list = get_file_list(folder_path)
        _chunk_and_output(csv_file_list,folder_path,output_file)
    
    else:
        print(csv_folder)
        output_file = _name_output(csv_folder,merge_out_folder)
        folder_path = args["csv_folder"]
        csv_file_list = get_file_list(folder_path)
        _chunk_and_output(csv_file_list,folder_path,output_file)

        

