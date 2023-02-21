import pandas as pd
import os
import glob

def get_file_list(folder):
    '''Only returns names of csv files in folder'''
    # base_path = 'C:/Users/Robert/Documents/Vis_network_data/'
    # # base_path = 'C:/Users/Robert/git/visibility_NN/results_merge/'
    path = base_path + folder
    extension = 'csv'
    os.chdir(path)
    result = glob.glob('*.{}'.format(extension))
    return result

def get_dir_list(path):
    result = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path,name))]
    return result

def csv_reader(fname):
    for row in open(fname):
        yield row

def data_check(data_gen,ncolumns):
    '''checks if file described by data_gen has correct number of columns, ncolumns'''
    valid_file = True
    ii = 0
    for row in data_gen:
        ii += 1
        tokens = row.split(',')
        if len(tokens) != ncolumns:
            valid_file = False
            print(f'row {ii} is invalid with {len(tokens)} tokens')
    print(f'nrows = {ii}')
    return valid_file

def _name_output(csv_folder):
    if ".csv" in csv_folder:
        output_file = f"{merge_out_folder}{csv_folder[0:21]}_merge.csv"
    else:
        output_file = f"{merge_out_folder}{csv_folder}_merge.csv"
    return output_file

def _chunk_and_output(csv_file_list,biggest_file=None):
    
    for csv_file_name in csv_file_list:
        if csv_file_name == biggest_file:
            print('biggest file encountered')
            continue
        else:
            chunk_container = pd.read_csv(csv_file_name, chunksize=CHUNK_SIZE)
            for chunk in chunk_container:
                chunk.to_csv(output_file, mode="a", index=False,header=False)

# User options
CHUNK_SIZE = 50000
dir_mode = False
merge_out_folder = "C:/Users/Robert/git/visibility_NN/results_merge/"

make_new_file = True # TODO: this mode doesnt work, and is activated by changing to false... this mode creates a new file for outputting.  Use this if all the files are approx the same size
base_path = 'C:/Users/Robert/Documents/Vis_network_data/'
# base_path = 'C:/Users/Robert/git/visibility_NN/results_merge/'
biggest_file = "file1.csv"
biggest_file_path = base_path + biggest_file


csv_folder = '23_02_20_aws_batchb'
# csv_folder = '23_02_19_aws_batch1_0_course_1_obs_data.csv[+13]'
# '23_02_18_batch2'
# 
# csv_folder = 'Test'
# if make_new_file:
if dir_mode:
    dir_list = get_dir_list('C:/Users/Robert/Documents/Vis_network_data')
    for csv_folder in dir_list:
        print(csv_folder)
    #     csv_file_list = get_file_list(csv_folder)
    #     # csv_file_list = ["file1.csv", "file2.csv", "file3.csv"]
    #     # output_file = f"./results_merge/{csv_folder}_merge.csv"
    #     if ".csv" in csv_folder:
    #         output_file = f"{merge_out_folder}{csv_fol   der[0:21]}_merge.csv"
    #     else:
    #         output_file = f"{merge_out_folder}{csv_folder}_merge.csv"
        # for csv_file_name in csv_file_list:
        #     chunk_container = pd.read_csv(csv_file_name, chunksize=CHUNK_SIZE)
        #     for chunk in chunk_container:
        #         chunk.to_csv(output_file, mode="a", index=False,header=False)
        output_file = _name_output(csv_folder)
        csv_file_list = get_file_list(csv_folder)
        _chunk_and_output(csv_file_list)
else:
    print(csv_folder)
    # csv_file_list = get_file_list(csv_folder)
    # # csv_file_list = ["file1.csv", "file2.csv", "file3.csv"]
    # # output_file = f"./results_merge/{csv_folder}_merge.csv"
    # if ".csv" in csv_folder:
    #     output_file = f"{merge_out_folder}{csv_folder[0:21]}_merge.csv"
    # else:
    #     output_file = f"{merge_out_folder}{csv_folder}_merge.csv"
    # for csv_file_name in csv_file_list:
    #     chunk_container = pd.read_csv(csv_file_name, chunksize=CHUNK_SIZE)
    #     for chunk in chunk_container:
    #         chunk.to_csv(output_file, mode="a", index=False, header=False)
    output_file = _name_output(csv_folder)
    csv_file_list = get_file_list(csv_folder)
    _chunk_and_output(csv_file_list)
# else:
#     '''This mode appends the other files to the biggest file to save time'''
#     output_file = biggest_file_path
#     csv_file_list = get_file_list(csv_folder)
#     _chunk_and_output(csv_file_list,biggest_file)
    
    

