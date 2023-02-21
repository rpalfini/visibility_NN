import pandas as pd
import os
import glob

def get_file_list(folder):

    # base_path = 'C:/Users/Robert/Documents/Vis_network_data/'
    base_path = 'C:/Users/Robert/git/visibility_NN/results_merge/'
    path = base_path + folder
    extension = 'csv'
    os.chdir(path)
    result = glob.glob('*.{}'.format(extension))
    return result

def get_dir_list(path):
    result = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path,name))]
    return result


CHUNK_SIZE = 50000

dir_list = get_dir_list('C:/Users/Robert/Documents/Vis_network_data')
dir_mode = False

csv_folder = '23_02_19'
# csv_folder = '23_02_19_aws_batch1_0_course_1_obs_data.csv[+13]'
# '23_02_18_batch2'
# 
# csv_folder = 'Test'

if dir_mode:

    for csv_folder in dir_list:

        csv_file_list = get_file_list(csv_folder)
        # csv_file_list = ["file1.csv", "file2.csv", "file3.csv"]
        # output_file = f"./results_merge/{csv_folder}_merge.csv"
        if ".csv" in csv_folder:
            output_file = f"C:/Users/Robert/git/visibility_NN/results_merge/{csv_folder[0:21]}_merge.csv"
        else:
            output_file = f"C:/Users/Robert/git/visibility_NN/results_merge/{csv_folder}_merge.csv"

        ii = 0
        for csv_file_name in csv_file_list:
            ii += 1
            chunk_container = pd.read_csv(csv_file_name, chunksize=CHUNK_SIZE)
            for chunk in chunk_container:
                chunk.to_csv(output_file, mode="a", index=False,header=False)
        print(csv_folder)

else:
    csv_file_list = get_file_list(csv_folder)
    # csv_file_list = ["file1.csv", "file2.csv", "file3.csv"]
    # output_file = f"./results_merge/{csv_folder}_merge.csv"
    if ".csv" in csv_folder:
        output_file = f"C:/Users/Robert/git/visibility_NN/results_merge/{csv_folder[0:21]}_merge.csv"
    else:
        output_file = f"C:/Users/Robert/git/visibility_NN/results_merge/{csv_folder}_merge.csv"
    ii = 0
    for csv_file_name in csv_file_list:
        ii += 1
        chunk_container = pd.read_csv(csv_file_name, chunksize=CHUNK_SIZE)
        for chunk in chunk_container:
            chunk.to_csv(output_file, mode="a", index=False, header=False)
    print(csv_folder)