import pandas as pd
import os
import glob

def get_file_list(folder):

    base_path = 'C:/Users/Robert/Documents/Vis_network_data/'
    path = base_path + folder
    extension = 'csv'
    os.chdir(path)
    result = glob.glob('*.{}'.format(extension))
    return result

CHUNK_SIZE = 50000
csv_folder = '23_02_19_aws_batchb'
# csv_folder = 'Test'
csv_file_list = get_file_list(csv_folder)
# csv_file_list = ["file1.csv", "file2.csv", "file3.csv"]
# output_file = f"./results_merge/{csv_folder}_merge.csv"
output_file = f"C:/Users/Robert/git/visibility_NN/results_merge/{csv_folder}_merge.csv"


for csv_file_name in csv_file_list:
    chunk_container = pd.read_csv(csv_file_name, chunksize=CHUNK_SIZE)
    for chunk in chunk_container:
        chunk.to_csv(output_file, mode="a", index=False)