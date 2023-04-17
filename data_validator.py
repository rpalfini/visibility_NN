from visibility_graph import calc_ndata_col
import csv_file_combiner as cfc
import os
from argparse import ArgumentParser

def arg_parse():
    parser = ArgumentParser(description="script checks files in a folder to either verify they are valid or fix based on option selected.  Combines multiple csv data files into one file")
    parser.add_argument("csv_folder_path",help="csv folder to check csv files in")
    parser.add_argument("-f", "--fix", dest='fix_data', action='store_const', const=True, default=False,help="changes validator to fix data files in folder")
    parser.add_argument("-n","--num_obs", type=int, default=20,help="set the max number of obstacles expected in the data set")
    args = parser.parse_args()
    args = vars(args)
    return args

if __name__ == "__main__":
    '''Use this to verify data file is valid and doesnt have any misformatted data'''
    args = arg_parse()
    num_obs = args["num_obs"]
    csv_folder_path = args["csv_folder_path"]

    csv_file_list = cfc.get_file_list(csv_folder_path)

    is_all_valid = True
    ii = 0
    num_data = 0
    if not args["fix_data"]: #TODO implement fix_data mode
        for csv_file in csv_file_list:
            file_path = os.path.join(csv_folder_path,csv_file)
            is_valid,nrows,total_rows = cfc.data_check(file_path,calc_ndata_col(args["num_obs"]))
            if not is_valid:
                print(f'file {file_path} has {nrows} rows of invalid data')
                is_all_valid = False
            print(f"{csv_file} has {total_rows} rows of data")
            num_data += total_rows
    print(f'{csv_folder_path} data is_all_valid = {is_all_valid}')
    print(f'{csv_folder_path} has {num_data} rows of data')


